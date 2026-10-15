import csv
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from simulation.config import ModelConfig
from simulation.model import Model

RESULTS_DIRECTORY = Path(__file__).resolve().parent.parent / "results"
TABLE_FILE = RESULTS_DIRECTORY / "verification.csv"
PROTOCOL_FILE = RESULTS_DIRECTORY / "event_protocol.txt"
SWITCH_PROTOCOL_FILE = RESULTS_DIRECTORY / "event_protocol_switch.txt"

DETERMINISTIC_WARMUP_MS = 0.0
DETERMINISTIC_SIMULATION_MS = 100000.0
STOCHASTIC_WARMUP_MS = 5000.0
STOCHASTIC_SIMULATION_MS = 200000.0
STOCHASTIC_REPLICATIONS = 10
SEED_STRIDE = 100

PROTOCOL_EVENT_COUNT = 40
SWITCH_CONTEXT_EVENTS = 6
SWITCH_SEARCH_EVENTS = 5000
PROTOCOL_SEED = 1
PROTOCOL_SIMULATION_MS = 5000.0

ABSOLUTE_TOLERANCE = 1e-6
UTILIZATION_TOLERANCE = 5e-3

HEADER = [
    "Сценарій",
    "Змінені входи",
    "T_warm, мс",
    "T_mod, мс",
    "Прогонів",
    "P_зн",
    "Знищень, пак/с",
    "Підключень, 1/с",
    "Підключень, шт",
    "Відключень, шт",
    "Частка часу R=1",
    "L1, пакетів",
    "L2, пакетів",
    "U1",
    "U2",
    "Середній час перебування, мс",
    "Очікування",
    "Пройдено",
]

OUTPUT_KEYS = [
    "destroyed_fraction",
    "destroyed_per_second",
    "activations_per_second",
    "activation_count",
    "deactivation_count",
    "accelerated_time_fraction",
    "mean_queue_length_1",
    "mean_queue_length_2",
    "utilization_1",
    "utilization_2",
    "mean_sojourn_time",
]


@dataclass(frozen=True)
class VerificationScenario:
    key: str
    label: str
    changed_inputs: str
    overrides: dict
    expectation: str
    check: Callable
    deterministic: bool = True

    @property
    def warmup_time(self):
        return self.overrides.get(
            "warmup_time", DETERMINISTIC_WARMUP_MS if self.deterministic else STOCHASTIC_WARMUP_MS
        )

    @property
    def simulation_time(self):
        return self.overrides.get(
            "simulation_time", DETERMINISTIC_SIMULATION_MS if self.deterministic else STOCHASTIC_SIMULATION_MS
        )

    @property
    def replications(self):
        return 1 if self.deterministic else STOCHASTIC_REPLICATIONS

    def build_config(self, seed):
        return ModelConfig(
            **{
                "warmup_time": self.warmup_time,
                "simulation_time": self.simulation_time,
                **self.overrides,
                "seed": seed,
            }
        )


def constant_arrivals(interval_ms, **extra):
    return {"arrival_distribution": "constant", "arrival_mean": interval_ms, **extra}


def is_close(value, expected, tolerance=ABSOLUTE_TOLERANCE):
    return abs(value - expected) <= tolerance


def check_dd1_baseline(result, all_results):
    return (
        is_close(result["mean_sojourn_time"], 10.0)
        and result["destroyed_fraction"] == 0.0
        and is_close(result["utilization_1"], 5.0 / 6.0, UTILIZATION_TOLERANCE)
        and is_close(result["utilization_2"], 5.0 / 6.0, UTILIZATION_TOLERANCE)
        and is_close(result["mean_queue_length_1"], 0.0)
        and is_close(result["mean_queue_length_2"], 0.0)
        and result["activation_count"] == 0
    )


def check_overload_adaptive(result, all_results):
    return (
        result["destroyed_fraction"] > 0.99
        and result["activation_count"] >= 1
        and result["accelerated_time_fraction"] > 0.9
        and result["mean_queue_length_1"] > 1.0
    )


def check_overload_never(result, all_results):
    return (
        result["destroyed_fraction"] > 0.99
        and result["activation_count"] == 0
        and result["mean_queue_length_1"] > 100.0
        and all_results["overload_never"]["mean_queue_length_1"]
        > all_results["overload_adaptive"]["mean_queue_length_1"]
    )


def check_light_load(result, all_results):
    return (
        result["destroyed_fraction"] == 0.0
        and result["activation_count"] == 0
        and is_close(result["mean_sojourn_time"], 10.0)
    )


def check_threshold_above_one(result, all_results):
    return (
        result["activation_count"] == 0
        and result["accelerated_time_fraction"] == 0.0
        and is_close(result["destroyed_fraction"], all_results["mode_never"]["destroyed_fraction"], 0.02)
    )


def check_threshold_above_one_overload(result, all_results):
    return result["activation_count"] == 0 and result["destroyed_fraction"] > 0.99


def check_threshold_zero(result, all_results):
    return (
        result["activation_count"] == 1.0
        and result["deactivation_count"] == 0.0
        and result["accelerated_time_fraction"] > 0.98
    )


def check_window_one(result, all_results):
    base_rate = all_results["mode_adaptive"]["activations_per_second"]
    return result["activations_per_second"] > 2.0 * base_rate and result["activation_count"] > 0


def check_mode_ordering(result, all_results):
    always = all_results["mode_always"]["destroyed_fraction"]
    adaptive = all_results["mode_adaptive"]["destroyed_fraction"]
    never = all_results["mode_never"]["destroyed_fraction"]
    return always < adaptive < never


def check_faster_service(result, all_results):
    base = all_results["mode_adaptive"]
    return (
        result["destroyed_fraction"] < base["destroyed_fraction"]
        and result["activations_per_second"] < base["activations_per_second"]
        and result["utilization_1"] < base["utilization_1"]
    )


def check_slower_service(result, all_results):
    base = all_results["mode_adaptive"]
    return (
        result["destroyed_fraction"] > base["destroyed_fraction"]
        and result["utilization_1"] > base["utilization_1"]
    )


def check_higher_destruction_threshold(result, all_results):
    base = all_results["mode_adaptive"]
    return result["destroyed_fraction"] < base["destroyed_fraction"]


def check_lower_destruction_threshold(result, all_results):
    base = all_results["mode_adaptive"]
    return result["destroyed_fraction"] > base["destroyed_fraction"]


def check_threshold_edge_inclusive(result, all_results):
    return result["destroyed_fraction"] == 0.0 and is_close(result["mean_sojourn_time"], 10.0)


def check_threshold_edge_below(result, all_results):
    return result["destroyed_fraction"] == 1.0


SCENARIOS = [
    VerificationScenario(
        "dd1_baseline",
        "D/D/1: інтервал 6 мс, обслуговування 5 мс, керування вимкнене",
        "arrival=constant 6; control=never",
        constant_arrivals(6.0, control_mode="never"),
        "T=10 мс, знищень 0, U=5/6, L=0",
        check_dd1_baseline,
    ),
    VerificationScenario(
        "threshold_edge_inclusive",
        "D/D/1: поріг знищення 10 мс, T=10 мс рівно (допуск 1e-9)",
        "arrival=constant 6; control=never; поріг знищення=10",
        constant_arrivals(6.0, control_mode="never", destruction_threshold=10.0),
        "T=10 не перевищує поріг: P_зн=0",
        check_threshold_edge_inclusive,
    ),
    VerificationScenario(
        "threshold_edge_below",
        "D/D/1: поріг знищення 9.9 мс < T=10 мс",
        "arrival=constant 6; control=never; поріг знищення=9.9",
        constant_arrivals(6.0, control_mode="never", destruction_threshold=9.9),
        "Кожен пакет має T=10>9.9: P_зн=1",
        check_threshold_edge_below,
    ),
    VerificationScenario(
        "overload_adaptive",
        "Перевантаження: інтервал 4 мс < 5 мс, адаптивне керування",
        "arrival=constant 4; control=adaptive",
        constant_arrivals(4.0, control_mode="adaptive"),
        "Черга росте, P_зн майже 1, ресурс вмикається й залишається",
        check_overload_adaptive,
    ),
    VerificationScenario(
        "overload_never",
        "Перевантаження: інтервал 4 мс < 5 мс, без керування",
        "arrival=constant 4; control=never",
        constant_arrivals(4.0, control_mode="never"),
        "Черга необмежено росте (L1>100, більше ніж із ресурсом), P_зн майже 1, підключень 0",
        check_overload_never,
    ),
    VerificationScenario(
        "light_load_constant",
        "Мале навантаження: інтервал 12 мс (constant)",
        "arrival=constant 12; control=adaptive",
        constant_arrivals(12.0, control_mode="adaptive"),
        "Знищень 0, ресурс не вмикається, T=10 мс",
        check_light_load,
    ),
    VerificationScenario(
        "light_load_uniform",
        "Мале навантаження: інтервал unif(9;15), центр 12 мс",
        "arrival=uniform 9..15; control=adaptive",
        {"arrival_min": 9.0, "arrival_max": 15.0, "arrival_mean": 12.0, "control_mode": "adaptive"},
        "Знищень 0, ресурс не вмикається, T=10 мс",
        check_light_load,
    ),
    VerificationScenario(
        "threshold_above_one_base",
        "Поріг ресурсу θ=1.01, базовий B8",
        "θ=1.01",
        {"activation_threshold": 1.01},
        "Підключень 0; P_зн як у режимі «без керування»",
        check_threshold_above_one,
        deterministic=False,
    ),
    VerificationScenario(
        "threshold_above_one_overload",
        "Поріг ресурсу θ=1.01, інтервал 4 мс",
        "θ=1.01; arrival=constant 4",
        constant_arrivals(4.0, activation_threshold=1.01),
        "Підключень 0 навіть при P_зн≈1",
        check_threshold_above_one_overload,
    ),
    VerificationScenario(
        "threshold_zero",
        "Поріг ресурсу θ=0, без прогріву",
        "θ=0; T_warm=0",
        {"activation_threshold": 0.0, "warmup_time": 0.0},
        "Рівно одне підключення після заповнення вікна, відключень 0, частка часу R=1 близька до 1",
        check_threshold_zero,
        deterministic=False,
    ),
    VerificationScenario(
        "window_one",
        "Вікно N=1",
        "N=1",
        {"window_size": 1},
        "Дуже часті перемикання: частота підключень суттєво (понад 2 рази) вища, ніж при N=100",
        check_window_one,
        deterministic=False,
    ),
    VerificationScenario(
        "mode_adaptive",
        "Базовий B8: адаптивне керування, N=100, θ=0.30",
        "control=adaptive",
        {"control_mode": "adaptive"},
        "Порядок P_зн: always < adaptive < never",
        check_mode_ordering,
        deterministic=False,
    ),
    VerificationScenario(
        "mode_always",
        "Базовий B8: ресурс завжди ввімкнений (4 мс)",
        "control=always",
        {"control_mode": "always"},
        "Порядок P_зн: always < adaptive < never; U≈4/6",
        check_mode_ordering,
        deterministic=False,
    ),
    VerificationScenario(
        "mode_never",
        "Базовий B8: ресурс вимкнений (5 мс)",
        "control=never",
        {"control_mode": "never"},
        "Порядок P_зн: always < adaptive < never; U≈5/6",
        check_mode_ordering,
        deterministic=False,
    ),
    VerificationScenario(
        "service_faster",
        "Швидше обслуговування: 4 мс / 3 мс (норма / прискорене)",
        "service=4/3",
        {"service_time_normal": 4.0, "service_time_accelerated": 3.0},
        "P_зн, частота підключень і U1 менші, ніж у базового B8",
        check_faster_service,
        deterministic=False,
    ),
    VerificationScenario(
        "service_slower",
        "Повільніше обслуговування: 5.5 мс / 4.5 мс (норма / прискорене)",
        "service=5.5/4.5",
        {"service_time_normal": 5.5, "service_time_accelerated": 4.5},
        "P_зн і U1 більші, ніж у базового B8",
        check_slower_service,
        deterministic=False,
    ),
    VerificationScenario(
        "destruction_threshold_higher",
        "Поріг знищення 12 мс",
        "поріг знищення=12",
        {"destruction_threshold": 12.0},
        "P_зн менше, ніж у базового B8",
        check_higher_destruction_threshold,
        deterministic=False,
    ),
    VerificationScenario(
        "destruction_threshold_lower",
        "Поріг знищення 8 мс",
        "поріг знищення=8",
        {"destruction_threshold": 8.0},
        "P_зн більше, ніж у базового B8",
        check_lower_destruction_threshold,
        deterministic=False,
    ),
]


def run_single_replication(scenario, seed):
    return Model(scenario.build_config(seed)).simulate().to_dict()


def mean_outputs(replications):
    return {key: float(np.mean([replication[key] for replication in replications])) for key in OUTPUT_KEYS}


def run_all_scenarios(executor):
    futures = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        base_seed = SEED_STRIDE * (scenario_index + 1)
        futures[scenario.key] = [
            executor.submit(run_single_replication, scenario, base_seed + replication)
            for replication in range(scenario.replications)
        ]
    return {
        key: mean_outputs([future.result() for future in scenario_futures])
        for key, scenario_futures in futures.items()
    }


def build_rows(all_results):
    rows = []
    for scenario in SCENARIOS:
        result = all_results[scenario.key]
        passed = bool(scenario.check(result, all_results))
        rows.append({
            "scenario": scenario,
            "result": result,
            "passed": passed,
        })
    return rows


def table_cells(row):
    scenario = row["scenario"]
    result = row["result"]
    return [
        scenario.label,
        scenario.changed_inputs,
        f"{scenario.warmup_time:.0f}",
        f"{scenario.simulation_time:.0f}",
        scenario.replications,
        f"{result['destroyed_fraction']:.4f}",
        f"{result['destroyed_per_second']:.3f}",
        f"{result['activations_per_second']:.3f}",
        f"{result['activation_count']:.1f}",
        f"{result['deactivation_count']:.1f}",
        f"{result['accelerated_time_fraction']:.4f}",
        f"{result['mean_queue_length_1']:.3f}",
        f"{result['mean_queue_length_2']:.3f}",
        f"{result['utilization_1']:.4f}",
        f"{result['utilization_2']:.4f}",
        f"{result['mean_sojourn_time']:.3f}",
        scenario.expectation,
        "так" if row["passed"] else "ні",
    ]


def save_table(rows):
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    with open(TABLE_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(table_cells(row))


def print_table(rows):
    print("Верифікація моделі: детерміновані прогони (1 прогін), стохастичні (середнє за "
          f"{STOCHASTIC_REPLICATIONS} прогонів)")
    print(f"  детерміновані: T_warm = {DETERMINISTIC_WARMUP_MS:.0f} мс, T_mod = {DETERMINISTIC_SIMULATION_MS:.0f} мс")
    print(f"  стохастичні:   T_warm = {STOCHASTIC_WARMUP_MS:.0f} мс, T_mod = {STOCHASTIC_SIMULATION_MS:.0f} мс "
          "(якщо не вказано інакше)")
    print()
    header = (f"{'Ключ':<30} {'T_warm':>7} {'T_mod':>8} {'P_зн':>7} {'зн/с':>8} {'підкл/с':>8} "
              f"{'підкл':>6} {'R=1':>6} {'L1':>9} {'L2':>6} {'U1':>6} {'U2':>6} {'T, мс':>9}  Пройдено")
    print(header)
    for row in rows:
        scenario = row["scenario"]
        result = row["result"]
        print(f"{scenario.key:<30} {scenario.warmup_time:>7.0f} {scenario.simulation_time:>8.0f} "
              f"{result['destroyed_fraction']:>7.4f} {result['destroyed_per_second']:>8.2f} "
              f"{result['activations_per_second']:>8.3f} {result['activation_count']:>6.1f} "
              f"{result['accelerated_time_fraction']:>6.3f} {result['mean_queue_length_1']:>9.2f} "
              f"{result['mean_queue_length_2']:>6.2f} {result['utilization_1']:>6.3f} "
              f"{result['utilization_2']:>6.3f} {result['mean_sojourn_time']:>9.2f}  "
              f"{'так' if row['passed'] else 'НІ'}")
    passed_count = sum(1 for row in rows if row["passed"])
    print()
    print(f"Пройдено сценаріїв: {passed_count} із {len(rows)}")


def protocol_config(event_limit):
    return ModelConfig(
        seed=PROTOCOL_SEED,
        warmup_time=0.0,
        simulation_time=PROTOCOL_SIMULATION_MS,
        record_protocol=True,
        protocol_limit=event_limit,
    )


def protocol_title(text):
    return (f"{text}\nБазовий B8: інтервал unif(3;9) мс, обслуговування 5/4 мс, поріг знищення 10 мс, N=100, θ=0.30, "
            f"seed={PROTOCOL_SEED}, T_warm=0 мс, T_mod={PROTOCOL_SIMULATION_MS:.0f} мс\n\n")


def write_text(path, text):
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as file:
        file.write(text)


def save_event_protocols():
    first_events_model = Model(protocol_config(PROTOCOL_EVENT_COUNT))
    first_events_model.simulate()
    write_text(
        PROTOCOL_FILE,
        protocol_title(f"Протокол перших {PROTOCOL_EVENT_COUNT} подій") + first_events_model.format_protocol() + "\n",
    )

    switch_model = Model(protocol_config(SWITCH_SEARCH_EVENTS))
    switch_model.simulate()
    switch_positions = [index for index, entry in enumerate(switch_model.protocol) if entry.resource_switch]
    if not switch_positions:
        write_text(SWITCH_PROTOCOL_FILE, protocol_title("Перемикання ресурсу не знайдено в межах протоколу") + "\n")
        return None
    first_switch = switch_positions[0]
    start = max(0, first_switch - SWITCH_CONTEXT_EVENTS)
    end = first_switch + SWITCH_CONTEXT_EVENTS + 1
    switch_model.protocol = switch_model.protocol[start:end]
    switch_step = switch_model.protocol[first_switch - start].step
    write_text(
        SWITCH_PROTOCOL_FILE,
        protocol_title(f"Фрагмент протоколу навколо першого перемикання ресурсу (подія №{switch_step})")
        + switch_model.format_protocol()
        + "\n",
    )
    return switch_step


def main():
    started = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        all_results = run_all_scenarios(executor)
    rows = build_rows(all_results)
    save_table(rows)
    print_table(rows)
    switch_step = save_event_protocols()
    print()
    print(f"Таблиця: {TABLE_FILE}")
    print(f"Протокол перших {PROTOCOL_EVENT_COUNT} подій: {PROTOCOL_FILE}")
    print(f"Фрагмент першого перемикання (подія №{switch_step}): {SWITCH_PROTOCOL_FILE}")
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
