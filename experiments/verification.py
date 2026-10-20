import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from experiments.common import parallel_map, results_path, write_csv, write_text
from simulation.config import ModelConfig
from simulation.transmission_model import MILLISECONDS_PER_SECOND, TransmissionModel

TABLE_FILE = "verification.csv"
PROTOCOL_FILE = "event_protocol.txt"
SWITCH_PROTOCOL_FILE = "event_protocol_switch.txt"

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
LITTLE_LAW_RELATIVE_TOLERANCE = 1e-3
DESTRUCTION_REQUIREMENT = 0.30

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
    "max L1, пакетів",
    "max L2, пакетів",
    "W1, мс",
    "W2, мс",
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
    "max_queue_length_1",
    "max_queue_length_2",
    "mean_waiting_time_1",
    "mean_waiting_time_2",
    "utilization_1",
    "utilization_2",
    "mean_sojourn_time",
    "throughput_per_second",
    "generated",
    "exited",
    "destroyed",
    "discarded",
    "in_system",
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
        and result["max_queue_length_1"] == 0
        and result["max_queue_length_2"] == 0
        and result["activation_count"] == 0
    )


def check_arrival_after_channel_1(result, all_results):
    return (
        result["max_queue_length_1"] == 0
        and result["max_queue_length_2"] == 0
        and result["destroyed_fraction"] == 0.0
        and is_close(result["mean_sojourn_time"], 10.0)
        and is_close(result["utilization_1"], 1.0, UTILIZATION_TOLERANCE)
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
        and result["max_queue_length_1"] == 0
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


def check_hysteresis(result, all_results):
    base = all_results["mode_adaptive"]
    switch_balance = result["activation_count"] - result["deactivation_count"]
    return (
        result["activations_per_second"] < base["activations_per_second"]
        and result["accelerated_time_fraction"] > base["accelerated_time_fraction"]
        and result["destroyed_fraction"] < base["destroyed_fraction"]
        and abs(switch_balance) <= 1.0
    )


def check_mode_adaptive(result, all_results):
    always = all_results["mode_always"]["destroyed_fraction"]
    never = all_results["mode_never"]["destroyed_fraction"]
    switch_balance = result["activation_count"] - result["deactivation_count"]
    return (
        always < result["destroyed_fraction"] < never
        and result["destroyed_fraction"] <= DESTRUCTION_REQUIREMENT
        and 0.0 < result["accelerated_time_fraction"] < 1.0
        and abs(switch_balance) <= 1.0
    )


def check_mode_always(result, all_results):
    return (
        result["activation_count"] == 0
        and result["accelerated_time_fraction"] == 1.0
        and is_close(result["utilization_1"], 4.0 / 6.0, 2 * UTILIZATION_TOLERANCE)
        and is_close(result["utilization_2"], 4.0 / 6.0, 2 * UTILIZATION_TOLERANCE)
    )


def check_mode_never(result, all_results):
    return (
        result["activation_count"] == 0
        and result["accelerated_time_fraction"] == 0.0
        and is_close(result["utilization_1"], 5.0 / 6.0, 2 * UTILIZATION_TOLERANCE)
        and is_close(result["utilization_2"], 5.0 / 6.0, 2 * UTILIZATION_TOLERANCE)
    )


def check_channel_2_queue_never_forms(result, all_results):
    return (
        result["max_queue_length_2"] == 0
        and result["mean_queue_length_2"] == 0.0
        and result["mean_waiting_time_2"] == 0.0
        and result["activation_count"] > 0
    )


def check_little_law(result, all_results):
    arrival_rate_per_ms = result["throughput_per_second"] / MILLISECONDS_PER_SECOND
    little_queue_length = arrival_rate_per_ms * result["mean_waiting_time_1"]
    return abs(result["mean_queue_length_1"] - little_queue_length) <= (
        LITTLE_LAW_RELATIVE_TOLERANCE * result["mean_queue_length_1"]
    )


def check_flow_conservation(result, all_results):
    return is_close(result["generated"], result["exited"] + result["in_system"], 1e-9)


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
    return result["destroyed_fraction"] < all_results["mode_adaptive"]["destroyed_fraction"]


def check_lower_destruction_threshold(result, all_results):
    return result["destroyed_fraction"] > all_results["mode_adaptive"]["destroyed_fraction"]


def check_early_discard_deterministic(result, all_results):
    return (
        is_close(result["destroyed_fraction"], 0.5, 1e-3)
        and result["discarded"] == result["destroyed"]
        and is_close(result["mean_sojourn_time"], 10.0)
        and result["mean_waiting_time_1"] == 0.0
        and result["max_queue_length_1"] <= 1
    )


def check_early_discard_never(result, all_results):
    return (
        result["destroyed_fraction"] < all_results["mode_never"]["destroyed_fraction"]
        and result["discarded"] == result["destroyed"]
        and is_close(result["mean_sojourn_time"], 10.0)
        and result["mean_waiting_time_1"] == 0.0
    )


def check_early_discard_adaptive(result, all_results):
    return (
        result["destroyed_fraction"] < all_results["mode_adaptive"]["destroyed_fraction"]
        and result["accelerated_time_fraction"] < all_results["mode_adaptive"]["accelerated_time_fraction"]
        and 0 < result["discarded"] <= result["destroyed"]
    )


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
        "T=10 мс, знищень 0, U=5/6, черги не утворюються",
        check_dd1_baseline,
    ),
    VerificationScenario(
        "priority_arrival_after_channel_1",
        "Пріоритет: надходження одночасно з кінцем обслуговування в K1 (інтервал 5 мс = обслуговуванню)",
        "arrival=constant 5; control=never",
        constant_arrivals(5.0, control_mode="never"),
        "K1 звільняється раніше, ніж обробляється надходження: max L1=0, U1=1, T=10 мс",
        check_arrival_after_channel_1,
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
        "Знищень 0, ресурс не вмикається, черги немає, T=10 мс",
        check_light_load,
    ),
    VerificationScenario(
        "light_load_uniform",
        "Мале навантаження: інтервал unif(9;15), центр 12 мс",
        "arrival=uniform 9..15; control=adaptive",
        {"arrival_min": 9.0, "arrival_max": 15.0, "arrival_mean": 12.0, "control_mode": "adaptive"},
        "Знищень 0, ресурс не вмикається, черги немає, T=10 мс",
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
        "hysteresis_30_20",
        "Гістерезис: вмикання при ≥30%, вимикання при <20%",
        "θ_on=0.30; θ_off=0.20",
        {"activation_threshold": 0.30, "deactivation_threshold": 0.20},
        "Підключень рідше, ресурс увімкнений довше, P_зн менше, ніж без гістерезису; |підкл − відкл| ≤ 1",
        check_hysteresis,
        deterministic=False,
    ),
    VerificationScenario(
        "mode_adaptive",
        "Базовий B8: адаптивне керування, N=100, θ=0.30",
        "control=adaptive",
        {"control_mode": "adaptive"},
        "always < P_зн < never; P_зн ≤ 0.30; 0 < частка часу R=1 < 1; |підкл − відкл| ≤ 1",
        check_mode_adaptive,
        deterministic=False,
    ),
    VerificationScenario(
        "mode_always",
        "Базовий B8: ресурс завжди ввімкнений (4 мс)",
        "control=always",
        {"control_mode": "always"},
        "Підключень 0, частка часу R=1 дорівнює 1, U≈4/6",
        check_mode_always,
        deterministic=False,
    ),
    VerificationScenario(
        "mode_never",
        "Базовий B8: ресурс вимкнений (5 мс)",
        "control=never",
        {"control_mode": "never"},
        "Підключень 0, частка часу R=1 дорівнює 0, U≈5/6",
        check_mode_never,
        deterministic=False,
    ),
    VerificationScenario(
        "priority_channel_2_before_channel_1",
        "Пріоритет K2 → K1: рішення контролера на виході застосовується до наступного запуску K1",
        "control=adaptive",
        {"control_mode": "adaptive"},
        "Черга K2 не утворюється попри перемикання ресурсу: max L2=0, W2=0",
        check_channel_2_queue_never_forms,
        deterministic=False,
    ),
    VerificationScenario(
        "little_law",
        "Закон Літтла для черги K1: L1 = λ·W1",
        "control=adaptive",
        {"control_mode": "adaptive"},
        "|L1 − λ·W1| ≤ 0.1% від L1",
        check_little_law,
        deterministic=False,
    ),
    VerificationScenario(
        "flow_conservation",
        "Баланс потоку без прогріву: створено = вийшло + у системі",
        "control=adaptive; T_warm=0",
        {"control_mode": "adaptive", "warmup_time": 0.0},
        "Жоден пакет не губиться і не дублюється",
        check_flow_conservation,
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

SCENARIOS += [
    VerificationScenario(
        "early_discard_deterministic",
        "Раннє відкидання: інтервал 4 мс, без ресурсу — відкидається кожен другий пакет",
        "arrival=constant 4; control=never; early_discard",
        constant_arrivals(4.0, control_mode="never", early_discard=True),
        "P_зн = 0.5; усі знищені відкинуто до K1; доставлені мають T = 10 мс без очікування",
        check_early_discard_deterministic,
    ),
    VerificationScenario(
        "early_discard_never",
        "Раннє відкидання в базовому B8 без ресурсу",
        "control=never; early_discard",
        {"control_mode": "never", "early_discard": True},
        "P_зн менше, ніж у never; усі знищені відкинуто, доставлені мають T = 10 мс",
        check_early_discard_never,
        deterministic=False,
    ),
    VerificationScenario(
        "early_discard_adaptive",
        "Раннє відкидання в базовому B8 з керуванням ресурсом",
        "control=adaptive; early_discard",
        {"control_mode": "adaptive", "early_discard": True},
        "P_зн і частка часу R=1 менші, ніж у базового B8",
        check_early_discard_adaptive,
        deterministic=False,
    ),
]

SCENARIOS_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}


def run_single_replication(task):
    scenario_key, seed = task
    return scenario_key, TransmissionModel(SCENARIOS_BY_KEY[scenario_key].build_config(seed)).simulate().to_dict()


def replication_tasks():
    return [
        (scenario.key, SEED_STRIDE * (scenario_index + 1) + replication)
        for scenario_index, scenario in enumerate(SCENARIOS)
        for replication in range(scenario.replications)
    ]


def run_all_scenarios():
    replications_by_key = {scenario.key: [] for scenario in SCENARIOS}
    for scenario_key, result in parallel_map(run_single_replication, replication_tasks()):
        replications_by_key[scenario_key].append(result)
    return {
        key: {output: float(np.mean([result[output] for result in results])) for output in OUTPUT_KEYS}
        for key, results in replications_by_key.items()
    }


def build_rows(all_results):
    return [
        {"scenario": scenario, "result": all_results[scenario.key],
         "passed": bool(scenario.check(all_results[scenario.key], all_results))}
        for scenario in SCENARIOS
    ]


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
        f"{result['max_queue_length_1']:.1f}",
        f"{result['max_queue_length_2']:.1f}",
        f"{result['mean_waiting_time_1']:.3f}",
        f"{result['mean_waiting_time_2']:.3f}",
        f"{result['utilization_1']:.4f}",
        f"{result['utilization_2']:.4f}",
        f"{result['mean_sojourn_time']:.3f}",
        scenario.expectation,
        "так" if row["passed"] else "ні",
    ]


def print_table(rows):
    print("Верифікація моделі: детерміновані прогони (1 прогін), стохастичні (середнє за "
          f"{STOCHASTIC_REPLICATIONS} прогонів)")
    print(f"  детерміновані: T_warm = {DETERMINISTIC_WARMUP_MS:.0f} мс, T_mod = {DETERMINISTIC_SIMULATION_MS:.0f} мс")
    print(f"  стохастичні:   T_warm = {STOCHASTIC_WARMUP_MS:.0f} мс, T_mod = {STOCHASTIC_SIMULATION_MS:.0f} мс "
          "(якщо не вказано інакше)")
    print()
    header = (f"{'Ключ':<36} {'T_warm':>7} {'T_mod':>8} {'P_зн':>7} {'підкл/с':>8} {'R=1':>6} "
              f"{'L1':>9} {'maxL2':>6} {'U1':>6} {'T, мс':>9}  Пройдено")
    print(header)
    for row in rows:
        scenario = row["scenario"]
        result = row["result"]
        print(f"{scenario.key:<36} {scenario.warmup_time:>7.0f} {scenario.simulation_time:>8.0f} "
              f"{result['destroyed_fraction']:>7.4f} {result['activations_per_second']:>8.3f} "
              f"{result['accelerated_time_fraction']:>6.3f} {result['mean_queue_length_1']:>9.2f} "
              f"{result['max_queue_length_2']:>6.1f} {result['utilization_1']:>6.3f} "
              f"{result['mean_sojourn_time']:>9.2f}  {'так' if row['passed'] else 'НІ'}")
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
            f"seed={PROTOCOL_SEED}, T_warm=0 мс, T_mod={PROTOCOL_SIMULATION_MS:.0f} мс\n"
            "Стан після події: t_next — момент наступної події елемента (∞ — подію не заплановано), "
            "У вікні — знищених / пакетів у ковзному вікні.\n"
            "Пріоритет одночасних подій: K2 → K1 → надходження.\n\n")


def save_event_protocols():
    first_events_model = TransmissionModel(protocol_config(PROTOCOL_EVENT_COUNT))
    first_events_model.simulate()
    write_text(
        PROTOCOL_FILE,
        protocol_title(f"Протокол перших {PROTOCOL_EVENT_COUNT} подій") + first_events_model.format_protocol() + "\n",
    )

    switch_model = TransmissionModel(protocol_config(SWITCH_SEARCH_EVENTS))
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
    rows = build_rows(run_all_scenarios())
    write_csv(TABLE_FILE, HEADER, [table_cells(row) for row in rows])
    print_table(rows)
    switch_step = save_event_protocols()
    print()
    print(f"Таблиця: {results_path(TABLE_FILE)}")
    print(f"Протокол перших {PROTOCOL_EVENT_COUNT} подій: {results_path(PROTOCOL_FILE)}")
    print(f"Фрагмент першого перемикання (подія №{switch_step}): {results_path(SWITCH_PROTOCOL_FILE)}")
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
