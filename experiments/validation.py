import math
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import stats

from experiments.common import CONFIDENCE_LEVEL, confidence_half_width, parallel_map, results_path, write_csv
from simulation.analytics import jackson_tandem_characteristics, tandem_destruction_probability
from simulation.config import ModelConfig
from simulation.fun_rand import FunRand
from simulation.transmission_model import TransmissionModel

OUTPUT_FILE = "validation.csv"
RECURSION_OUTPUT_FILE = "validation_recursion.csv"

REPLICATIONS = 20
WARMUP_TIME_MS = 10000.0
SIMULATION_TIME_MS = 1_000_000.0
DESTRUCTION_THRESHOLD_MS = 10.0
SEED_STRIDE = 1000
NORMAL_SERVICE_MS = 5.0
ACCELERATED_SERVICE_MS = 4.0

RECURSION_SEEDS = (1, 2, 3, 4, 5)
RECURSION_WARMUP_MS = 30_000.0
RECURSION_SIMULATION_MS = 600_000.0
RECURSION_RELATIVE_TOLERANCE = 1e-9
RECURSION_SCENARIOS = (
    ("Базовий B8, без керування (never)", dict(control_mode="never")),
    ("Базовий B8, ресурс завжди підключено (always)", dict(control_mode="always")),
    ("Базовий B8, керування θ = 30%, N = 100", dict(control_mode="adaptive")),
    ("Гістерезис 30/20, N = 300", dict(control_mode="adaptive", deactivation_threshold=0.20, window_size=300)),
    ("Інтервал unif(2; 8) мс, керування θ = 30%", dict(control_mode="adaptive", arrival_min=2.0, arrival_max=8.0,
                                                    arrival_mean=5.0)),
)
RECURSION_INDICATORS = (
    ("Вийшло пакетів, шт", "exited"),
    ("Знищено пакетів, шт", "destroyed"),
    ("Підключень ресурсу, шт", "activation_count"),
    ("Частка часу R=1", "accelerated_time_fraction"),
    ("Середній час перебування, мс", "mean_sojourn_time"),
    ("W1, мс", "mean_waiting_time_1"),
)
RECURSION_HEADER = [
    "Сценарій", "Показник", "Подійна модель (середнє)", "Рекурсія (середнє)", "Макс. відносна розбіжність",
    "Збіг", "Прогони (seed)", "T_warm, мс", "T_mod, мс",
]


@dataclass(frozen=True)
class ValidationScenario:
    name: str
    mean_interarrival_ms: float
    mean_service_ms: float

    @property
    def arrival_rate(self):
        return 1.0 / self.mean_interarrival_ms

    @property
    def service_rate(self):
        return 1.0 / self.mean_service_ms

    @property
    def control_mode(self):
        return "always" if self.mean_service_ms == ACCELERATED_SERVICE_MS else "never"


SCENARIOS = [
    ValidationScenario("lambda=1/6, mu=1/5 (інтервал 6 мс, обслуговування 5 мс)", 6.0, 5.0),
    ValidationScenario("lambda=1/6, mu=1/4 (інтервал 6 мс, обслуговування 4 мс)", 6.0, 4.0),
    ValidationScenario("lambda=1/10, mu=1/5 (інтервал 10 мс, обслуговування 5 мс)", 10.0, 5.0),
]

INDICATORS = [
    ("L1, пакетів", "mean_queue_length_1"),
    ("L2, пакетів", "mean_queue_length_2"),
    ("W1, мс", "mean_waiting_time_1"),
    ("W2, мс", "mean_waiting_time_2"),
    ("U1", "utilization_1"),
    ("U2", "utilization_2"),
    ("Середній час перебування, мс", "mean_sojourn_time"),
    ("P(T>10 мс)", "destroyed_fraction"),
]

HEADER = [
    "Сценарій",
    "Показник",
    "Аналітика",
    "Імітація (середнє)",
    "Напівширина 95% ДІ",
    "Відносна похибка, %",
    "Аналітика в ДІ",
    "Прогонів",
    "T_warm, мс",
    "T_mod, мс",
]


def build_config(scenario, seed):
    return ModelConfig(
        arrival_mean=scenario.mean_interarrival_ms,
        arrival_distribution="exponential",
        service_time_normal=NORMAL_SERVICE_MS,
        service_time_accelerated=ACCELERATED_SERVICE_MS,
        service_distribution="exponential",
        destruction_threshold=DESTRUCTION_THRESHOLD_MS,
        control_mode=scenario.control_mode,
        warmup_time=WARMUP_TIME_MS,
        simulation_time=SIMULATION_TIME_MS,
        seed=seed,
    )


def analytic_values(scenario):
    tandem = jackson_tandem_characteristics(scenario.arrival_rate, scenario.service_rate, scenario.service_rate)
    destruction = tandem_destruction_probability(
        scenario.arrival_rate, scenario.service_rate, scenario.service_rate, DESTRUCTION_THRESHOLD_MS
    )
    return {
        "mean_queue_length_1": tandem.node_1.mean_queue_length,
        "mean_queue_length_2": tandem.node_2.mean_queue_length,
        "mean_waiting_time_1": tandem.node_1.mean_waiting_time,
        "mean_waiting_time_2": tandem.node_2.mean_waiting_time,
        "utilization_1": tandem.node_1.utilization,
        "utilization_2": tandem.node_2.utilization,
        "mean_sojourn_time": tandem.mean_sojourn_time,
        "destroyed_fraction": destruction,
    }


def run_single_replication(task):
    scenario_index, seed = task
    return scenario_index, TransmissionModel(build_config(SCENARIOS[scenario_index], seed)).simulate().to_dict()


def run_all_replications():
    tasks = [
        (scenario_index, SEED_STRIDE * (scenario_index + 1) + replication + 1)
        for scenario_index in range(len(SCENARIOS))
        for replication in range(REPLICATIONS)
    ]
    replications_by_scenario = [[] for _ in SCENARIOS]
    for scenario_index, result in parallel_map(run_single_replication, tasks):
        replications_by_scenario[scenario_index].append(result)
    return replications_by_scenario


def build_rows(scenario, replications):
    analytic = analytic_values(scenario)
    rows = []
    for label, key in INDICATORS:
        samples = np.array([result[key] for result in replications])
        simulated_mean = float(samples.mean())
        half_width = confidence_half_width(samples)
        expected = analytic[key]
        rows.append({
            "scenario": scenario.name,
            "indicator": label,
            "analytic": expected,
            "simulated": simulated_mean,
            "half_width": half_width,
            "relative_error": abs(simulated_mean - expected) / expected * 100.0,
            "inside": abs(simulated_mean - expected) <= half_width,
        })
    return rows


class RecursiveController:
    def __init__(self, config):
        self.config = config
        self.window = deque()
        self.destroyed_in_window = 0
        self.switch_times = [0.0]
        self.switch_states = [config.control_mode == "always"]
        self.exit_times = []
        self.exit_destroyed = []
        self.applied_exits = 0

    def register_exit(self, exit_time, destroyed):
        self.exit_times.append(exit_time)
        self.exit_destroyed.append(destroyed)

    def apply_exits_until(self, time):
        while self.applied_exits < len(self.exit_times) and self.exit_times[self.applied_exits] <= time:
            if self.config.control_mode == "adaptive":
                self.update_window(self.exit_times[self.applied_exits], self.exit_destroyed[self.applied_exits])
            self.applied_exits += 1

    def update_window(self, exit_time, destroyed):
        window_size = self.config.window_size
        if len(self.window) == window_size:
            self.destroyed_in_window -= self.window.popleft()
        self.window.append(int(destroyed))
        self.destroyed_in_window += int(destroyed)
        if len(self.window) < window_size:
            return
        share = self.destroyed_in_window / window_size
        accelerated = self.switch_states[-1]
        if not accelerated and share >= self.config.activation_threshold:
            self.switch_times.append(exit_time)
            self.switch_states.append(True)
        elif accelerated and share < self.config.effective_deactivation_threshold:
            self.switch_times.append(exit_time)
            self.switch_states.append(False)

    def accelerated_at(self, time):
        self.apply_exits_until(time)
        return self.switch_states[bisect_right(self.switch_times, time) - 1]

    def accelerated_time(self, start, end):
        boundaries = self.switch_times[1:] + [math.inf]
        total = 0.0
        for begin, finish, accelerated in zip(self.switch_times, boundaries, self.switch_states):
            if accelerated:
                total += max(0.0, min(finish, end) - max(begin, start))
        return total

    def activations_between(self, start, end):
        return sum(
            1 for index in range(1, len(self.switch_times))
            if self.switch_states[index] and start < self.switch_times[index] <= end
        )


def recursive_tandem(config):
    rand = FunRand(config.seed)
    controller = RecursiveController(config)
    deadline = config.destruction_threshold + config.time_tolerance
    start, end = config.warmup_time, config.end_time

    def service_time(time):
        return config.service_time_accelerated if controller.accelerated_at(time) else config.service_time_normal

    exited = destroyed = started = 0
    sojourn_sum = waiting_sum = 0.0
    departure_1 = departure_2 = -math.inf
    arrival = rand.uniform(config.arrival_min, config.arrival_max)
    while arrival <= end:
        start_1 = max(arrival, departure_1)
        departure_1 = start_1 + service_time(start_1)
        start_2 = max(departure_1, departure_2)
        departure_2 = start_2 + service_time(start_2)
        sojourn = departure_2 - arrival
        controller.register_exit(departure_2, sojourn > deadline)
        if start < start_1 <= end:
            started += 1
            waiting_sum += start_1 - arrival
        if start < departure_2 <= end:
            exited += 1
            destroyed += sojourn > deadline
            sojourn_sum += sojourn
        arrival = arrival + rand.uniform(config.arrival_min, config.arrival_max)
    controller.apply_exits_until(end)
    return {
        "exited": exited,
        "destroyed": destroyed,
        "activation_count": controller.activations_between(start, end),
        "accelerated_time_fraction": controller.accelerated_time(start, end) / config.simulation_time,
        "mean_sojourn_time": sojourn_sum / exited if exited else 0.0,
        "mean_waiting_time_1": waiting_sum / started if started else 0.0,
    }


def recursion_config(overrides, seed):
    return ModelConfig(warmup_time=RECURSION_WARMUP_MS, simulation_time=RECURSION_SIMULATION_MS, seed=seed,
                       **overrides)


def compare_with_recursion(task):
    scenario_index, seed = task
    config = recursion_config(RECURSION_SCENARIOS[scenario_index][1], seed)
    return scenario_index, TransmissionModel(config).simulate().to_dict(), recursive_tandem(config)


def relative_difference(first, second):
    scale = max(abs(first), abs(second))
    return abs(first - second) / scale if scale > 0 else 0.0


def run_recursion_check():
    tasks = [(index, seed) for index in range(len(RECURSION_SCENARIOS)) for seed in RECURSION_SEEDS]
    pairs_by_scenario = [[] for _ in RECURSION_SCENARIOS]
    for scenario_index, model_result, recursion_result in parallel_map(compare_with_recursion, tasks):
        pairs_by_scenario[scenario_index].append((model_result, recursion_result))
    rows = []
    for (name, _), pairs in zip(RECURSION_SCENARIOS, pairs_by_scenario):
        for label, key in RECURSION_INDICATORS:
            model_values = [model_result[key] for model_result, _ in pairs]
            recursion_values = [recursion_result[key] for _, recursion_result in pairs]
            largest_difference = max(relative_difference(a, b) for a, b in zip(model_values, recursion_values))
            rows.append({
                "scenario": name,
                "indicator": label,
                "model": float(np.mean(model_values)),
                "recursion": float(np.mean(recursion_values)),
                "difference": largest_difference,
                "matched": largest_difference <= RECURSION_RELATIVE_TOLERANCE,
            })
    return rows


def recursion_csv_rows(rows):
    seeds = ", ".join(str(seed) for seed in RECURSION_SEEDS)
    return [
        [row["scenario"], row["indicator"], f"{row['model']:.6f}", f"{row['recursion']:.6f}",
         f"{row['difference']:.2e}", "так" if row["matched"] else "ні", seeds,
         f"{RECURSION_WARMUP_MS:.0f}", f"{RECURSION_SIMULATION_MS:.0f}"]
        for row in rows
    ]


def print_recursion_table(rows):
    print(f"Перевірка базової моделі B8 незалежним рекурсивним алгоритмом (рівняння тандему "
          f"D = max(A, D_попер) + S): T_warm = {RECURSION_WARMUP_MS:.0f} мс, "
          f"T_mod = {RECURSION_SIMULATION_MS:.0f} мс, seed = {RECURSION_SEEDS}")
    current_scenario = None
    for row in rows:
        if row["scenario"] != current_scenario:
            current_scenario = row["scenario"]
            print()
            print(current_scenario)
        print(f"  {row['indicator']:<30} {row['model']:>14.6f} {row['recursion']:>14.6f} "
              f"{row['difference']:>10.1e} {'так' if row['matched'] else 'НІ':>5}")
    matched = sum(1 for row in rows if row["matched"])
    print()
    print(f"Збіг подійної моделі з рекурсією: {matched} із {len(rows)} показників")


def coverage_summary(rows):
    inside_count = sum(1 for row in rows if row["inside"])
    total = len(rows)
    lowest_plausible = int(stats.binom.ppf(0.025, total, CONFIDENCE_LEVEL))
    return inside_count, total, lowest_plausible


def csv_rows(rows):
    return [
        [row["scenario"], row["indicator"], f"{row['analytic']:.5f}", f"{row['simulated']:.5f}",
         f"{row['half_width']:.5f}", f"{row['relative_error']:.3f}", "так" if row["inside"] else "ні",
         REPLICATIONS, f"{WARMUP_TIME_MS:.0f}", f"{SIMULATION_TIME_MS:.0f}"]
        for row in rows
    ]


def print_table(rows):
    print(f"Валідація проти аналітики: T_warm = {WARMUP_TIME_MS:.0f} мс, "
          f"T_mod = {SIMULATION_TIME_MS:.0f} мс, прогонів = {REPLICATIONS}, довірча ймовірність {CONFIDENCE_LEVEL}")
    current_scenario = None
    for row in rows:
        if row["scenario"] != current_scenario:
            current_scenario = row["scenario"]
            print()
            print(current_scenario)
            print(f"  {'Показник':<30} {'Аналітика':>11} {'Імітація':>11} {'± ДІ':>10} {'Похибка, %':>11} {'В ДІ':>5}")
        print(f"  {row['indicator']:<30} {row['analytic']:>11.5f} {row['simulated']:>11.5f} "
              f"{row['half_width']:>10.5f} {row['relative_error']:>11.3f} {'так' if row['inside'] else 'ні':>5}")
    inside_count, total, lowest_plausible = coverage_summary(rows)
    print()
    print(f"Аналітичне значення в 95% ДІ: {inside_count} із {total}. Для {total} незалежних інтервалів "
          f"очікується ≈{CONFIDENCE_LEVEL * total:.1f}; значення від {lowest_plausible} і вище узгоджуються "
          "з довірчою ймовірністю 0.95.")
    print("Аналітика існує лише для експоненційних законів без керування (режими never/always). Базову модель B8 "
          "з рівномірним надходженням, детермінованим обслуговуванням і контролером перевірено нижче "
          "незалежним рекурсивним алгоритмом.")


def main():
    started = time.perf_counter()
    rows = []
    for scenario, replications in zip(SCENARIOS, run_all_replications()):
        rows.extend(build_rows(scenario, replications))
    write_csv(OUTPUT_FILE, HEADER, csv_rows(rows))
    print_table(rows)
    print()
    recursion_rows = run_recursion_check()
    write_csv(RECURSION_OUTPUT_FILE, RECURSION_HEADER, recursion_csv_rows(recursion_rows))
    print_recursion_table(recursion_rows)
    print()
    print(f"Збережено: {results_path(OUTPUT_FILE)}, {results_path(RECURSION_OUTPUT_FILE)}")
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
