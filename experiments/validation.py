import time
from dataclasses import dataclass

import numpy as np
from scipy import stats

from experiments.common import CONFIDENCE_LEVEL, confidence_half_width, parallel_map, results_path, write_csv
from simulation.analytics import jackson_tandem_characteristics, tandem_destruction_probability
from simulation.config import ModelConfig
from simulation.model import Model

OUTPUT_FILE = "validation.csv"

REPLICATIONS = 20
WARMUP_TIME_MS = 10000.0
SIMULATION_TIME_MS = 1_000_000.0
DESTRUCTION_THRESHOLD_MS = 10.0
SEED_STRIDE = 1000
NORMAL_SERVICE_MS = 5.0
ACCELERATED_SERVICE_MS = 4.0


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
    return scenario_index, Model(build_config(SCENARIOS[scenario_index], seed)).simulate().to_dict()


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
    print("Аналітика існує лише без керування (режими never/always): контролер ресурсу перевірено "
          "верифікаційними сценаріями, а не формулами.")


def main():
    started = time.perf_counter()
    rows = []
    for scenario, replications in zip(SCENARIOS, run_all_replications()):
        rows.extend(build_rows(scenario, replications))
    write_csv(OUTPUT_FILE, HEADER, csv_rows(rows))
    print_table(rows)
    print()
    print(f"Збережено: {results_path(OUTPUT_FILE)}")
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
