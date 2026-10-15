import csv
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from simulation.analytics import jackson_tandem_characteristics, tandem_destruction_probability
from simulation.config import ModelConfig
from simulation.model import Model

RESULTS_DIRECTORY = Path(__file__).resolve().parent.parent / "results"
OUTPUT_FILE = RESULTS_DIRECTORY / "validation.csv"

REPLICATIONS = 20
WARMUP_TIME_MS = 10000.0
SIMULATION_TIME_MS = 1_000_000.0
DESTRUCTION_THRESHOLD_MS = 10.0
CONFIDENCE_LEVEL = 0.95
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


def run_single_replication(scenario, seed):
    return Model(build_config(scenario, seed)).simulate().to_dict()


def run_replications(scenario_index, scenario, executor):
    seeds = [SEED_STRIDE * (scenario_index + 1) + replication + 1 for replication in range(REPLICATIONS)]
    return list(executor.map(run_single_replication, [scenario] * REPLICATIONS, seeds))


def confidence_half_width(samples):
    standard_error = np.std(samples, ddof=1) / np.sqrt(len(samples))
    quantile = stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, df=len(samples) - 1)
    return float(quantile * standard_error)


def build_rows(scenario_index, scenario, executor):
    analytic = analytic_values(scenario)
    replications = run_replications(scenario_index, scenario, executor)
    rows = []
    for label, key in INDICATORS:
        samples = np.array([result[key] for result in replications])
        simulated_mean = float(samples.mean())
        half_width = confidence_half_width(samples)
        expected = analytic[key]
        relative_error = abs(simulated_mean - expected) / expected * 100.0
        inside = abs(simulated_mean - expected) <= half_width
        rows.append([
            scenario.name,
            label,
            expected,
            simulated_mean,
            half_width,
            relative_error,
            "так" if inside else "ні",
            REPLICATIONS,
            WARMUP_TIME_MS,
            SIMULATION_TIME_MS,
        ])
    return rows


def format_rows(rows):
    formatted = []
    for row in rows:
        formatted.append([
            row[0], row[1], f"{row[2]:.5f}", f"{row[3]:.5f}", f"{row[4]:.5f}", f"{row[5]:.3f}", row[6],
            row[7], f"{row[8]:.0f}", f"{row[9]:.0f}",
        ])
    return formatted


def save_csv(rows):
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        writer.writerows(format_rows(rows))


def print_table(rows):
    print(f"Валідація проти аналітики: T_warm = {WARMUP_TIME_MS:.0f} мс, "
          f"T_mod = {SIMULATION_TIME_MS:.0f} мс, прогонів = {REPLICATIONS}, довірча ймовірність 0.95")
    current_scenario = None
    for row in rows:
        if row[0] != current_scenario:
            current_scenario = row[0]
            print()
            print(current_scenario)
            print(f"  {'Показник':<30} {'Аналітика':>11} {'Імітація':>11} {'± ДІ':>10} {'Похибка, %':>11} {'В ДІ':>5}")
        print(f"  {row[1]:<30} {row[2]:>11.5f} {row[3]:>11.5f} {row[4]:>10.5f} {row[5]:>11.3f} {row[6]:>5}")


def main():
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor() as executor:
        for scenario_index, scenario in enumerate(SCENARIOS):
            rows.extend(build_rows(scenario_index, scenario, executor))
    save_csv(rows)
    print_table(rows)
    print()
    print(f"Збережено: {OUTPUT_FILE}")
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
