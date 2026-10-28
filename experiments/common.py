import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import stats

from simulation.config import ModelConfig
from simulation.transmission_model import TransmissionModel

RESULTS_DIRECTORY = Path(__file__).resolve().parent.parent / "results"
RUN_PARAMS_FILE = RESULTS_DIRECTORY / "run_params.json"
CONFIDENCE_LEVEL = 0.95
DESTRUCTION_REQUIREMENT = 0.30
SMALLEST_REPORTED_P_VALUE = 1e-300
MODE_LABELS = {
    "adaptive": "З керуванням",
    "never": "Без керування",
    "always": "Завжди прискорено",
}


def meets_destruction_requirement(destroyed_fraction_samples):
    upper_bound = destroyed_fraction_samples.mean() + confidence_half_width(destroyed_fraction_samples)
    return upper_bound <= DESTRUCTION_REQUIREMENT


def format_p_value(p_value):
    if p_value < SMALLEST_REPORTED_P_VALUE:
        return f"< {SMALLEST_REPORTED_P_VALUE:g}"
    return f"{p_value:.3g}"


def results_path(filename):
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIRECTORY / filename


def simulate_to_dict(config):
    return TransmissionModel(config).simulate().to_dict()


def simulate_with_trace(config):
    results = TransmissionModel(config).simulate()
    return results.to_dict(), [asdict(snapshot) for snapshot in results.trace]


def parallel_map(function, items):
    items = list(items)
    worker_count = min(len(items), os.cpu_count() or 1)
    if worker_count <= 1:
        return [function(item) for item in items]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(function, items))


def replication_seeds(replications, seed_offset=0):
    return [seed_offset + replication + 1 for replication in range(replications)]


def replicate_configs(base_config, replications, seed_offset=0):
    return [base_config.with_overrides(seed=seed) for seed in replication_seeds(replications, seed_offset)]


def run_replications(base_config, replications, seed_offset=0):
    return parallel_map(simulate_to_dict, replicate_configs(base_config, replications, seed_offset))


def confidence_half_width(samples, confidence_level=CONFIDENCE_LEVEL):
    samples = np.asarray(samples, dtype=float)
    if samples.size < 2:
        return 0.0
    standard_error = samples.std(ddof=1) / np.sqrt(samples.size)
    return float(stats.t.ppf(0.5 + confidence_level / 2.0, df=samples.size - 1) * standard_error)


def summarize_samples(samples):
    samples = np.asarray(samples, dtype=float)
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        "half_width": confidence_half_width(samples),
    }


def summarize_replications(replication_results, keys):
    return {key: summarize_samples([result[key] for result in replication_results]) for key in keys}


def save_run_params(params):
    with open(results_path(RUN_PARAMS_FILE.name), "w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=False, indent=2)


def load_run_params():
    if not RUN_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"{RUN_PARAMS_FILE} не знайдено: спершу виконайте python -m experiments.transient "
            f"і python -m experiments.replications"
        )
    with open(RUN_PARAMS_FILE, encoding="utf-8") as file:
        return json.load(file)


def base_config_from_run_params(params=None, **overrides):
    params = params or load_run_params()
    config = ModelConfig(
        warmup_time=params["warmup_time_ms"],
        simulation_time=params["simulation_time_ms"],
    )
    return config.with_overrides(**overrides)


def write_csv(filename, header, rows):
    path = results_path(filename)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_text(filename, text):
    path = results_path(filename)
    with open(path, "w", encoding="utf-8-sig") as file:
        file.write(text)
    return path
