import math
import time

import numpy as np
from scipy import stats

from experiments.common import (
    base_config_from_run_params,
    load_run_params,
    parallel_map,
    replicate_configs,
    run_replications,
    write_csv,
)
from experiments.plotting import (
    TEXT_SECONDARY,
    add_horizontal_reference,
    add_legend,
    create_figure,
    label_axes,
    plot_line,
    save_figure,
    series_color,
    simulation_caption,
)
from simulation.transmission_model import TransmissionModel

SIGNIFICANCE_LEVEL = 0.05
GAP_RUNS = 20
EXAMPLE_RUN_INDEX = 0
MINIMUM_EXPECTED_PER_BIN = 5
HISTOGRAM_UPPER_QUANTILE = 0.995
HISTOGRAM_BINS = 60
DENSITY_GRID_POINTS = 600
PERCENT_SCALE = 100.0
LEGEND_HEADROOM = 1.3

GAP_CANDIDATES = (
    ("shifted_exponential", "Експоненційний зі зсувом", stats.expon, {}, 2),
    ("gamma", "Гамма", stats.gamma, {"floc": 0.0}, 2),
    ("weibull", "Вейбулла", stats.weibull_min, {"floc": 0.0}, 2),
    ("shifted_lognormal", "Логнормальний зі зсувом", stats.lognorm, {}, 3),
)

REPLICATION_RESPONSES = (
    ("destroyed_fraction", "Частка знищених пакетів P_зн", True),
    ("activations_per_second", "Частота підключення ресурсу, 1/с", False),
    ("accelerated_time_fraction", "Частка часу з R=1", True),
)
NORMAL_ESTIMATED_PARAMETERS = 2


class ActivationRecorder:
    def __init__(self, resource_state, warmup_time):
        self.resource_state = resource_state
        self.warmup_time = warmup_time
        self.was_accelerated = resource_state.accelerated
        self.activation_times = []

    def on_packet_exit(self, packet):
        accelerated = self.resource_state.accelerated
        if accelerated and not self.was_accelerated and packet.exit_time >= self.warmup_time:
            self.activation_times.append(packet.exit_time)
        self.was_accelerated = accelerated


def collect_activation_gaps(config):
    model = TransmissionModel(config)
    recorder = ActivationRecorder(model.resource_state, config.warmup_time)
    model.sink.add_exit_listener(recorder.on_packet_exit)
    model.simulate()
    return np.diff(recorder.activation_times)


def sturges_bin_count(sample_size):
    by_sturges = round(1 + 3.322 * math.log10(sample_size))
    by_expected_frequency = sample_size // MINIMUM_EXPECTED_PER_BIN
    return int(max(2, min(by_sturges, by_expected_frequency)))


def chi_square_test(samples, distribution, parameters, estimated_parameter_count):
    bin_count = sturges_bin_count(samples.size)
    inner_edges = distribution.ppf(np.linspace(0.0, 1.0, bin_count + 1)[1:-1], *parameters)
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    observed = np.histogram(samples, bins=edges)[0]
    expected = np.full(bin_count, samples.size / bin_count)
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    degrees = bin_count - 1 - estimated_parameter_count
    critical = float(stats.chi2.ppf(1 - SIGNIFICANCE_LEVEL, degrees))
    p_value = float(stats.chi2.sf(statistic, degrees))
    return {
        "bins": bin_count,
        "statistic": statistic,
        "degrees": degrees,
        "critical": critical,
        "p_value": p_value,
        "accepted": statistic <= critical,
    }


def describe_parameters(candidate_key, parameters):
    if candidate_key == "shifted_exponential":
        shift, scale = parameters
        return f"зсув a = {shift:.3f} мс; середнє над зсувом 1/λ = {scale:.3f} мс"
    if candidate_key == "gamma":
        shape, _, scale = parameters
        return f"форма k = {shape:.4f}; масштаб θ = {scale:.3f} мс"
    if candidate_key == "weibull":
        shape, _, scale = parameters
        return f"форма c = {shape:.4f}; масштаб λ = {scale:.3f} мс"
    shape, shift, scale = parameters
    return f"σ = {shape:.4f}; μ = ln(масштабу) = {math.log(scale):.4f}; зсув a = {shift:.3f} мс"


def fit_candidate(samples, distribution, fit_options):
    return distribution.fit(samples, **fit_options)


def test_candidates(samples):
    outcomes = {}
    for key, _, distribution, fit_options, estimated in GAP_CANDIDATES:
        parameters = fit_candidate(samples, distribution, fit_options)
        outcomes[key] = {"parameters": parameters, **chi_square_test(samples, distribution, parameters, estimated)}
    return outcomes


def descriptive_row(label, samples, unit_scale=1.0):
    values = np.asarray(samples, dtype=float) * unit_scale
    mean = float(values.mean())
    deviation = float(values.std(ddof=1))
    return [label, values.size, mean, deviation, deviation / mean if mean else float("nan"),
            float(values.min()), float(values.max())]


def gap_rows(outcomes_by_run):
    example = outcomes_by_run[EXAMPLE_RUN_INDEX]
    rows = []
    for key, title, _, _, estimated in GAP_CANDIDATES:
        outcome = example[key]
        p_values = [outcomes[key]["p_value"] for outcomes in outcomes_by_run]
        accepted_runs = sum(1 for outcomes in outcomes_by_run if outcomes[key]["accepted"])
        rows.append([
            title, describe_parameters(key, outcome["parameters"]), estimated, outcome["bins"],
            outcome["statistic"], outcome["degrees"], outcome["critical"], outcome["p_value"],
            "приймається" if outcome["accepted"] else "відхиляється",
            f"{accepted_runs} із {len(outcomes_by_run)}", float(np.median(p_values)),
        ])
    return rows


def replication_rows(results):
    rows = []
    for key, title, _ in REPLICATION_RESPONSES:
        samples = np.array([result[key] for result in results])
        mean, deviation = stats.norm.fit(samples)
        outcome = chi_square_test(samples, stats.norm, (mean, deviation), NORMAL_ESTIMATED_PARAMETERS)
        rows.append([
            title, samples.size, mean, deviation, outcome["bins"], outcome["statistic"], outcome["degrees"],
            outcome["critical"], outcome["p_value"],
            "приймається" if outcome["accepted"] else "відхиляється",
        ])
    return rows


def plot_activation_gaps(gaps_by_run, outcomes_by_run, caption):
    figure, axes = create_figure(caption, rows=1, columns=2, width=14.0, height=5.2)
    histogram_axis, p_value_axis = axes
    samples = gaps_by_run[EXAMPLE_RUN_INDEX]
    upper = float(np.quantile(samples, HISTOGRAM_UPPER_QUANTILE))
    histogram_axis.hist(samples[samples <= upper], bins=HISTOGRAM_BINS, density=True,
                        color="#d9d8d3", edgecolor="#ffffff", linewidth=0.5,
                        label=f"Імітація, прогін {EXAMPLE_RUN_INDEX + 1} (n = {samples.size})")
    grid = np.linspace(0.0, upper, DENSITY_GRID_POINTS)
    example = outcomes_by_run[EXAMPLE_RUN_INDEX]
    for slot, (key, title, distribution, _, _) in enumerate(GAP_CANDIDATES):
        outcome = example[key]
        plot_line(histogram_axis, grid, distribution.pdf(grid, *outcome["parameters"]),
                  f"{title}: p = {outcome['p_value']:.3g}", slot, markers=False, linewidth=1.8)
    label_axes(histogram_axis, "Інтервал між підключеннями ресурсу",
               "Інтервал між послідовними підключеннями, мс", "Щільність ймовірності, 1/мс")
    add_legend(histogram_axis, "upper right")

    rng = np.random.default_rng(0)
    for slot, (key, _, _, _, _) in enumerate(GAP_CANDIDATES):
        p_values = np.array([outcomes[key]["p_value"] for outcomes in outcomes_by_run])
        jitter = rng.uniform(-0.15, 0.15, p_values.size)
        p_value_axis.scatter(slot + jitter, np.maximum(p_values, 1e-300), color=series_color(slot), s=22, zorder=3)
    add_horizontal_reference(p_value_axis, SIGNIFICANCE_LEVEL, f"Рівень значущості α = {SIGNIFICANCE_LEVEL}")
    p_value_axis.set_yscale("log")
    p_value_axis.set_xticks(range(len(GAP_CANDIDATES)), [title for _, title, _, _, _ in GAP_CANDIDATES])
    p_value_axis.tick_params(axis="x", labelsize=8)
    p_value_axis.grid(axis="x", visible=False)
    label_axes(p_value_axis, f"Критерій χ² у кожному з {len(outcomes_by_run)} прогонів",
               "Гіпотетичний закон розподілу", "p-значення (логарифмічна шкала)")
    add_legend(p_value_axis, "lower right")
    figure.tight_layout()
    return save_figure(figure, "distribution_activation_gaps.png")


def plot_replication_estimates(results, caption):
    figure, axes = create_figure(caption, rows=1, columns=len(REPLICATION_RESPONSES), width=15.0, height=4.6)
    for slot, (axis, (key, title, is_fraction)) in enumerate(zip(axes, REPLICATION_RESPONSES)):
        scale = PERCENT_SCALE if is_fraction else 1.0
        samples = np.array([result[key] for result in results]) * scale
        bin_count = sturges_bin_count(samples.size)
        axis.hist(samples, bins=bin_count, density=True, color="#d9d8d3", edgecolor="#ffffff",
                  label=f"Оцінки за прогонами (n = {samples.size})")
        mean, deviation = stats.norm.fit(samples)
        grid = np.linspace(samples.min() - deviation, samples.max() + deviation, DENSITY_GRID_POINTS)
        density = stats.norm.pdf(grid, mean, deviation)
        plot_line(axis, grid, density, "Нормальний закон", slot, markers=False)
        unit = "%" if is_fraction else "1/с"
        density_unit = "1/%" if is_fraction else "с"
        label_axes(axis, title, f"Оцінка за один прогін, {unit}", f"Щільність ймовірності, {density_unit}")
        axis.set_ylim(0, max(axis.get_ylim()[1], density.max()) * LEGEND_HEADROOM)
        axis.tick_params(axis="x", labelsize=8, colors=TEXT_SECONDARY)
        add_legend(axis, "upper left")
    figure.tight_layout()
    return save_figure(figure, "distribution_replications.png")


def main():
    started = time.perf_counter()
    params = load_run_params()
    base_config = base_config_from_run_params(params)

    gaps_by_run = parallel_map(collect_activation_gaps, replicate_configs(base_config, GAP_RUNS))
    outcomes_by_run = [test_candidates(gaps) for gaps in gaps_by_run]
    replication_results = run_replications(base_config, params["replications"])

    descriptive = [descriptive_row(f"Інтервал між підключеннями ресурсу, мс (прогін {EXAMPLE_RUN_INDEX + 1})",
                                   gaps_by_run[EXAMPLE_RUN_INDEX])]
    descriptive.append(descriptive_row(f"Інтервал між підключеннями ресурсу, мс (усі {GAP_RUNS} прогонів разом)",
                                       np.concatenate(gaps_by_run)))
    for key, title, _ in REPLICATION_RESPONSES:
        descriptive.append(descriptive_row(f"{title}: оцінка за один прогін",
                                           [result[key] for result in replication_results]))
    write_csv("distribution_descriptive.csv",
              ["Величина", "Обсяг вибірки n", "Середнє", "Середньоквадратичне відхилення", "Коефіцієнт варіації",
               "Мінімум", "Максимум"], descriptive)
    write_csv("distribution_activation_gaps.csv",
              ["Гіпотетичний закон", f"Параметри (прогін {EXAMPLE_RUN_INDEX + 1})", "Оцінених параметрів r",
               "Інтервалів k", "χ²", "df = k − 1 − r", f"χ²_крит (α = {SIGNIFICANCE_LEVEL})", "p-value",
               f"Гіпотеза (прогін {EXAMPLE_RUN_INDEX + 1})", "Прийнято в прогонах", "Медіана p за прогонами"],
              gap_rows(outcomes_by_run))
    write_csv("distribution_replications.csv",
              ["Відгук", "Прогонів n", "Середнє", "Середньоквадратичне відхилення", "Інтервалів k", "χ²",
               "df = k − 1 − 2", f"χ²_крит (α = {SIGNIFICANCE_LEVEL})", "p-value", "Гіпотеза про нормальний закон"],
              replication_rows(replication_results))

    plot_activation_gaps(gaps_by_run, outcomes_by_run, simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], GAP_RUNS,
        f"базовий сценарій B8; інтервали рівноймовірні, k за формулою Стерджеса; α = {SIGNIFICANCE_LEVEL}",
    ))
    plot_replication_estimates(replication_results, simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], params["replications"],
        "базовий сценарій B8; криві — нормальний закон з оціненими параметрами",
    ))

    for row in descriptive:
        print(" | ".join(f"{cell:.5g}" if isinstance(cell, float) else str(cell) for cell in row))
    print("\nІнтервал між підключеннями ресурсу, критерій χ²")
    for row in gap_rows(outcomes_by_run):
        print(" | ".join(f"{cell:.4g}" if isinstance(cell, float) else str(cell) for cell in row))
    print("\nНормальність оцінок за прогонами, критерій χ²")
    for row in replication_rows(replication_results):
        print(" | ".join(f"{cell:.4g}" if isinstance(cell, float) else str(cell) for cell in row))
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
