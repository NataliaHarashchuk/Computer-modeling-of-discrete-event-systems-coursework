import time

import numpy as np
from scipy import stats

from experiments.common import (
    base_config_from_run_params,
    confidence_half_width,
    load_run_params,
    run_replications,
    write_csv,
)
from experiments.plotting import (
    TEXT_SECONDARY,
    add_horizontal_reference,
    add_legend,
    create_figure,
    label_axes,
    plot_line_with_interval,
    save_figure,
    series_color,
    simulation_caption,
)

DESTRUCTION_LIMIT = 0.30
SIGNIFICANCE_LEVEL = 0.05
TWO_FACTOR_OFF_THRESHOLDS = (0.10, 0.20, 0.30)
TWO_FACTOR_WINDOW_SIZES = (50, 100, 300)
TWO_FACTOR_SEED_STRIDE = 10_000
TWO_FACTOR_RESPONSES = (
    ("destroyed_fraction", "P_зн", 100.0, "Частка знищених пакетів P_зн, %"),
    ("activations_per_second", "Підключень ресурсу, 1/с", 1.0, "Частота підключень ресурсу, 1/с"),
    ("accelerated_time_fraction", "Частка часу з R=1", 100.0, "Частка часу з R=1, %"),
)
RESPONSE_KEYS = (
    "destroyed_fraction",
    "destroyed_per_second",
    "activations_per_second",
    "accelerated_time_fraction",
    "accelerated_packets_fraction",
    "limit_violation_time_fraction",
    "max_window_destroyed_fraction",
    "mean_sojourn_time",
)
HYSTERESIS_OFF_THRESHOLDS = [round(0.30 - 0.025 * step, 3) for step in range(9)]

PROPOSAL_VARIANTS = (
    ("Базовий (30/30, N=100)", "Базовий 30/30",
     dict(activation_threshold=0.30, deactivation_threshold=0.30, window_size=100)),
    ("Гістерезис 30/20", "Гіст. 30/20",
     dict(activation_threshold=0.30, deactivation_threshold=0.20, window_size=100)),
    ("Гістерезис 30/25", "Гіст. 30/25",
     dict(activation_threshold=0.30, deactivation_threshold=0.25, window_size=100)),
    ("Гістерезис 25/15", "Гіст. 25/15",
     dict(activation_threshold=0.25, deactivation_threshold=0.15, window_size=100)),
    ("Поріг 25 без гістерезису", "Поріг 25/25",
     dict(activation_threshold=0.25, deactivation_threshold=0.25, window_size=100)),
    ("Вікно N=300", "N=300",
     dict(activation_threshold=0.30, deactivation_threshold=0.30, window_size=300)),
    ("Гістерезис 30/20 + N=300", "Гіст. 30/20 + N=300",
     dict(activation_threshold=0.30, deactivation_threshold=0.20, window_size=300)),
    ("Раннє відкидання без ресурсу", "Відкидання, R=0",
     dict(control_mode="never", early_discard=True)),
    ("Раннє відкидання + керування 30/30", "Відкидання + 30/30",
     dict(activation_threshold=0.30, deactivation_threshold=0.30, window_size=100, early_discard=True)),
)
REFERENCE_VARIANTS = (
    ("Без керування (never)", dict(control_mode="never")),
    ("Завжди прискорено (always)", dict(control_mode="always")),
)


def collect_samples(results):
    return {key: np.array([result[key] for result in results]) for key in RESPONSE_KEYS}


def run_variant(params, overrides):
    config = base_config_from_run_params(params, **overrides)
    return collect_samples(run_replications(config, params["replications"]))


def relative_change_percent(samples, baseline_samples):
    baseline_mean = baseline_samples.mean()
    if baseline_mean == 0:
        return float("nan")
    return (samples.mean() - baseline_mean) / baseline_mean * 100.0


def paired_p_value(samples, baseline_samples):
    if np.allclose(samples, baseline_samples):
        return 1.0
    return float(stats.ttest_rel(samples, baseline_samples).pvalue)


def meets_requirement(samples):
    return samples.mean() + confidence_half_width(samples) <= DESTRUCTION_LIMIT


def comparison_row(label, samples, baseline):
    destroyed = samples["destroyed_fraction"]
    activations = samples["activations_per_second"]
    accelerated = samples["accelerated_time_fraction"]
    violation = samples["limit_violation_time_fraction"]
    return [
        label,
        round(destroyed.mean() * 100, 3),
        round(confidence_half_width(destroyed) * 100, 3),
        round(samples["destroyed_per_second"].mean(), 3),
        round(activations.mean(), 3),
        round(confidence_half_width(activations), 3),
        round(accelerated.mean() * 100, 3),
        round(confidence_half_width(accelerated) * 100, 3),
        round(samples["accelerated_packets_fraction"].mean() * 100, 3),
        round(violation.mean() * 100, 3),
        round(confidence_half_width(violation) * 100, 3),
        round(samples["max_window_destroyed_fraction"].mean() * 100, 2),
        round(samples["mean_sojourn_time"].mean(), 3),
        round(relative_change_percent(destroyed, baseline["destroyed_fraction"]), 2),
        round(relative_change_percent(activations, baseline["activations_per_second"]), 2),
        round(relative_change_percent(accelerated, baseline["accelerated_time_fraction"]), 2),
        f"{paired_p_value(destroyed, baseline['destroyed_fraction']):.3g}",
        f"{paired_p_value(activations, baseline['activations_per_second']):.3g}",
        f"{paired_p_value(accelerated, baseline['accelerated_time_fraction']):.3g}",
        "так" if meets_requirement(destroyed) else "ні",
    ]


COMPARISON_HEADER = [
    "Варіант",
    "P_зн, %",
    "P_зн ДІ ±, %",
    "Знищень/с",
    "Підключень/с",
    "Підключень/с ДІ ±",
    "Частка часу R=1, %",
    "Частка часу R=1 ДІ ±, %",
    "Частка пакетів з прискоренням, %",
    "Частка часу з рівнем знищення >30% у вікні 100 пакетів, %",
    "Частка часу з рівнем >30% ДІ ±, %",
    "Найбільша частка знищених у вікні (середнє за прогонами), %",
    "Середній час перебування, мс",
    "Зміна P_зн відносно базового, %",
    "Зміна підключень/с відносно базового, %",
    "Зміна частки часу R=1 відносно базового, %",
    "p-value P_зн (парний t)",
    "p-value підключень/с (парний t)",
    "p-value частки часу R=1 (парний t)",
    "Виконується P_зн ≤ 30%",
    "Парето-оптимальний (підключення/с і частка часу R=1)",
]


def plot_horizontal_bars(axis, labels, samples_list, key, scale, x_label, title, reference=None, decimals=1):
    headroom = 1.22 if reference is None else 1.6
    means = [samples[key].mean() * scale for samples in samples_list]
    halves = [confidence_half_width(samples[key]) * scale for samples in samples_list]
    positions = np.arange(len(labels))
    axis.barh(
        positions, means, xerr=halves, height=0.62, color=series_color(0),
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.2, "capsize": 3},
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    label_axes(axis, title, x_label, "")
    upper = max(mean + half for mean, half in zip(means, halves))
    axis.set_xlim(0, upper * headroom)
    for position, mean, half in zip(positions, means, halves):
        axis.annotate(
            f"{mean:.{decimals}f}", (mean + half, position), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center", fontsize=8.5, zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"),
        )
    if reference is not None:
        axis.axvline(reference[0], color="#6f6e69", linewidth=1.2, linestyle="--", label=reference[1])
        add_legend(axis, "lower right")


def plot_comparison(labels, samples_list, params):
    caption = simulation_caption_for(params, "спільні випадкові числа для всіх варіантів")
    figure, axes = create_figure(caption, rows=1, columns=4, width=20.0, height=6.0)
    plot_horizontal_bars(
        axes[0], labels, samples_list, "destroyed_fraction", 100.0,
        "Частка знищених пакетів P_зн, %", "Частка знищених пакетів",
        reference=(DESTRUCTION_LIMIT * 100, "допустимо 30%"), decimals=2,
    )
    plot_horizontal_bars(
        axes[1], labels, samples_list, "activations_per_second", 1.0,
        "Частота підключень ресурсу, 1/с", "Частота підключень ресурсу", decimals=2,
    )
    plot_horizontal_bars(
        axes[2], labels, samples_list, "accelerated_time_fraction", 100.0,
        "Частка часу з R=1, %", "Частка часу з підключеним ресурсом", decimals=1,
    )
    plot_horizontal_bars(
        axes[3], labels, samples_list, "limit_violation_time_fraction", 100.0,
        "Частка часу, %", "Рівень знищення у вікні 100 пакетів > 30%", decimals=1,
    )
    for axis in axes[1:]:
        axis.set_yticklabels([])
    figure.tight_layout()
    return save_figure(figure, "proposal_comparison.png")


def simulation_caption_for(params, extra):
    return simulation_caption(params["simulation_time_ms"], params["warmup_time_ms"], params["replications"], extra)


def run_hysteresis_sweep(params):
    sweep_samples = []
    for off_threshold in HYSTERESIS_OFF_THRESHOLDS:
        sweep_samples.append(run_variant(params, dict(
            activation_threshold=0.30, deactivation_threshold=off_threshold, window_size=100,
        )))
    return sweep_samples


def hysteresis_rows(sweep_samples):
    rows = []
    for off_threshold, samples in zip(HYSTERESIS_OFF_THRESHOLDS, sweep_samples):
        destroyed = samples["destroyed_fraction"]
        activations = samples["activations_per_second"]
        accelerated = samples["accelerated_time_fraction"]
        violation = samples["limit_violation_time_fraction"]
        rows.append([
            round(off_threshold * 100, 1),
            round(destroyed.mean() * 100, 3), round(confidence_half_width(destroyed) * 100, 3),
            round(activations.mean(), 3), round(confidence_half_width(activations), 3),
            round(accelerated.mean() * 100, 3), round(confidence_half_width(accelerated) * 100, 3),
            round(violation.mean() * 100, 3), round(confidence_half_width(violation) * 100, 3),
            "так" if meets_requirement(destroyed) else "ні",
        ])
    return rows


def plot_hysteresis(sweep_samples, params):
    caption = simulation_caption_for(params, "θ_on = 30%, N = 100")
    figure, axes = create_figure(caption, rows=1, columns=3, width=15.5, height=4.6)
    x_values = [threshold * 100 for threshold in HYSTERESIS_OFF_THRESHOLDS]
    x_label = "Поріг вимкнення ресурсу θ_off, %"
    panels = (
        ("destroyed_fraction", 100.0, "Частка знищених пакетів P_зн, %", "Частка знищених пакетів"),
        ("activations_per_second", 1.0, "Частота підключень ресурсу, 1/с", "Частота підключень ресурсу"),
        ("accelerated_time_fraction", 100.0, "Частка часу з R=1, %", "Частка часу з підключеним ресурсом"),
    )
    for axis, (key, scale, y_label, title) in zip(axes, panels):
        means = [samples[key].mean() * scale for samples in sweep_samples]
        halves = [confidence_half_width(samples[key]) * scale for samples in sweep_samples]
        plot_line_with_interval(axis, x_values, means, halves, "середнє ± 95% ДІ", 0)
        label_axes(axis, title, x_label, y_label)
        axis.invert_xaxis()
        if key == "destroyed_fraction":
            add_horizontal_reference(axis, DESTRUCTION_LIMIT * 100, "допустимо 30%")
        add_legend(axis, "best" if key != "destroyed_fraction" else "lower left")
    figure.tight_layout()
    return save_figure(figure, "proposal_hysteresis.png")


def resource_costs(samples):
    return samples["activations_per_second"].mean(), samples["accelerated_time_fraction"].mean()


def dominates(first, second):
    return all(a <= b for a, b in zip(first, second)) and any(a < b for a, b in zip(first, second))


def pareto_indices(samples_list):
    feasible = [index for index, samples in enumerate(samples_list) if meets_requirement(samples["destroyed_fraction"])]
    costs = {index: resource_costs(samples_list[index]) for index in feasible}
    return [
        index for index in feasible
        if not any(dominates(costs[other], costs[index]) for other in feasible if other != index)
    ]


def two_factor_cells():
    return [
        (off_threshold, window_size)
        for off_threshold in TWO_FACTOR_OFF_THRESHOLDS
        for window_size in TWO_FACTOR_WINDOW_SIZES
    ]


def run_two_factor_experiment(params):
    samples_by_cell = {}
    for cell_index, (off_threshold, window_size) in enumerate(two_factor_cells()):
        config = base_config_from_run_params(
            params, activation_threshold=0.30, deactivation_threshold=off_threshold, window_size=window_size,
        )
        results = run_replications(config, params["replications"], TWO_FACTOR_SEED_STRIDE * (cell_index + 1))
        samples_by_cell[(off_threshold, window_size)] = collect_samples(results)
    return samples_by_cell


def two_way_anova(samples_by_cell, key):
    levels_a = TWO_FACTOR_OFF_THRESHOLDS
    levels_b = TWO_FACTOR_WINDOW_SIZES
    data = np.array([[samples_by_cell[(a, b)][key] for b in levels_b] for a in levels_a])
    count_a, count_b, replications = data.shape
    grand_mean = data.mean()
    cell_means = data.mean(axis=2)
    means_a = data.mean(axis=(1, 2))
    means_b = data.mean(axis=(0, 2))
    sum_squares = {
        "A": count_b * replications * float(np.sum((means_a - grand_mean) ** 2)),
        "B": count_a * replications * float(np.sum((means_b - grand_mean) ** 2)),
        "AB": replications * float(np.sum((cell_means - means_a[:, None] - means_b[None, :] + grand_mean) ** 2)),
    }
    sum_squares_error = float(np.sum((data - cell_means[:, :, None]) ** 2))
    degrees = {"A": count_a - 1, "B": count_b - 1, "AB": (count_a - 1) * (count_b - 1)}
    degrees_error = count_a * count_b * (replications - 1)
    mean_square_error = sum_squares_error / degrees_error
    sum_squares_total = float(np.sum((data - grand_mean) ** 2))
    rows = []
    for source_key, source_name in (("A", "θ_off"), ("B", "N"), ("AB", "θ_off × N")):
        mean_square = sum_squares[source_key] / degrees[source_key]
        f_value = mean_square / mean_square_error
        f_critical = float(stats.f.ppf(1 - SIGNIFICANCE_LEVEL, degrees[source_key], degrees_error))
        p_value = float(stats.f.sf(f_value, degrees[source_key], degrees_error))
        effect_share = sum_squares[source_key] / sum_squares_total
        partial_effect_share = sum_squares[source_key] / (sum_squares[source_key] + sum_squares_error)
        rows.append([source_name, sum_squares[source_key], degrees[source_key], mean_square, f_value, f_critical,
                     p_value, effect_share, partial_effect_share,
                     "істотний" if f_value > f_critical else "неістотний"])
    rows.append(["Залишок", sum_squares_error, degrees_error, mean_square_error, "", "", "", "", "", ""])
    return rows


def two_factor_means_rows(samples_by_cell):
    rows = []
    for (off_threshold, window_size), samples in samples_by_cell.items():
        row = [round(off_threshold * 100, 1), window_size]
        for key, _, scale, _ in TWO_FACTOR_RESPONSES:
            row.extend([round(samples[key].mean() * scale, 4), round(confidence_half_width(samples[key]) * scale, 4)])
        row.append("так" if meets_requirement(samples["destroyed_fraction"]) else "ні")
        rows.append(row)
    return rows


def plot_two_factor(samples_by_cell, params):
    caption = simulation_caption_for(
        params, "θ_on = 30%; незалежні потоки випадкових чисел для кожної комбінації; смуга — 95% ДІ",
    )
    figure, axes = create_figure(caption, rows=1, columns=3, width=15.5, height=4.6)
    x_values = [threshold * 100 for threshold in TWO_FACTOR_OFF_THRESHOLDS]
    for axis, (key, _, scale, y_label) in zip(axes, TWO_FACTOR_RESPONSES):
        for slot, window_size in enumerate(TWO_FACTOR_WINDOW_SIZES):
            cells = [samples_by_cell[(threshold, window_size)][key] for threshold in TWO_FACTOR_OFF_THRESHOLDS]
            plot_line_with_interval(axis, x_values, [cell.mean() * scale for cell in cells],
                                    [confidence_half_width(cell) * scale for cell in cells], f"N = {window_size}", slot)
        label_axes(axis, y_label.split(",")[0], "Поріг вимкнення ресурсу θ_off, %", y_label)
        axis.set_xticks(x_values)
        axis.invert_xaxis()
        if key == "destroyed_fraction":
            add_horizontal_reference(axis, DESTRUCTION_LIMIT * 100, "допустимо 30%")
        add_legend(axis, "best")
    figure.tight_layout()
    return save_figure(figure, "proposal_two_factor.png")


def main():
    start = time.perf_counter()
    params = load_run_params()

    labels = [label for label, _, _ in PROPOSAL_VARIANTS]
    samples_list = [run_variant(params, overrides) for _, _, overrides in PROPOSAL_VARIANTS]
    baseline = samples_list[0]
    reference_samples = [(label, run_variant(params, overrides)) for label, overrides in REFERENCE_VARIANTS]

    pareto = set(pareto_indices(samples_list))
    rows = [
        comparison_row(label, samples, baseline) + ["так" if index in pareto else "ні"]
        for index, (label, samples) in enumerate(zip(labels, samples_list))
    ]
    rows += [comparison_row(label, samples, baseline) + ["—"] for label, samples in reference_samples]
    write_csv("proposal_comparison.csv", COMPARISON_HEADER, rows)
    plot_comparison([short_label for _, short_label, _ in PROPOSAL_VARIANTS], samples_list, params)

    sweep_samples = run_hysteresis_sweep(params)
    write_csv(
        "proposal_hysteresis.csv",
        ["θ_off, %", "P_зн, %", "P_зн ДІ ±, %", "Підключень/с", "Підключень/с ДІ ±",
         "Частка часу R=1, %", "Частка часу R=1 ДІ ±, %",
         "Частка часу з рівнем >30% у вікні, %", "Частка часу з рівнем >30% ДІ ±, %", "Виконується P_зн ≤ 30%"],
        hysteresis_rows(sweep_samples),
    )
    plot_hysteresis(sweep_samples, params)

    samples_by_cell = run_two_factor_experiment(params)
    write_csv(
        "proposal_two_factor_means.csv",
        ["θ_off, %", "N, пакетів"]
        + [column for _, label, _, _ in TWO_FACTOR_RESPONSES
           for column in (f"{label}{', %' if '1/с' not in label else ''}: середнє",
                          f"{label}{', %' if '1/с' not in label else ''}: напівширина 95% ДІ")]
        + ["Виконується P_зн ≤ 30%"],
        two_factor_means_rows(samples_by_cell),
    )
    anova_rows = []
    for key, label, _, _ in TWO_FACTOR_RESPONSES:
        anova_rows += [[label] + row for row in two_way_anova(samples_by_cell, key)]
    write_csv(
        "proposal_two_factor_anova.csv",
        ["Відгук", "Джерело мінливості", "SS", "df", "MS", "F", "F_крит (alpha=0.05)", "p-value",
         "η² (частка загальної SS)", "Частковий η²", "Вплив"],
        anova_rows,
    )
    plot_two_factor(samples_by_cell, params)

    for row in rows:
        print(" | ".join(str(cell) for cell in row))
    print("\nДвофакторний дисперсійний аналіз (θ_off × N)")
    for row in anova_rows:
        print(" | ".join(f"{cell:.4g}" if isinstance(cell, float) else str(cell) for cell in row))

    if not pareto:
        print("Жоден варіант не задовольняє P_зн ≤ 30% за верхньою межею ДІ")
    else:
        print("\nПарето-оптимальні варіанти (мінімум і частоти підключень, і частки часу з R=1 при P_зн ≤ 30%):")
        for index in sorted(pareto, key=lambda index: resource_costs(samples_list[index])):
            activations, accelerated = resource_costs(samples_list[index])
            print(f"  {labels[index]}: підключень {activations:.3f} 1/с "
                  f"({relative_change_percent(samples_list[index]['activations_per_second'], baseline['activations_per_second']):+.1f}%), "
                  f"частка часу R=1 {accelerated:.4f} "
                  f"({relative_change_percent(samples_list[index]['accelerated_time_fraction'], baseline['accelerated_time_fraction']):+.1f}%), "
                  f"P_зн {samples_list[index]['destroyed_fraction'].mean():.4f}")
    print(f"Час виконання: {time.perf_counter() - start:.1f} с")


if __name__ == "__main__":
    main()
