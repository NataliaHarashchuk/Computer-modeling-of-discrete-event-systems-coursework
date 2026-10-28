import argparse
import time

import numpy as np
from scipy import stats

from experiments.common import (
    DESTRUCTION_REQUIREMENT,
    base_config_from_run_params,
    load_run_params,
    run_replications,
    summarize_replications,
    write_csv,
)
from experiments.plotting import (
    add_horizontal_reference,
    add_framed_legend,
    add_legend,
    add_vertical_reference,
    create_figure,
    label_axes,
    plot_line,
    plot_line_with_interval,
    save_figure,
    simulation_caption,
    use_percent_axis,
)

MAX_POLYNOMIAL_DEGREE = 4
SIGNIFICANCE_LEVEL = 0.05
MIN_CORRELATION_INDEX = 0.7
ACTIVE_RESPONSE_SHARE = 0.01
NEGATIVE_PREDICTION_TOLERANCE_SHARE = 0.01
PREDICTION_GRID_POINTS = 400
REGRESSION_LEGEND_HEADROOM = 1.5
MINIMUM_REGRESSION_POINTS = 4

INTERVAL_CENTERS = [5.0 + 0.25 * step for step in range(13)] + [8.5, 9.0, 9.5, 10.0]
ARRIVAL_SPREAD_MS = 3.0
WINDOW_SIZES = [10, 20, 50, 100, 150, 200, 300, 500]
THRESHOLDS = [round(0.10 + 0.05 * step, 2) for step in range(9)]

RESPONSE_LABELS = {
    "destroyed_fraction": "Частка знищених пакетів P_зн",
    "destroyed_per_second": "Знищених пакетів, 1/с",
    "activations_per_second": "Частота підключень, 1/с",
    "accelerated_time_fraction": "Частка часу з R = 1",
}
RESPONSE_AXIS_LABELS = {
    "destroyed_fraction": "Частка знищених пакетів P_зн",
    "destroyed_per_second": "Знищених пакетів, 1/с",
    "activations_per_second": "Підключень додаткового ресурсу, 1/с",
    "accelerated_time_fraction": "Частка часу з R = 1",
}
RESPONSE_COLUMN_LABELS = {
    "destroyed_fraction": "P_зн",
    "destroyed_per_second": "Знищених пакетів, 1/с",
    "activations_per_second": "Підключень ресурсу, 1/с",
    "accelerated_time_fraction": "Частка часу з R=1",
}
FRACTION_RESPONSES = ("destroyed_fraction", "accelerated_time_fraction")
RESPONSE_KEYS = tuple(RESPONSE_LABELS)

INTERVAL_FACTOR_LABEL = "Середній інтервал надходження, мс"
WINDOW_FACTOR_LABEL = "Розмір вікна N, пакетів"
THRESHOLD_FACTOR_LABEL = "Поріг θ (частка знищених у вікні)"


def run_level(config, replications):
    return summarize_replications(run_replications(config, replications), RESPONSE_KEYS)


def run_factor(configs_by_level, replications):
    return [run_level(config, replications) for config in configs_by_level]


def level_rows(levels, summaries, response_keys):
    rows = []
    for level, summary in zip(levels, summaries):
        row = [level]
        for key in response_keys:
            row.extend([summary[key]["mean"], summary[key]["half_width"]])
        rows.append(row)
    return rows


def level_header(factor_label, response_keys):
    header = [factor_label]
    for key in response_keys:
        label = RESPONSE_COLUMN_LABELS[key]
        header.extend([f"{label}: середнє", f"{label}: напівширина 95% ДІ"])
    return header


def means_and_halves(summaries, key):
    means = [summary[key]["mean"] for summary in summaries]
    halves = [summary[key]["half_width"] for summary in summaries]
    return means, halves


def normalization(levels):
    center = (max(levels) + min(levels)) / 2
    half_range = (max(levels) - min(levels)) / 2
    return center, half_range


def normalize(values, center, half_range):
    return (np.asarray(values, dtype=float) - center) / half_range


def leave_one_out_error(normalized_x, y_values, degree):
    point_indices = np.arange(len(y_values))
    errors = []
    for left_out in point_indices:
        kept = point_indices != left_out
        coefficients = np.polyfit(normalized_x[kept], y_values[kept], degree)
        errors.append(y_values[left_out] - np.polyval(coefficients, normalized_x[left_out]))
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_polynomial(normalized_x, y_values, degree):
    y_values = np.asarray(y_values, dtype=float)
    coefficients = np.polyfit(normalized_x, y_values, degree)
    predicted = np.polyval(coefficients, normalized_x)
    residual_sum = float(np.sum((y_values - predicted) ** 2))
    total_sum = float(np.sum((y_values - y_values.mean()) ** 2))
    points = len(y_values)
    residual_dof = points - degree - 1
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
    correlation_index = float(np.sqrt(max(0.0, r_squared)))
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (points - 1) / residual_dof if residual_dof > 0 else float("nan")
    if residual_dof > 0 and residual_sum > 0:
        f_statistic = ((total_sum - residual_sum) / degree) / (residual_sum / residual_dof)
        p_value = float(stats.f.sf(f_statistic, degree, residual_dof))
    else:
        f_statistic = float("inf")
        p_value = 0.0
    grid = np.linspace(-1.0, 1.0, PREDICTION_GRID_POINTS)
    negative_floor = -NEGATIVE_PREDICTION_TOLERANCE_SHARE * float(np.max(np.abs(y_values)))
    return {
        "degree": degree,
        "coefficients": coefficients,
        "correlation_index": correlation_index,
        "adjusted_r_squared": adjusted_r_squared,
        "f_statistic": float(f_statistic),
        "p_value": p_value,
        "leave_one_out_error": leave_one_out_error(normalized_x, y_values, degree),
        "non_negative": bool(np.polyval(coefficients, grid).min() >= negative_floor),
    }


def choose_degree(fits):
    admissible = [fit for fit in fits if fit["p_value"] < SIGNIFICANCE_LEVEL and fit["non_negative"]]
    candidates = admissible or [fit for fit in fits if fit["non_negative"]] or fits
    return min(candidates, key=lambda fit: fit["leave_one_out_error"])


def format_equation(coefficients, variable="x"):
    degree = len(coefficients) - 1
    text = ""
    for position, coefficient in enumerate(coefficients):
        power = degree - position
        magnitude = f"{abs(coefficient):.4g}"
        body = magnitude if power == 0 else f"{magnitude}{variable}" if power == 1 else f"{magnitude}{variable}^{power}"
        if position == 0:
            text = ("-" if coefficient < 0 else "") + body
        else:
            text += f" {'-' if coefficient < 0 else '+'} {body}"
    return text


def padded_coefficients(coefficients):
    padded = [""] * (MAX_POLYNOMIAL_DEGREE + 1)
    for position, coefficient in enumerate(reversed(coefficients)):
        padded[position] = f"{coefficient:.6g}"
    return padded


def active_region_end(levels, means):
    threshold = ACTIVE_RESPONSE_SHARE * max(means)
    active_indices = [index for index, mean in enumerate(means) if mean > threshold]
    last_active = max(active_indices) if active_indices else 0
    return max(last_active, MINIMUM_REGRESSION_POINTS - 1)


def analyse_interval_response(levels, means, halves):
    region_end = active_region_end(levels, means)
    region_levels = levels[:region_end + 1]
    region_means = means[:region_end + 1]
    center, half_range = normalization(region_levels)
    normalized_x = normalize(region_levels, center, half_range)
    highest_degree = min(MAX_POLYNOMIAL_DEGREE, len(region_levels) - 2)
    fits = [fit_polynomial(normalized_x, region_means, degree) for degree in range(1, highest_degree + 1)]
    plateau_means = means[region_end + 1:]
    plateau_halves = halves[region_end + 1:]
    return {
        "region_levels": region_levels,
        "center": center,
        "half_range": half_range,
        "fits": fits,
        "chosen": choose_degree(fits),
        "plateau_levels": levels[region_end + 1:],
        "plateau_level": float(np.mean(plateau_means)) if plateau_means else float("nan"),
        "plateau_upper_bound": float(max(mean + half for mean, half in zip(plateau_means, plateau_halves)))
        if plateau_means else float("nan"),
    }


def apply_fraction_axis(axis, key):
    if key in FRACTION_RESPONSES:
        use_percent_axis(axis, decimals=0)


def write_interval_regression(summaries, regression_caption):
    analyses = {}
    regression_rows = []
    plateau_rows = []
    for key in RESPONSE_KEYS:
        means, halves = means_and_halves(summaries, key)
        analysis = analyse_interval_response(INTERVAL_CENTERS, means, halves)
        analyses[key] = analysis
        for fit in analysis["fits"]:
            regression_rows.append([
                RESPONSE_COLUMN_LABELS[key], analysis["region_levels"][0], analysis["region_levels"][-1],
                len(analysis["region_levels"]), analysis["center"], analysis["half_range"],
                fit["degree"], "так" if fit is analysis["chosen"] else "ні",
                *padded_coefficients(fit["coefficients"]),
                fit["correlation_index"], fit["adjusted_r_squared"], fit["f_statistic"], fit["p_value"],
                fit["leave_one_out_error"], "так" if fit["non_negative"] else "ні",
                "так" if fit["correlation_index"] >= MIN_CORRELATION_INDEX else "ні",
            ])
        if analysis["plateau_levels"]:
            plateau_rows.append([
                RESPONSE_COLUMN_LABELS[key], analysis["plateau_levels"][0], analysis["plateau_levels"][-1],
                len(analysis["plateau_levels"]), analysis["plateau_level"], analysis["plateau_upper_bound"],
            ])
    write_csv(
        "factor_interval_regression.csv",
        ["Відгук", "Ділянка регресії від, мс", "Ділянка регресії до, мс", "Точок",
         "c0 (центр ділянки), мс", "Δ (піврозмах ділянки), мс", "Степінь полінома", "Обрано",
         "b0", "b1", "b2", "b3", "b4", "Індекс кореляції R", "Скоригований R²", "F", "p-value",
         "Похибка перехресної перевірки (leave-one-out RMSE)", "Невід'ємний на ділянці", "R ≥ 0.7"],
        regression_rows,
    )
    write_csv(
        "factor_interval_plateau.csv",
        ["Відгук", "Ділянка від, мс", "Ділянка до, мс", "Рівнів", "Середній рівень відгуку",
         "Найбільша верхня межа 95% ДІ"],
        plateau_rows,
    )

    figure, axes = create_figure(regression_caption, rows=2, columns=2, width=11.0, height=8.5)
    for axis, key in zip(axes, RESPONSE_KEYS):
        means, _ = means_and_halves(summaries, key)
        analysis = analyses[key]
        chosen = analysis["chosen"]
        region_start, region_end = analysis["region_levels"][0], analysis["region_levels"][-1]
        plot_line(axis, INTERVAL_CENTERS, means, "Середнє за прогонами", 0, markers=True, linewidth=0.0)
        regression_x = np.linspace(region_start, region_end, PREDICTION_GRID_POINTS)
        regression_y = np.polyval(chosen["coefficients"], normalize(regression_x, analysis["center"], analysis["half_range"]))
        response_unit_note = "; y — частка одиниці" if key in FRACTION_RESPONSES else ""
        label = (
            f"Поліном ст. {chosen['degree']} на {region_start:g}–{region_end:g} мс, R = {chosen['correlation_index']:.4f}\n"
            f"y = {format_equation(chosen['coefficients'])},\n"
            f"x = (c − {analysis['center']:g}) / {analysis['half_range']:g}{response_unit_note}"
        )
        plot_line(axis, regression_x, regression_y, label, 1, markers=False)
        if analysis["plateau_levels"]:
            plateau_x = [analysis["plateau_levels"][0], analysis["plateau_levels"][-1]]
            plot_line(axis, plateau_x, [analysis["plateau_level"]] * 2,
                      f"Плато > {region_end:g} мс: y = {analysis['plateau_level']:.4g}", 2, markers=False)
            add_vertical_reference(axis, region_end, "Межа ділянок")
        label_axes(axis, RESPONSE_LABELS[key], INTERVAL_FACTOR_LABEL, RESPONSE_AXIS_LABELS[key])
        apply_fraction_axis(axis, key)
        axis.set_ylim(top=max(means) * REGRESSION_LEGEND_HEADROOM)
        add_framed_legend(axis, "upper right", fontsize=7.5)
    figure.tight_layout()
    save_figure(figure, "factor_interval_regression.png")
    return analyses


def run_interval_factor(params, replications):
    base = base_config_from_run_params(params, control_mode="adaptive")
    summaries = run_factor([base.with_arrival_center(center, ARRIVAL_SPREAD_MS) for center in INTERVAL_CENTERS],
                           replications)
    write_csv(
        "factor_interval.csv",
        level_header(INTERVAL_FACTOR_LABEL, RESPONSE_KEYS),
        level_rows(INTERVAL_CENTERS, summaries, RESPONSE_KEYS),
    )
    caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], replications,
        f"розкид інтервалу ±{ARRIVAL_SPREAD_MS:g} мс, смуга — 95% ДІ",
    )
    figure, axes = create_figure(caption, rows=2, columns=2, width=11.0, height=8.0)
    for axis, key in zip(axes, RESPONSE_KEYS):
        means, halves = means_and_halves(summaries, key)
        plot_line_with_interval(axis, INTERVAL_CENTERS, means, halves, RESPONSE_LABELS[key], 0)
        label_axes(axis, RESPONSE_LABELS[key], INTERVAL_FACTOR_LABEL, RESPONSE_AXIS_LABELS[key])
        apply_fraction_axis(axis, key)
        if key == "destroyed_fraction":
            add_horizontal_reference(axis, DESTRUCTION_REQUIREMENT, "Вимога: 30%")
            add_legend(axis, "upper right")
    figure.tight_layout()
    save_figure(figure, "factor_interval.png")

    regression_caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], replications,
        f"розкид інтервалу ±{ARRIVAL_SPREAD_MS:g} мс; регресія за нормалізованим фактором (3.6) "
        "на ділянці, де відгук ненульовий",
    )
    return write_interval_regression(summaries, regression_caption)


def run_window_factor(params, replications):
    base = base_config_from_run_params(params, control_mode="adaptive")
    summaries = run_factor([base.with_overrides(window_size=size) for size in WINDOW_SIZES], replications)
    keys = ("destroyed_fraction", "activations_per_second", "accelerated_time_fraction")
    write_csv("factor_window.csv", level_header(WINDOW_FACTOR_LABEL, keys), level_rows(WINDOW_SIZES, summaries, keys))
    caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], replications, "смуга — 95% ДІ",
    )
    figure, axes = create_figure(caption, rows=1, columns=3, width=15.0, height=4.6)
    for axis, key in zip(axes, keys):
        means, halves = means_and_halves(summaries, key)
        plot_line_with_interval(axis, WINDOW_SIZES, means, halves, RESPONSE_LABELS[key], 0)
        axis.set_xscale("log")
        axis.set_xticks(WINDOW_SIZES, [str(size) for size in WINDOW_SIZES])
        axis.minorticks_off()
        label_axes(axis, RESPONSE_LABELS[key], WINDOW_FACTOR_LABEL, RESPONSE_AXIS_LABELS[key])
        apply_fraction_axis(axis, key)
        axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    save_figure(figure, "factor_window.png")
    return {key: means_and_halves(summaries, key)[0] for key in keys}


def run_threshold_factor(params, replications):
    base = base_config_from_run_params(params, control_mode="adaptive")
    configs = [
        base.with_overrides(activation_threshold=threshold, deactivation_threshold=None)
        for threshold in THRESHOLDS
    ]
    summaries = run_factor(configs, replications)
    keys = ("destroyed_fraction", "accelerated_time_fraction", "activations_per_second")
    write_csv("factor_threshold.csv", level_header(THRESHOLD_FACTOR_LABEL, keys),
              level_rows(THRESHOLDS, summaries, keys))
    caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], replications, "смуга — 95% ДІ",
    )
    figure, axes = create_figure(caption, rows=1, columns=3, width=15.0, height=4.8)
    for axis, key in zip(axes, keys):
        means, halves = means_and_halves(summaries, key)
        plot_line_with_interval(axis, THRESHOLDS, means, halves, RESPONSE_LABELS[key], 0)
        label_axes(axis, RESPONSE_LABELS[key], THRESHOLD_FACTOR_LABEL, RESPONSE_AXIS_LABELS[key])
        apply_fraction_axis(axis, key)
    destroyed_axis = axes[0]
    plot_line(destroyed_axis, THRESHOLDS, THRESHOLDS, "y = θ", 1, markers=False, linewidth=1.4)
    add_horizontal_reference(destroyed_axis, DESTRUCTION_REQUIREMENT, "Вимога 0.30")
    add_legend(destroyed_axis, "upper left")
    figure.tight_layout()
    save_figure(figure, "factor_threshold.png")
    return {key: means_and_halves(summaries, key)[0] for key in keys}


def print_interval_report(analyses):
    for key, analysis in analyses.items():
        chosen = analysis["chosen"]
        print(
            f"{key}: ділянка {analysis['region_levels'][0]:g}–{analysis['region_levels'][-1]:g} мс, "
            f"степінь {chosen['degree']}, R = {chosen['correlation_index']:.4f}, "
            f"adjR2 = {chosen['adjusted_r_squared']:.4f}, LOO RMSE = {chosen['leave_one_out_error']:.4g}, "
            f"F = {chosen['f_statistic']:.2f}, p = {chosen['p_value']:.3g}, "
            f"невід'ємний: {chosen['non_negative']}, y = {format_equation(chosen['coefficients'])}, "
            f"плато = {analysis['plateau_level']:.4g}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", choices=("interval", "window", "threshold", "all"), default="all")
    arguments = parser.parse_args()
    params = load_run_params()
    replications = params["replications"]
    selected = ("interval", "window", "threshold") if arguments.factor == "all" else (arguments.factor,)
    for name in selected:
        started = time.perf_counter()
        if name == "interval":
            print_interval_report(run_interval_factor(params, replications))
        elif name == "window":
            for key, values in run_window_factor(params, replications).items():
                print(key, [round(value, 4) for value in values])
        else:
            for key, values in run_threshold_factor(params, replications).items():
                print(key, [round(value, 4) for value in values])
        print(f"{name}: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
