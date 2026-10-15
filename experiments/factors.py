import argparse
import time

import numpy as np
from scipy import stats

from experiments.common import (
    base_config_from_run_params,
    load_run_params,
    run_replications,
    summarize_replications,
    write_csv,
)
from experiments.plotting import (
    add_horizontal_reference,
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

REQUIREMENT_DESTROYED_FRACTION = 0.30
MAX_POLYNOMIAL_DEGREE = 3
MIN_ADJUSTED_R_SQUARED_GAIN = 0.01
SIGNIFICANCE_LEVEL = 0.05

INTERVAL_CENTERS = [5.0 + 0.5 * step for step in range(11)]
REGRESSION_UPPER_BOUND_MS = 8.0
PLATEAU_LOWER_BOUND_MS = 8.0
REGRESSION_CENTERS = [center for center in INTERVAL_CENTERS if center <= REGRESSION_UPPER_BOUND_MS]
PLATEAU_CENTERS = [center for center in INTERVAL_CENTERS if center >= PLATEAU_LOWER_BOUND_MS]
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
FRACTION_RESPONSES = ("destroyed_fraction", "accelerated_time_fraction")
RESPONSE_KEYS = tuple(RESPONSE_LABELS)


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


def level_header(factor_name, response_keys):
    header = [factor_name]
    for key in response_keys:
        header.extend([f"{key}_mean", f"{key}_half_width"])
    return header


def means_and_halves(summaries, key):
    means = [summary[key]["mean"] for summary in summaries]
    halves = [summary[key]["half_width"] for summary in summaries]
    return means, halves


def fit_polynomial(x_values, y_values, degree):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    coefficients = np.polyfit(x_values, y_values, degree)
    predicted = np.polyval(coefficients, x_values)
    residual_sum = float(np.sum((y_values - predicted) ** 2))
    total_sum = float(np.sum((y_values - y_values.mean()) ** 2))
    points = len(x_values)
    explained_sum = total_sum - residual_sum
    residual_dof = points - degree - 1
    correlation_index = float(np.sqrt(max(0.0, 1.0 - residual_sum / total_sum))) if total_sum > 0 else 0.0
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (points - 1) / residual_dof if residual_dof > 0 else float("nan")
    if residual_dof > 0 and residual_sum > 0:
        f_statistic = (explained_sum / degree) / (residual_sum / residual_dof)
        p_value = float(stats.f.sf(f_statistic, degree, residual_dof))
    else:
        f_statistic = float("inf")
        p_value = 0.0
    return {
        "degree": degree,
        "coefficients": coefficients,
        "correlation_index": correlation_index,
        "adjusted_r_squared": adjusted_r_squared,
        "f_statistic": float(f_statistic),
        "p_value": p_value,
    }


def choose_degree(fits):
    significant_fits = [fit for fit in fits if fit["p_value"] < SIGNIFICANCE_LEVEL] or fits
    best_adjusted_r_squared = max(fit["adjusted_r_squared"] for fit in significant_fits)
    for fit in significant_fits:
        if best_adjusted_r_squared - fit["adjusted_r_squared"] < MIN_ADJUSTED_R_SQUARED_GAIN:
            return fit
    return significant_fits[-1]


def format_equation(coefficients):
    degree = len(coefficients) - 1
    terms = []
    for position, coefficient in enumerate(coefficients):
        power = degree - position
        sign = "-" if coefficient < 0 else "+"
        magnitude = f"{abs(coefficient):.4g}"
        if power == 0:
            body = magnitude
        elif power == 1:
            body = f"{magnitude}x"
        else:
            body = f"{magnitude}x^{power}"
        terms.append((sign, body))
    first_sign, first_body = terms[0]
    text = ("-" if first_sign == "-" else "") + first_body
    for sign, body in terms[1:]:
        text += f" {sign} {body}"
    return text


def padded_coefficients(coefficients):
    padded = [""] * (MAX_POLYNOMIAL_DEGREE + 1)
    for position, coefficient in enumerate(reversed(coefficients)):
        padded[position] = f"{coefficient:.6g}"
    return padded


def extremum_inside(x_values, y_values):
    y_values = np.asarray(y_values)
    index = int(np.argmax(y_values))
    minimum_index = int(np.argmin(y_values))
    return {
        "argmax_x": x_values[index],
        "max_y": float(y_values[index]),
        "argmin_x": x_values[minimum_index],
        "min_y": float(y_values[minimum_index]),
        "monotonic_increasing": bool(np.all(np.diff(y_values) >= 0)),
        "monotonic_decreasing": bool(np.all(np.diff(y_values) <= 0)),
    }


def apply_fraction_axis(axis, key):
    if key in FRACTION_RESPONSES:
        use_percent_axis(axis, decimals=0)


def values_at_centers(centers, values, selected_centers):
    value_by_center = dict(zip(centers, values))
    return [value_by_center[center] for center in selected_centers]


def describe_plateau(summaries):
    plateau = {}
    for key in RESPONSE_KEYS:
        means, halves = means_and_halves(summaries, key)
        plateau_means = values_at_centers(INTERVAL_CENTERS, means, PLATEAU_CENTERS)
        plateau_halves = values_at_centers(INTERVAL_CENTERS, halves, PLATEAU_CENTERS)
        level = float(np.mean(plateau_means))
        plateau[key] = {
            "level": level,
            "max_deviation": float(max(abs(mean - level) for mean in plateau_means)),
            "max_upper_bound": float(max(mean + half for mean, half in zip(plateau_means, plateau_halves))),
        }
    return plateau


def write_interval_plateau(plateau):
    write_csv(
        "factor_interval_plateau.csv",
        ["response", "from_ms", "to_ms", "levels", "plateau_level", "max_deviation_from_level",
         "max_upper_confidence_bound"],
        [
            [key, PLATEAU_CENTERS[0], PLATEAU_CENTERS[-1], len(PLATEAU_CENTERS),
             item["level"], item["max_deviation"], item["max_upper_bound"]]
            for key, item in plateau.items()
        ],
    )


def write_interval_regression(summaries, regression_caption):
    regression_rows = []
    chosen_fits = {}
    for key in RESPONSE_KEYS:
        means, _ = means_and_halves(summaries, key)
        regression_means = values_at_centers(INTERVAL_CENTERS, means, REGRESSION_CENTERS)
        fits = [
            fit_polynomial(REGRESSION_CENTERS, regression_means, degree)
            for degree in range(1, MAX_POLYNOMIAL_DEGREE + 1)
        ]
        chosen = choose_degree(fits)
        chosen_fits[key] = (means, chosen)
        for fit in fits:
            regression_rows.append([
                key, REGRESSION_CENTERS[0], REGRESSION_CENTERS[-1], len(REGRESSION_CENTERS),
                fit["degree"], int(fit is chosen),
                *padded_coefficients(fit["coefficients"]),
                fit["correlation_index"], fit["adjusted_r_squared"], fit["f_statistic"], fit["p_value"],
            ])
    write_csv(
        "factor_interval_regression.csv",
        ["response", "from_ms", "to_ms", "points", "degree", "chosen", "coef_x0", "coef_x1", "coef_x2", "coef_x3",
         "correlation_index_R", "adjusted_R2", "F", "p_value"],
        regression_rows,
    )
    plateau = describe_plateau(summaries)
    write_interval_plateau(plateau)

    figure, axes = create_figure(regression_caption, rows=2, columns=2, width=11.0, height=8.5)
    regression_x = np.linspace(REGRESSION_CENTERS[0], REGRESSION_CENTERS[-1], 200)
    plateau_x = [PLATEAU_CENTERS[0], PLATEAU_CENTERS[-1]]
    for axis, key in zip(axes, RESPONSE_KEYS):
        means, chosen = chosen_fits[key]
        plot_line(axis, INTERVAL_CENTERS, means, "Середнє за прогонами", 0, markers=True, linewidth=0.0)
        equation = format_equation(chosen["coefficients"])
        label = (
            f"Поліном ст. {chosen['degree']} на {REGRESSION_CENTERS[0]:g}–{REGRESSION_CENTERS[-1]:g} мс, "
            f"R = {chosen['correlation_index']:.4f}\ny = {equation}"
        )
        plot_line(axis, regression_x, np.polyval(chosen["coefficients"], regression_x), label, 1, markers=False)
        plateau_level = plateau[key]["level"]
        plateau_label = f"Плато ≥ {PLATEAU_LOWER_BOUND_MS:g} мс: y = {plateau_level:.4g}"
        plot_line(axis, plateau_x, [plateau_level, plateau_level], plateau_label, 2, markers=False)
        add_vertical_reference(axis, PLATEAU_LOWER_BOUND_MS, "Межа ділянок")
        label_axes(axis, RESPONSE_LABELS[key], "Середній інтервал надходження, мс", RESPONSE_AXIS_LABELS[key])
        apply_fraction_axis(axis, key)
        axis.legend(loc="upper right", fontsize=7.5)
    figure.tight_layout()
    save_figure(figure, "factor_interval_regression.png")

    return chosen_fits, plateau


def run_interval_factor(params, replications):
    base = base_config_from_run_params(params, control_mode="adaptive")
    summaries = run_factor([base.with_arrival_center(center) for center in INTERVAL_CENTERS], replications)
    write_csv(
        "factor_interval.csv",
        level_header("arrival_center_ms", RESPONSE_KEYS),
        level_rows(INTERVAL_CENTERS, summaries, RESPONSE_KEYS),
    )
    caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], replications,
        "розкид інтервалу ±3 мс, смуга — 95% ДІ",
    )
    figure, axes = create_figure(caption, rows=2, columns=2, width=11.0, height=8.0)
    for axis, key in zip(axes, RESPONSE_KEYS):
        means, halves = means_and_halves(summaries, key)
        plot_line_with_interval(axis, INTERVAL_CENTERS, means, halves, RESPONSE_LABELS[key], 0)
        label_axes(axis, RESPONSE_LABELS[key], "Середній інтервал надходження, мс", RESPONSE_AXIS_LABELS[key])
        apply_fraction_axis(axis, key)
    figure.tight_layout()
    save_figure(figure, "factor_interval.png")

    regression_caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], replications,
        f"розкид інтервалу ±3 мс, точки — середні за прогонами, регресія на "
        f"{REGRESSION_CENTERS[0]:g}–{REGRESSION_CENTERS[-1]:g} мс, плато ≥ {PLATEAU_LOWER_BOUND_MS:g} мс окремо",
    )
    chosen_fits, plateau = write_interval_regression(summaries, regression_caption)

    report = {}
    for key in RESPONSE_KEYS:
        means, chosen = chosen_fits[key]
        report[key] = {
            "plateau": plateau[key],
            "degree": chosen["degree"],
            "R": chosen["correlation_index"],
            "adjusted_R2": chosen["adjusted_r_squared"],
            "F": chosen["f_statistic"],
            "p": chosen["p_value"],
            "shape": extremum_inside(INTERVAL_CENTERS, means),
            "first": means[0],
            "last": means[-1],
        }
    return report


def run_window_factor(params, replications):
    base = base_config_from_run_params(params, control_mode="adaptive")
    summaries = run_factor([base.with_overrides(window_size=size) for size in WINDOW_SIZES], replications)
    keys = ("destroyed_fraction", "activations_per_second", "accelerated_time_fraction")
    write_csv("factor_window.csv", level_header("window_size", keys), level_rows(WINDOW_SIZES, summaries, keys))
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
        label_axes(axis, RESPONSE_LABELS[key], "Розмір ковзного вікна N, пакетів", RESPONSE_AXIS_LABELS[key])
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
    write_csv("factor_threshold.csv", level_header("threshold", keys), level_rows(THRESHOLDS, summaries, keys))
    caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], replications, "смуга — 95% ДІ",
    )
    figure, axes = create_figure(caption, rows=1, columns=3, width=15.0, height=4.8)
    for axis, key in zip(axes, keys):
        means, halves = means_and_halves(summaries, key)
        plot_line_with_interval(axis, THRESHOLDS, means, halves, RESPONSE_LABELS[key], 0)
        label_axes(axis, RESPONSE_LABELS[key], "Поріг θ (частка знищених у вікні)", RESPONSE_AXIS_LABELS[key])
        apply_fraction_axis(axis, key)
    destroyed_axis = axes[0]
    plot_line(destroyed_axis, THRESHOLDS, THRESHOLDS, "y = θ", 1, markers=False, linewidth=1.4)
    add_horizontal_reference(destroyed_axis, REQUIREMENT_DESTROYED_FRACTION, "Вимога 0.30")
    add_legend(destroyed_axis, "upper left")
    figure.tight_layout()
    save_figure(figure, "factor_threshold.png")
    return {key: means_and_halves(summaries, key)[0] for key in keys}


def print_interval_report(report):
    for key, item in report.items():
        shape = item["shape"]
        print(
            f"{key}: degree={item['degree']} R={item['R']:.4f} adjR2={item['adjusted_R2']:.4f} "
            f"F={item['F']:.2f} p={item['p']:.3g} first={item['first']:.4f} last={item['last']:.4f} "
            f"max@{shape['argmax_x']}={shape['max_y']:.4f} min@{shape['argmin_x']}={shape['min_y']:.4f} "
            f"inc={shape['monotonic_increasing']} dec={shape['monotonic_decreasing']} "
            f"plateau={item['plateau']['level']:.4g} plateau_upper={item['plateau']['max_upper_bound']:.4g}"
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
