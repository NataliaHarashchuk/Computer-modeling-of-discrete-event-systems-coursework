import math
import time

import numpy as np
from scipy import stats

from experiments.common import (
    DESTRUCTION_REQUIREMENT, MODE_LABELS, base_config_from_run_params, confidence_half_width, format_p_value,
    load_run_params, parallel_map, run_replications, write_csv,
)
from experiments.plotting import (
    add_horizontal_reference, add_legend, add_vertical_reference, create_figure, label_axes,
    plot_bars, save_figure, series_color, simulation_caption, use_percent_axis,
)
from simulation.config import CONTROL_MODES
from simulation.transmission_model import TransmissionModel

SIGNIFICANCE_LEVEL = 0.05
DESTRUCTION_THRESHOLD_MS = 10.0
HISTOGRAM_BIN_WIDTH_MS = 0.5
HISTOGRAM_LOWER_EDGE_MS = 7.0
HISTOGRAM_MODES = ("adaptive", "never")
HISTOGRAM_SEED = 1
MODE_SEED_OFFSETS = {"adaptive": 0, "never": 10_000, "always": 20_000}

ALL_OUTPUTS = (
    ("destroyed_fraction", "Частка знищених пакетів P_зн"),
    ("destroyed_per_second", "Знищених пакетів, 1/с"),
    ("activation_count", "Кількість підключень ресурсу, шт."),
    ("activations_per_second", "Підключень ресурсу, 1/с"),
    ("deactivation_count", "Кількість відключень ресурсу, шт."),
    ("accelerated_time_fraction", "Частка часу з прискоренням R=1"),
    ("accelerated_packets_fraction", "Частка пакетів, обслужених з прискоренням"),
    ("limit_violation_time_fraction", "Частка часу, коли у вікні 100 пакетів знищено понад 30%"),
    ("limit_violation_exits_fraction", "Частка виходів пакетів, коли у вікні 100 пакетів знищено понад 30%"),
    ("max_window_destroyed_fraction", "Найбільша частка знищених у вікні 100 пакетів"),
    ("mean_queue_length_1", "Середня довжина черги К1, пакетів"),
    ("mean_queue_length_2", "Середня довжина черги К2, пакетів"),
    ("max_queue_length_1", "Максимальна довжина черги К1, пакетів"),
    ("max_queue_length_2", "Максимальна довжина черги К2, пакетів"),
    ("mean_waiting_time_1", "Середній час очікування в К1, мс"),
    ("mean_waiting_time_2", "Середній час очікування в К2, мс"),
    ("utilization_1", "Завантаження К1"),
    ("utilization_2", "Завантаження К2"),
    ("mean_sojourn_time", "Середній час перебування пакета, мс"),
    ("throughput_per_second", "Пропускна здатність, пакетів/с"),
)

MODE_OUTPUTS = (
    ("destroyed_fraction", "P_зн"),
    ("destroyed_per_second", "Знищень, 1/с"),
    ("activations_per_second", "Підключень, 1/с"),
    ("accelerated_time_fraction", "Частка часу R=1"),
    ("accelerated_packets_fraction", "Частка пакетів із прискоренням"),
    ("limit_violation_time_fraction", "Частка часу з рівнем знищення >30% у вікні"),
    ("max_window_destroyed_fraction", "Макс. частка знищених у вікні"),
    ("mean_queue_length_1", "L1, пакетів"),
    ("mean_waiting_time_1", "W1, мс"),
    ("utilization_1", "U1"),
    ("utilization_2", "U2"),
    ("mean_sojourn_time", "Середній час перебування, мс"),
)

ANOVA_RESPONSES = (
    ("destroyed_fraction", "P_зн"),
    ("mean_sojourn_time", "Середній час перебування, мс"),
)


def column(replication_results, key):
    return np.array([result[key] for result in replication_results], dtype=float)


def print_table(header, rows):
    print("\t".join(header))
    for row in rows:
        print("\t".join(format_cell(cell) for cell in row))


def format_cell(cell):
    return f"{cell:.6g}" if isinstance(cell, float) else str(cell)


def build_base_summary(adaptive_results):
    rows = []
    for key, name in ALL_OUTPUTS:
        samples = column(adaptive_results, key)
        rows.append([
            name, float(samples.mean()), confidence_half_width(samples),
            float(samples.min()), float(samples.max()),
        ])
    return rows


def build_modes_table(results_by_mode):
    rows = []
    for key, name in MODE_OUTPUTS:
        for mode in CONTROL_MODES:
            samples = column(results_by_mode[mode], key)
            rows.append([name, MODE_LABELS[mode], float(samples.mean()), confidence_half_width(samples)])
    return rows


def plot_modes(results_by_mode, params):
    caption = simulation_caption(params["simulation_time_ms"], params["warmup_time_ms"], params["replications"],
                                 "стовпчики: середнє та 95% ДІ")
    figure, axes = create_figure(caption, rows=2, columns=2, width=11.0, height=8.0)
    labels = [MODE_LABELS[mode] for mode in CONTROL_MODES]
    panels = (
        ("destroyed_fraction", "Частка знищених пакетів P_зн", "Частка знищених пакетів, %", 0, True),
        ("destroyed_per_second", "Інтенсивність знищення пакетів", "Знищених пакетів, 1/с", 1, False),
        ("accelerated_time_fraction", "Частка часу з прискоренням R=1", "Частка часу, %", 2, True),
        ("mean_sojourn_time", "Середній час перебування пакета", "Час перебування, мс", 3, False),
    )
    for axis, (key, title, y_label, slot, percent) in zip(axes, panels):
        means = [column(results_by_mode[mode], key).mean() for mode in CONTROL_MODES]
        half_widths = [confidence_half_width(column(results_by_mode[mode], key)) for mode in CONTROL_MODES]
        value_format = "{:.1%}" if percent else "{:.3f}"
        plot_bars(axis, labels, means, half_widths, slot, value_format)
        label_axes(axis, title, "Режим керування", y_label)
        if percent:
            use_percent_axis(axis)
        axis.set_ylim(0, max(m + h for m, h in zip(means, half_widths)) * 1.18)
        if key == "destroyed_fraction":
            add_horizontal_reference(axis, DESTRUCTION_REQUIREMENT, "Вимога: 30%")
            axis.set_ylim(0, max(DESTRUCTION_REQUIREMENT, max(means)) * 1.18)
            add_legend(axis, "upper left")
    figure.tight_layout()
    return save_figure(figure, "base_modes.png")


def anova_one_way(groups):
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    group_count = len(groups)
    total_count = all_values.size
    sum_squares_between = float(sum(group.size * (group.mean() - grand_mean) ** 2 for group in groups))
    sum_squares_within = float(sum(((group - group.mean()) ** 2).sum() for group in groups))
    df_between = group_count - 1
    df_within = total_count - group_count
    mean_square_between = sum_squares_between / df_between
    mean_square_within = sum_squares_within / df_within
    f_value = mean_square_between / mean_square_within
    f_critical = float(stats.f.ppf(1 - SIGNIFICANCE_LEVEL, df_between, df_within))
    p_value = float(stats.f.sf(f_value, df_between, df_within))
    effect_share = sum_squares_between / (sum_squares_between + sum_squares_within)
    return {
        "ss_between": sum_squares_between, "ss_within": sum_squares_within,
        "df_between": df_between, "df_within": df_within,
        "ms_between": mean_square_between, "ms_within": mean_square_within,
        "f": f_value, "f_critical": f_critical, "p": p_value, "eta_squared": effect_share,
    }


def build_anova_rows(results_by_mode):
    rows = []
    for key, name in ANOVA_RESPONSES:
        groups = [column(results_by_mode[mode], key) for mode in CONTROL_MODES]
        manual = anova_one_way(groups)
        reference = stats.f_oneway(*groups)
        if not (np.isclose(manual["f"], reference.statistic, rtol=1e-9)
                and np.isclose(manual["p"], reference.pvalue, rtol=1e-6, atol=1e-300)):
            raise AssertionError(f"ANOVA mismatch for {key}: {manual['f']} vs {reference.statistic}")
        significant = manual["f"] > manual["f_critical"]
        conclusion = (
            "F > F_крит: режим керування істотно впливає на відгук (H0 відхиляється)" if significant
            else "F <= F_крит: істотного впливу режиму керування не виявлено"
        )
        rows.append([
            name, manual["ss_between"], manual["ss_within"], manual["df_between"], manual["df_within"],
            manual["ms_between"], manual["ms_within"], manual["f"], manual["f_critical"], format_p_value(manual["p"]),
            float(reference.statistic), format_p_value(float(reference.pvalue)), manual["eta_squared"], conclusion,
        ])
    return rows


def collect_sojourn_times(config):
    model = TransmissionModel(config)
    sojourn_times = []
    destroyed_flags = []
    warmup_time = config.warmup_time

    def record(packet):
        if packet.exit_time >= warmup_time:
            sojourn_times.append(packet.sojourn_time)
            destroyed_flags.append(packet.destroyed)

    model.sink.add_exit_listener(record)
    model.simulate()
    return np.array(sojourn_times), np.array(destroyed_flags, dtype=bool)


def right_closed_bin_shares(samples, edges, tolerance):
    bin_indices = np.searchsorted(edges, samples - tolerance, side="left") - 1
    counts = np.bincount(bin_indices, minlength=len(edges) - 1)[:len(edges) - 1]
    return counts / samples.size


def plot_sojourn_histogram(sojourn_by_mode, params, tolerance):
    caption = simulation_caption(params["simulation_time_ms"], params["warmup_time_ms"], 1,
                                 f"одна реплікація на режим, seed = {HISTOGRAM_SEED}; інтервали (a; b], "
                                 f"T = 10 мс рівно — доставлені пакети")
    figure, axes = create_figure(caption, rows=1, columns=2, width=11.0, height=4.8, sharex=True)
    upper_edge = max(samples.max() for samples, _ in sojourn_by_mode.values())
    shortest_sojourn = min(samples.min() for samples, _ in sojourn_by_mode.values())
    lower_edge = min(HISTOGRAM_LOWER_EDGE_MS,
                     math.floor((shortest_sojourn - tolerance) / HISTOGRAM_BIN_WIDTH_MS) * HISTOGRAM_BIN_WIDTH_MS)
    edges = np.arange(lower_edge, upper_edge + 2 * HISTOGRAM_BIN_WIDTH_MS, HISTOGRAM_BIN_WIDTH_MS)
    shares = {mode: right_closed_bin_shares(samples, edges, tolerance) for mode, (samples, _) in sojourn_by_mode.items()}
    y_limit = max(share.max() for share in shares.values()) * 1.15
    destroyed_bins = edges[:-1] >= DESTRUCTION_THRESHOLD_MS
    for axis, mode in zip(axes, HISTOGRAM_MODES):
        samples, destroyed = sojourn_by_mode[mode]
        axis.bar(edges[:-1][~destroyed_bins], shares[mode][~destroyed_bins], width=HISTOGRAM_BIN_WIDTH_MS,
                 align="edge", color=series_color(0), edgecolor="#ffffff", linewidth=0.6,
                 label=f"Доставлено, T ≤ 10 мс: {1 - destroyed.mean():.1%}")
        axis.bar(edges[:-1][destroyed_bins], shares[mode][destroyed_bins], width=HISTOGRAM_BIN_WIDTH_MS,
                 align="edge", color=series_color(1), edgecolor="#ffffff", linewidth=0.6,
                 label=f"Знищено, T > 10 мс: {destroyed.mean():.1%}")
        add_vertical_reference(axis, DESTRUCTION_THRESHOLD_MS, "Поріг знищення: 10 мс")
        label_axes(axis, f"Розподіл часу перебування: {MODE_LABELS[mode].lower()} (n = {samples.size})",
                   "Час перебування пакета, мс", "Частка пакетів")
        axis.set_ylim(0, y_limit)
        add_legend(axis, "upper right")
    figure.tight_layout()
    return save_figure(figure, "base_sojourn_histogram.png")


def main():
    started_at = time.perf_counter()
    params = load_run_params()
    replications = params["replications"]
    configs = {
        mode: base_config_from_run_params(params, control_mode=mode) for mode in CONTROL_MODES
    }
    results_by_mode = {
        mode: run_replications(configs[mode], replications, MODE_SEED_OFFSETS[mode]) for mode in CONTROL_MODES
    }

    summary_rows = build_base_summary(results_by_mode["adaptive"])
    write_csv("base_summary.csv",
              ["Характеристика", "Середнє", "Напівширина 95% ДІ", "Мінімум", "Максимум"], summary_rows)
    print(f"Базовий сценарій B8 (з керуванням), прогонів: {replications}")
    print_table(["Характеристика", "Середнє", "Півширина ДІ", "Мін", "Макс"], summary_rows)

    modes_rows = build_modes_table(results_by_mode)
    write_csv("base_modes.csv", ["Характеристика", "Режим керування", "Середнє", "Напівширина 95% ДІ"], modes_rows)
    print("\nПорівняння режимів керування")
    print_table(["Характеристика", "Режим", "Середнє", "Півширина ДІ"], modes_rows)
    plot_modes(results_by_mode, params)

    anova_rows = build_anova_rows(results_by_mode)
    write_csv("base_anova.csv", [
        "Відгук", "SS між групами", "SS всередині груп", "df між групами", "df всередині груп",
        "MS між групами", "MS всередині груп", "F", "F_крит (alpha=0.05)", "p-value",
        "F (scipy f_oneway)", "p-value (scipy f_oneway)", "η² (частка SS між групами)", "Висновок",
    ], anova_rows)
    print("\nДисперсійний аналіз (режим керування, 3 рівні, незалежні потоки випадкових чисел)")
    for row in anova_rows:
        print(f"{row[0]}: F = {row[7]:.4g}, F_крит = {row[8]:.4g}, p {row[9] if row[9].startswith('<') else '= ' + row[9]}, η² = {row[12]:.4f}; {row[13]}")

    histogram_configs = [
        base_config_from_run_params(params, control_mode=mode, seed=HISTOGRAM_SEED) for mode in HISTOGRAM_MODES
    ]
    sojourn_by_mode = dict(zip(HISTOGRAM_MODES, parallel_map(collect_sojourn_times, histogram_configs)))
    plot_sojourn_histogram(sojourn_by_mode, params, histogram_configs[0].time_tolerance)
    for mode, (samples, destroyed) in sojourn_by_mode.items():
        print(f"Час перебування, {mode}: n = {samples.size}, середнє = {samples.mean():.4f} мс, "
              f"макс = {samples.max():.2f} мс, частка знищених = {destroyed.mean():.4f}")

    print(f"\nЧас виконання: {time.perf_counter() - started_at:.1f} с")


if __name__ == "__main__":
    main()
