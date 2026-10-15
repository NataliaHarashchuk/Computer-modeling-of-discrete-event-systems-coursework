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
    use_percent_axis,
)

DESTRUCTION_LIMIT = 0.30
RESPONSE_KEYS = (
    "destroyed_fraction",
    "destroyed_per_second",
    "activations_per_second",
    "accelerated_time_fraction",
    "accelerated_packets_fraction",
    "mean_sojourn_time",
)
HYSTERESIS_OFF_THRESHOLDS = [round(0.30 - 0.025 * step, 3) for step in range(9)]

ADAPTIVE_VARIANTS = (
    ("Базовий (30/30, N=100)", dict(activation_threshold=0.30, deactivation_threshold=0.30, window_size=100)),
    ("Гістерезис 30/20", dict(activation_threshold=0.30, deactivation_threshold=0.20, window_size=100)),
    ("Гістерезис 30/25", dict(activation_threshold=0.30, deactivation_threshold=0.25, window_size=100)),
    ("Гістерезис 25/15", dict(activation_threshold=0.25, deactivation_threshold=0.15, window_size=100)),
    ("Поріг 25 без гістерезису", dict(activation_threshold=0.25, deactivation_threshold=0.25, window_size=100)),
    ("Вікно N=300", dict(activation_threshold=0.30, deactivation_threshold=0.30, window_size=300)),
    ("Гістерезис 30/20 + N=300", dict(activation_threshold=0.30, deactivation_threshold=0.20, window_size=300)),
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
    "Середній час перебування, мс",
    "Зміна P_зн відносно базового, %",
    "Зміна підключень/с відносно базового, %",
    "Зміна частки часу R=1 відносно базового, %",
    "p-value P_зн (парний t)",
    "p-value підключень/с (парний t)",
    "p-value частки часу R=1 (парний t)",
    "Виконується P_зн ≤ 30%",
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
    caption = simulation_caption_for(params, "керування adaptive, спільні випадкові числа")
    figure, axes = create_figure(caption, rows=1, columns=3, width=15.5, height=5.0)
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
        rows.append([
            round(off_threshold * 100, 1),
            round(destroyed.mean() * 100, 3), round(confidence_half_width(destroyed) * 100, 3),
            round(activations.mean(), 3), round(confidence_half_width(activations), 3),
            round(accelerated.mean() * 100, 3), round(confidence_half_width(accelerated) * 100, 3),
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


def choose_recommendation(labels, samples_list):
    candidates = [
        index for index, samples in enumerate(samples_list)
        if meets_requirement(samples["destroyed_fraction"])
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda index: (
        samples_list[index]["activations_per_second"].mean(),
        samples_list[index]["accelerated_time_fraction"].mean(),
    ))


def main():
    start = time.perf_counter()
    params = load_run_params()

    labels = [label for label, _ in ADAPTIVE_VARIANTS]
    samples_list = [run_variant(params, overrides) for _, overrides in ADAPTIVE_VARIANTS]
    baseline = samples_list[0]
    reference_samples = [(label, run_variant(params, overrides)) for label, overrides in REFERENCE_VARIANTS]

    rows = [comparison_row(label, samples, baseline) for label, samples in zip(labels, samples_list)]
    rows += [comparison_row(label, samples, baseline) for label, samples in reference_samples]
    write_csv("proposal_comparison.csv", COMPARISON_HEADER, rows)
    plot_comparison(
        ["Базовий 30/30", "Гіст. 30/20", "Гіст. 30/25", "Гіст. 25/15", "Поріг 25/25", "N=300", "Гіст. 30/20 + N=300"],
        samples_list, params,
    )

    sweep_samples = run_hysteresis_sweep(params)
    write_csv(
        "proposal_hysteresis.csv",
        ["θ_off, %", "P_зн, %", "P_зн ДІ ±, %", "Підключень/с", "Підключень/с ДІ ±",
         "Частка часу R=1, %", "Частка часу R=1 ДІ ±, %", "Виконується P_зн ≤ 30%"],
        hysteresis_rows(sweep_samples),
    )
    plot_hysteresis(sweep_samples, params)

    for row in rows:
        print(" | ".join(str(cell) for cell in row))

    best_index = choose_recommendation(labels, samples_list)
    if best_index is None:
        print("Жоден варіант не задовольняє P_зн ≤ 30% за верхньою межею ДІ")
    else:
        best = samples_list[best_index]
        print(f"Рекомендація: {labels[best_index]}")
        for key, name in (
            ("destroyed_fraction", "P_зн"),
            ("activations_per_second", "підключень/с"),
            ("accelerated_time_fraction", "частка часу R=1"),
        ):
            print(f"  {name}: {best[key].mean():.4f} проти {baseline[key].mean():.4f} "
                  f"({relative_change_percent(best[key], baseline[key]):+.1f}%)")
    print(f"Час виконання: {time.perf_counter() - start:.1f} с")


if __name__ == "__main__":
    main()
