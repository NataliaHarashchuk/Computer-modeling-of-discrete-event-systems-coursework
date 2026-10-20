import time

import numpy as np

from experiments.common import (
    base_config_from_run_params,
    confidence_half_width,
    load_run_params,
    run_replications,
    write_csv,
)
from experiments.plotting import (
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    add_horizontal_reference,
    add_legend,
    create_figure,
    label_axes,
    save_figure,
    series_color,
    simulation_caption,
    use_percent_axis,
)
from simulation.config import CONTROL_MODES

DESTRUCTION_REQUIREMENT = 0.30
BAR_WIDTH = 0.36
SERVICE_LAWS = (
    ("constant", "Детерміноване 5 / 4 мс"),
    ("exponential", "Експоненційне, середнє 5 / 4 мс"),
)
MODE_LABELS = {
    "adaptive": "З керуванням",
    "never": "Без керування",
    "always": "Завжди прискорено",
}
LAW_SEED_OFFSET = 50_000
MODE_SEED_STRIDE = 10_000
RESPONSES = (
    ("destroyed_fraction", "P_зн", True),
    ("activations_per_second", "Підключень ресурсу, 1/с", False),
    ("accelerated_time_fraction", "Частка часу R=1", True),
    ("mean_sojourn_time", "Середній час перебування, мс", False),
)
PLOTTED_RESPONSES = (
    ("destroyed_fraction", "Частка знищених пакетів P_зн", "Частка знищених пакетів, %"),
    ("accelerated_time_fraction", "Частка часу з підключеним ресурсом", "Частка часу з R=1, %"),
)


def run_law_and_mode(params, law, mode_index, mode):
    config = base_config_from_run_params(params, control_mode=mode, service_distribution=law)
    seed_offset = LAW_SEED_OFFSET + MODE_SEED_STRIDE * mode_index
    results = run_replications(config, params["replications"], seed_offset)
    return {key: np.array([result[key] for result in results]) for key, _, _ in RESPONSES}


def build_rows(samples):
    rows = []
    for law, law_label in SERVICE_LAWS:
        for mode in CONTROL_MODES:
            row = [law_label, MODE_LABELS[mode]]
            for key, _, _ in RESPONSES:
                values = samples[(law, mode)][key]
                row.extend([float(values.mean()), confidence_half_width(values)])
            row.append("так" if samples[(law, mode)]["destroyed_fraction"].mean()
                       + confidence_half_width(samples[(law, mode)]["destroyed_fraction"])
                       <= DESTRUCTION_REQUIREMENT else "ні")
            rows.append(row)
    return rows


def header():
    columns = ["Закон часу обслуговування", "Режим керування"]
    for _, label, _ in RESPONSES:
        columns.extend([f"{label}: середнє", f"{label}: напівширина 95% ДІ"])
    return columns + ["Виконується P_зн ≤ 30%"]


def plot_sensitivity(samples, params):
    caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], params["replications"],
        "стовпчики: середнє та 95% ДІ; незалежні потоки випадкових чисел",
    )
    figure, axes = create_figure(caption, rows=1, columns=2, width=12.0, height=4.8)
    positions = np.arange(len(CONTROL_MODES))
    for axis, (key, title, y_label) in zip(axes, PLOTTED_RESPONSES):
        for slot, (law, law_label) in enumerate(SERVICE_LAWS):
            means = [samples[(law, mode)][key].mean() for mode in CONTROL_MODES]
            halves = [confidence_half_width(samples[(law, mode)][key]) for mode in CONTROL_MODES]
            offsets = positions + (slot - 0.5) * BAR_WIDTH
            axis.bar(offsets, means, width=BAR_WIDTH, color=series_color(slot), edgecolor=SURFACE_COLOR,
                     yerr=halves, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.2, "capsize": 3},
                     label=law_label)
            for offset, mean in zip(offsets, means):
                axis.annotate(f"{mean:.1%}", (offset, mean), xytext=(0, 4), textcoords="offset points",
                              ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
        axis.set_xticks(list(positions), [MODE_LABELS[mode] for mode in CONTROL_MODES])
        axis.grid(axis="x", visible=False)
        label_axes(axis, title, "Режим керування", y_label)
        use_percent_axis(axis)
        axis.set_ylim(0, 1.45)
        if key == "destroyed_fraction":
            add_horizontal_reference(axis, DESTRUCTION_REQUIREMENT, "Вимога: 30%")
        add_legend(axis, "upper left")
    figure.tight_layout()
    return save_figure(figure, "sensitivity_service_law.png")


def main():
    started = time.perf_counter()
    params = load_run_params()
    samples = {
        (law, mode): run_law_and_mode(params, law, mode_index, mode)
        for law, _ in SERVICE_LAWS
        for mode_index, mode in enumerate(CONTROL_MODES)
    }
    rows = build_rows(samples)
    write_csv("sensitivity_service_law.csv", header(), rows)
    plot_sensitivity(samples, params)
    print("Чутливість до закону розподілу часу обслуговування")
    for row in rows:
        print(" | ".join(f"{cell:.4g}" if isinstance(cell, float) else str(cell) for cell in row))
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
