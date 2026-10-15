import math
import time

import numpy as np

from experiments.common import (
    base_config_from_run_params,
    load_run_params,
    run_replications,
    save_run_params,
    write_csv,
)
from experiments.plotting import (
    add_horizontal_reference,
    add_vertical_reference,
    create_figure,
    label_axes,
    plot_line,
    save_figure,
    simulation_caption,
    use_percent_axis,
)

PILOT_REPLICATIONS = 20
PILOT_SEED_OFFSET = 700
CONFIDENCE_PROBABILITY = 0.95
RELATIVE_PRECISION = 0.01
MINIMUM_REPLICATIONS = 20
PRECISION_GRID = np.linspace(0.0025, 0.03, 120)

RESPONSES = (
    ("destroyed_fraction", "Частка знищених пакетів P_зн"),
    ("destroyed_per_second", "Частота знищення, пакетів/с"),
    ("activations_per_second", "Частота підключення ресурсу, 1/с"),
    ("accelerated_time_fraction", "Частка часу з увімкненим ресурсом"),
)


def chebyshev_replications(standard_deviation, precision, confidence_probability=CONFIDENCE_PROBABILITY):
    return standard_deviation ** 2 / (precision ** 2 * (1 - confidence_probability))


def plot_required_replications(statistics, chosen_replications, caption):
    figure, axis = create_figure(caption, width=10.0, height=4.8)
    for slot, (key, title) in enumerate(RESPONSES):
        mean = statistics[key]["mean"]
        deviation = statistics[key]["std"]
        required = [chebyshev_replications(deviation, share * mean) for share in PRECISION_GRID]
        plot_line(axis, PRECISION_GRID, required, title, slot, markers=False)
    add_horizontal_reference(axis, chosen_replications, f"Обрана кількість прогонів: {chosen_replications}")
    add_vertical_reference(axis, RELATIVE_PRECISION, f"Задана точність ε = {RELATIVE_PRECISION:.0%} середнього")
    axis.lines[-1].set_color("#0b0b0b")
    axis.lines[-1].set_linestyle(":")
    axis.set_yscale("log")
    use_percent_axis(axis, which="x", decimals=1)
    label_axes(axis, "Необхідна кількість прогонів за формулою (3.4), β = 0.95",
               "Точність ε, % від середнього значення відгуку", "Кількість прогонів p")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return save_figure(figure, "replications_required.png")


def main():
    started = time.time()
    params = load_run_params()
    base_config = base_config_from_run_params(params)
    results = run_replications(base_config, PILOT_REPLICATIONS, PILOT_SEED_OFFSET)

    statistics = {}
    rows = []
    for key, title in RESPONSES:
        samples = np.array([result[key] for result in results])
        mean = float(samples.mean())
        deviation = float(samples.std(ddof=1))
        precision = RELATIVE_PRECISION * mean
        required = chebyshev_replications(deviation, precision)
        statistics[key] = {"mean": mean, "std": deviation, "precision": precision, "required": required}
        rows.append([
            title, f"{mean:.5f}", f"{deviation:.5f}", f"{deviation / mean:.4%}", f"{precision:.5f}",
            f"{required:.2f}", math.ceil(required),
        ])

    required_maximum = max(math.ceil(entry["required"]) for entry in statistics.values())
    chosen_replications = max(MINIMUM_REPLICATIONS, required_maximum)
    write_csv(
        "replications.csv",
        ["Відгук", "Середнє", "Стандартне відхилення σ", "Коефіцієнт варіації", "Точність ε (1% середнього)",
         "p за (3.4)", "p (округлено вгору)"],
        rows,
    )

    caption = simulation_caption(
        params["simulation_time_ms"], params["warmup_time_ms"], PILOT_REPLICATIONS,
        extra="пілотні прогони, базовий сценарій B8",
    )
    plot_required_replications(statistics, chosen_replications, caption)

    params["replications"] = chosen_replications
    params["replication_study"] = {
        "pilot_replications": PILOT_REPLICATIONS,
        "pilot_seed_offset": PILOT_SEED_OFFSET,
        "confidence_probability": CONFIDENCE_PROBABILITY,
        "relative_precision": RELATIVE_PRECISION,
        "minimum_replications": MINIMUM_REPLICATIONS,
        "responses": statistics,
    }
    save_run_params(params)

    print(f"Пілотні прогони: {PILOT_REPLICATIONS}, T_warm = {params['warmup_time_ms']:.0f} мс, "
          f"T_mod = {params['simulation_time_ms']:.0f} мс")
    for row in rows:
        print("  " + " | ".join(str(value) for value in row))
    print(f"Необхідно за (3.4): {required_maximum}; обрано прогонів: {chosen_replications}")
    print(f"Час виконання: {time.time() - started:.1f} с")


if __name__ == "__main__":
    main()
