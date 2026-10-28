import math
import time

import numpy as np

from experiments.common import (
    parallel_map,
    replicate_configs,
    save_run_params,
    simulate_with_trace,
    write_csv,
)
from experiments.plotting import (
    add_horizontal_reference,
    add_legend,
    add_vertical_reference,
    create_figure,
    label_axes,
    plot_line,
    save_figure,
    simulation_caption,
    use_percent_axis,
    use_thousands_separator,
)
from simulation.config import ModelConfig
from simulation.transmission_model import MILLISECONDS_PER_SECOND

PILOT_RUNS = 8
HORIZON_MS = 2_000_000.0
TRACE_INTERVAL_MS = 1000.0
RELATIVE_PRECISION = 0.05
STEADY_LEVEL_TAIL_SHARE = 0.5
WELCH_HALF_WINDOW_CANDIDATES = (5, 10, 20, 50, 100, 200)
WARMUP_SAFETY_FACTOR = 2.0
WARMUP_PLOT_MIN_HORIZON_MS = 60_000.0
WARMUP_PLOT_HORIZON_FACTOR = 4.0
WARMUP_ROUNDING_MS = 5_000.0
SIMULATION_TIME_ROUNDING_MS = 100_000.0
SEED_OFFSET = 500
HIGH_LOAD_ARRIVAL_CENTER_MS = 5.0
SCENARIOS = (
    ("base", "Базовий B8, інтервал unif(3; 9) мс", ModelConfig()),
    ("high_load", f"Найбільше навантаження, інтервал unif({HIGH_LOAD_ARRIVAL_CENTER_MS - 3:g}; "
                  f"{HIGH_LOAD_ARRIVAL_CENTER_MS + 3:g}) мс", ModelConfig().with_arrival_center(HIGH_LOAD_ARRIVAL_CENTER_MS)),
)

RESPONSES = (
    ("destroyed_fraction", "Частка знищених пакетів P_зн", "Частка знищених P_зн"),
    ("activations_per_second", "Частота підключення ресурсу", "Підключень за секунду, 1/с"),
    ("accelerated_time_fraction", "Частка часу з увімкненим ресурсом", "Частка часу з R=1"),
)


def cumulative_response(trace, key):
    times = np.array([snapshot["time"] for snapshot in trace])
    observed = np.array([snapshot["observed_time"] for snapshot in trace])
    if key == "destroyed_fraction":
        values = np.array([snapshot["destroyed_fraction"] for snapshot in trace])
    elif key == "activations_per_second":
        counts = np.array([snapshot["activation_count"] for snapshot in trace], dtype=float)
        values = np.divide(counts * MILLISECONDS_PER_SECOND, observed, out=np.zeros_like(counts), where=observed > 0)
    else:
        accelerated = np.array([snapshot["accelerated_time"] for snapshot in trace])
        values = np.divide(accelerated, observed, out=np.zeros_like(accelerated), where=observed > 0)
    return times, values


def interval_response(trace, key):
    times = np.array([snapshot["time"] for snapshot in trace])
    durations = np.diff(times)
    if key == "destroyed_fraction":
        exited = np.diff([snapshot["exited"] for snapshot in trace])
        destroyed = np.diff([snapshot["destroyed"] for snapshot in trace])
        values = np.divide(destroyed, exited, out=np.zeros(len(exited)), where=exited > 0)
    elif key == "activations_per_second":
        values = np.diff([snapshot["activation_count"] for snapshot in trace]) / durations * MILLISECONDS_PER_SECOND
    else:
        values = np.diff([snapshot["accelerated_time"] for snapshot in trace]) / durations
    return times[1:], values


def first_time_after_which_always(times, condition):
    violations = np.flatnonzero(~condition)
    if violations.size == 0:
        return float(times[0])
    if violations[-1] + 1 >= len(times):
        return math.inf
    return float(times[violations[-1] + 1])


def round_up(value, step):
    return math.ceil(value / step) * step


def welch_moving_average(values, half_window):
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    indices = np.arange(len(values) - half_window)
    reach = np.minimum(indices, half_window)
    window_sums = cumulative[indices + reach + 1] - cumulative[indices - reach]
    return window_sums / (2 * reach + 1)


def smooth_until_tail_within_precision(ensemble_interval, interval_times, tail_start_time, tail_level, precision):
    for half_window in WELCH_HALF_WINDOW_CANDIDATES:
        smoothed = welch_moving_average(ensemble_interval, half_window)
        smoothed_times = interval_times[:len(smoothed)]
        deviation = np.abs(smoothed - tail_level)
        tail_max_deviation = float(deviation[smoothed_times >= tail_start_time].max())
        if tail_max_deviation <= precision:
            break
    return half_window, smoothed, smoothed_times, deviation, tail_max_deviation


def analyse_response(traces, key):
    times = cumulative_response(traces[0], key)[0]
    interval_times = interval_response(traces[0], key)[0]
    cumulative_runs = np.array([cumulative_response(trace, key)[1] for trace in traces])
    interval_runs = np.array([interval_response(trace, key)[1] for trace in traces])
    steady_level = float(cumulative_runs[:, -1].mean())
    precision = RELATIVE_PRECISION * steady_level
    spread = cumulative_runs.max(axis=0) - cumulative_runs.min(axis=0)
    observed = times > 0
    criterion_time = first_time_after_which_always(times[observed], spread[observed] < precision)
    ensemble_interval = interval_runs.mean(axis=0)
    tail_start_index = int(len(ensemble_interval) * (1 - STEADY_LEVEL_TAIL_SHARE))
    tail_start_time = float(interval_times[tail_start_index])
    tail_level = float(ensemble_interval[tail_start_index:].mean())
    half_window, smoothed, smoothed_times, deviation, tail_max_deviation = smooth_until_tail_within_precision(
        ensemble_interval, interval_times, tail_start_time, tail_level, precision
    )
    warmup_time = first_time_after_which_always(smoothed_times, deviation <= precision)
    return {
        "times": times,
        "cumulative_runs": cumulative_runs,
        "spread": spread,
        "steady_level": steady_level,
        "precision": precision,
        "criterion_time": criterion_time,
        "smoothed_times": smoothed_times,
        "smoothed": smoothed,
        "tail_level": tail_level,
        "half_window": half_window,
        "tail_max_deviation": tail_max_deviation,
        "warmup_time": warmup_time,
    }


def plot_cumulative_means(analyses, warmup_time, simulation_time, caption):
    figure, axes = create_figure(caption, rows=len(RESPONSES), columns=1, width=8.5, height=10.5, sharex=True)
    for axis, (key, title, y_label) in zip(axes, RESPONSES):
        analysis = analyses[key]
        visible = analysis["times"] > 0
        for run_index, cumulative in enumerate(analysis["cumulative_runs"]):
            plot_line(axis, analysis["times"][visible], cumulative[visible], f"Прогін {run_index + 1}",
                      run_index, markers=False, linewidth=1.4)
        add_vertical_reference(axis, warmup_time, f"T_warm = {warmup_time:,.0f} мс".replace(",", " "))
        axis.axvline(simulation_time, color="#0b0b0b", linewidth=1.2, linestyle=":",
                     label=f"T_mod = {simulation_time:,.0f} мс".replace(",", " "))
        axis.set_xscale("log")
        label_axes(axis, f"{title}: середнє на інтервалі [0; t]", "", y_label)
        if key != "activations_per_second":
            use_percent_axis(axis)
    axes[-1].set_xlabel("Модельний час t, мс (логарифмічна шкала)")
    axes[1].legend(loc="lower right", ncol=2)
    figure.tight_layout()
    return save_figure(figure, "transient_cumulative.png")


def plot_spread(analyses, simulation_time, caption):
    figure, axis = create_figure(caption, width=8.5, height=4.8)
    for slot, (key, title, _) in enumerate(RESPONSES):
        analysis = analyses[key]
        visible = analysis["times"] > 0
        relative_spread = analysis["spread"][visible] / analysis["precision"]
        plot_line(axis, analysis["times"][visible], relative_spread, title, slot, markers=False, linewidth=1.6)
    add_horizontal_reference(axis, 1.0, "Межа критерію (3.1): розкид = ε")
    axis.axvline(simulation_time, color="#0b0b0b", linewidth=1.2, linestyle=":",
                 label=f"T_mod = {simulation_time:,.0f} мс".replace(",", " "))
    axis.set_xscale("log")
    axis.set_yscale("log")
    label_axes(axis, "Розкид середніх між прогонами відносно точності ε",
               "Модельний час t, мс (логарифмічна шкала)", "max |ȳ_i(t) − ȳ_j(t)| / ε")
    add_legend(axis, "lower left")
    figure.tight_layout()
    return save_figure(figure, "transient_spread.png")


def plot_warmup(analyses, warmup_time, caption):
    figure, axes = create_figure(caption, rows=len(RESPONSES), columns=1, width=8.5, height=9.5, sharex=True)
    horizon = max(WARMUP_PLOT_MIN_HORIZON_MS, WARMUP_PLOT_HORIZON_FACTOR * warmup_time)
    for slot, (axis, (key, title, y_label)) in enumerate(zip(axes, RESPONSES)):
        analysis = analyses[key]
        limit = analysis["smoothed_times"] <= horizon
        plot_line(axis, analysis["smoothed_times"][limit], analysis["smoothed"][limit],
                  f"Середнє за прогонами, ковзне (метод Велча, w = {analysis['half_window']})", slot,
                  markers=False, linewidth=1.6)
        add_horizontal_reference(axis, analysis["tail_level"], "Сталий рівень")
        axis.axhspan(analysis["tail_level"] - analysis["precision"], analysis["tail_level"] + analysis["precision"],
                     color="#8a8984", alpha=0.12, linewidth=0, label="Сталий рівень ± ε")
        axis.axvline(warmup_time, color="#0b0b0b", linewidth=1.2, linestyle=":",
                     label=f"T_warm = {warmup_time:,.0f} мс".replace(",", " "))
        label_axes(axis, f"{title} на інтервалах по {TRACE_INTERVAL_MS:.0f} мс", "", y_label)
        use_thousands_separator(axis)
        if key != "activations_per_second":
            use_percent_axis(axis, decimals=1)
        add_legend(axis, "lower right")
    axes[-1].set_xlabel("Модельний час t, мс")
    figure.tight_layout()
    return save_figure(figure, "transient_warmup.png")


def pilot_config(scenario_config):
    return scenario_config.with_overrides(warmup_time=0.0, simulation_time=HORIZON_MS, trace_interval=TRACE_INTERVAL_MS)


def analyse_scenarios():
    configs = [config for _, _, scenario_config in SCENARIOS
               for config in replicate_configs(pilot_config(scenario_config), PILOT_RUNS, SEED_OFFSET)]
    traces = [trace for _, trace in parallel_map(simulate_with_trace, configs)]
    analyses_by_scenario = {}
    for index, (scenario_key, _, _) in enumerate(SCENARIOS):
        scenario_traces = traces[index * PILOT_RUNS:(index + 1) * PILOT_RUNS]
        analyses_by_scenario[scenario_key] = {key: analyse_response(scenario_traces, key) for key, _, _ in RESPONSES}
    return analyses_by_scenario


def scenario_limits(analyses):
    criterion_time = max(analysis["criterion_time"] for analysis in analyses.values())
    detected_warmup = max(analysis["warmup_time"] for analysis in analyses.values())
    return criterion_time, detected_warmup


def main():
    started = time.time()
    analyses_by_scenario = analyse_scenarios()
    analyses = analyses_by_scenario["base"]
    limits = {scenario_key: scenario_limits(analyses_by_scenario[scenario_key]) for scenario_key, _, _ in SCENARIOS}
    criterion_time, detected_warmup = limits["base"]
    if math.isinf(criterion_time) or math.isinf(detected_warmup):
        raise RuntimeError("Базовий сценарій не досяг сталого режиму на пробному горизонті: "
                           "збільште горизонт пробних прогонів або оберіть інший відгук (посібник, с. 80)")
    warmup_time = round_up(max(detected_warmup, TRACE_INTERVAL_MS) * WARMUP_SAFETY_FACTOR, WARMUP_ROUNDING_MS)
    simulation_time = round_up(criterion_time, SIMULATION_TIME_ROUNDING_MS)

    rows = []
    for scenario_key, scenario_title, _ in SCENARIOS:
        for key, title, _ in RESPONSES:
            analysis = analyses_by_scenario[scenario_key][key]
            rows.append([scenario_title] + response_row(title, analysis)
                        + ["так" if is_sufficient(analysis, warmup_time, simulation_time) else "ні"])
    write_csv(
        "transient.csv",
        ["Сценарій", "Відгук", "Сталий рівень", "Точність ε (5% рівня)", "T за критерієм (3.1), мс",
         f"Півширина вікна Велча w, інтервалів по {TRACE_INTERVAL_MS:.0f} мс",
         "Макс. відхилення згладженого ряду на другій половині горизонту",
         "Кінець перехідного періоду (відхилення ≤ ε), мс", f"Розкид при t = {HORIZON_MS:.0f} мс",
         "Обраних T_warm і T_mod достатньо"],
        rows,
    )

    caption = simulation_caption(
        HORIZON_MS, replications=PILOT_RUNS,
        extra=f"пробні прогони без T_warm; базовий сценарій B8; ε = {RELATIVE_PRECISION:.0%} сталого рівня",
    )
    plot_cumulative_means(analyses, warmup_time, simulation_time, caption)
    plot_spread(analyses, simulation_time, caption)
    plot_warmup(analyses, warmup_time, caption)

    save_run_params({
        "warmup_time_ms": warmup_time,
        "simulation_time_ms": simulation_time,
        "transient": {
            "pilot_runs": PILOT_RUNS,
            "horizon_ms": HORIZON_MS,
            "trace_interval_ms": TRACE_INTERVAL_MS,
            "relative_precision": RELATIVE_PRECISION,
            "detected_warmup_ms": detected_warmup,
            "warmup_safety_factor": WARMUP_SAFETY_FACTOR,
            "criterion_time_ms": criterion_time,
            "scenarios": {
                scenario_key: {
                    "title": scenario_title,
                    "criterion_time_ms": limits[scenario_key][0],
                    "detected_warmup_ms": limits[scenario_key][1],
                    "sufficient": {
                        key: is_sufficient(analyses_by_scenario[scenario_key][key], warmup_time, simulation_time)
                        for key, _, _ in RESPONSES
                    },
                    "responses": {
                        key: {
                            "steady_level": analyses_by_scenario[scenario_key][key]["steady_level"],
                            "precision": analyses_by_scenario[scenario_key][key]["precision"],
                            "criterion_time_ms": analyses_by_scenario[scenario_key][key]["criterion_time"],
                            "welch_half_window": analyses_by_scenario[scenario_key][key]["half_window"],
                            "tail_max_deviation": analyses_by_scenario[scenario_key][key]["tail_max_deviation"],
                            "warmup_time_ms": analyses_by_scenario[scenario_key][key]["warmup_time"],
                        }
                        for key, _, _ in RESPONSES
                    },
                }
                for scenario_key, scenario_title, _ in SCENARIOS
            },
        },
    })

    print(f"Пробні прогони: {PILOT_RUNS} × {HORIZON_MS:.0f} мс на сценарій, крок трасування {TRACE_INTERVAL_MS:.0f} мс")
    for row in rows:
        print("  " + " | ".join(str(value) for value in row))
    for scenario_key, scenario_title, _ in SCENARIOS:
        print(f"{scenario_title}: T за (3.1) = {limits[scenario_key][0]:.0f} мс, "
              f"перехідний період = {limits[scenario_key][1]:.0f} мс")
    print(f"Перехідний період (виявлено): {detected_warmup:.0f} мс → T_warm = {warmup_time:.0f} мс")
    print(f"Критерій (3.1): T = {criterion_time:.0f} мс → T_mod = {simulation_time:.0f} мс")
    print(f"Час виконання: {time.time() - started:.1f} с")


def is_sufficient(analysis, warmup_time, simulation_time):
    return analysis["warmup_time"] <= warmup_time and analysis["criterion_time"] <= simulation_time


def response_row(title, analysis):
    return [
        title,
        f"{analysis['steady_level']:.5f}",
        f"{analysis['precision']:.5f}",
        f"{analysis['criterion_time']:.0f}",
        analysis["half_window"],
        f"{analysis['tail_max_deviation']:.5f}",
        f"{analysis['warmup_time']:.0f}",
        f"{analysis['spread'][-1]:.5f}",
    ]


if __name__ == "__main__":
    main()
