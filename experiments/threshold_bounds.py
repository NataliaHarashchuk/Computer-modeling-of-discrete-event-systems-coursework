import time

from experiments.common import load_run_params, write_csv
from experiments.proposals import COMPARISON_HEADER, comparison_row, run_variant

TABLE_FILE = "threshold_bounds.csv"

BOUND_VARIANTS = (
    ("Основний: вмикання ≥ 30%, вимикання < 30%",
     dict(activate_at_threshold=True, deactivate_at_threshold=False)),
    ("Строге вмикання: > 30%, вимикання < 30%",
     dict(activate_at_threshold=False, deactivate_at_threshold=False)),
    ("Повністю строге: вмикання > 30%, вимикання ≤ 30%",
     dict(activate_at_threshold=False, deactivate_at_threshold=True)),
    ("Обидві межі нестрогі: вмикання ≥ 30%, вимикання ≤ 30%",
     dict(activate_at_threshold=True, deactivate_at_threshold=True)),
)
BASE_CONTROL = dict(activation_threshold=0.30, deactivation_threshold=0.30, window_size=100)


def main():
    started = time.perf_counter()
    params = load_run_params()
    samples_list = [run_variant(params, {**BASE_CONTROL, **overrides}) for _, overrides in BOUND_VARIANTS]
    baseline = samples_list[0]
    rows = [comparison_row(label, samples, baseline) for (label, _), samples in zip(BOUND_VARIANTS, samples_list)]
    write_csv(TABLE_FILE, COMPARISON_HEADER[:-1], rows)
    print(f"Межі порогу керування: T_warm = {params['warmup_time_ms']:.0f} мс, "
          f"T_mod = {params['simulation_time_ms']:.0f} мс, {params['replications']} прогонів, спільні випадкові числа")
    for row in rows:
        print(f"  {row[0]}: P_зн {row[1]} ± {row[2]}%, підключень {row[4]} ± {row[5]} 1/с, "
              f"R=1 {row[6]} ± {row[7]}% часу, рівень >30% {row[9]} ± {row[10]}% часу, p(P_зн) {row[16]}")
    print(f"Час виконання: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
