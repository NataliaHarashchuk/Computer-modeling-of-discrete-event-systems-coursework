import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from experiments.common import results_path

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID_COLOR = "#e4e3df"
AXIS_COLOR = "#b5b4ae"
SURFACE_COLOR = "#ffffff"
REFERENCE_COLOR = "#6f6e69"
INTERVAL_ALPHA = 0.16
SAVE_DPI = 200


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": SURFACE_COLOR,
        "axes.facecolor": SURFACE_COLOR,
        "axes.edgecolor": AXIS_COLOR,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.7,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TEXT_PRIMARY,
        "savefig.dpi": SAVE_DPI,
        "savefig.facecolor": SURFACE_COLOR,
        "savefig.bbox": "tight",
    })


def series_color(slot):
    if not 0 <= slot < len(SERIES_COLORS):
        raise ValueError(f"series slot must be within 0..{len(SERIES_COLORS) - 1}")
    return SERIES_COLORS[slot]


def format_milliseconds(value):
    return f"{value:,.0f}".replace(",", " ")


def simulation_caption(simulation_time, warmup_time=None, replications=None, extra=None):
    parts = [f"Час моделювання: {format_milliseconds(simulation_time)} мс"]
    if warmup_time is not None:
        parts.append(f"T_warm = {format_milliseconds(warmup_time)} мс")
    if replications is not None:
        parts.append(f"прогонів: {replications}")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def create_figure(caption, rows=1, columns=1, width=8.0, height=4.6, sharex=False):
    apply_style()
    figure, axes = plt.subplots(rows, columns, figsize=(width, height), sharex=sharex, squeeze=False)
    figure.text(0.01, -0.01, caption, ha="left", va="top", fontsize=8.5, color=TEXT_SECONDARY)
    if rows == 1 and columns == 1:
        return figure, axes[0][0]
    return figure, axes.flatten()


def label_axes(axis, title, x_label, y_label):
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


def use_thousands_separator(axis, which="x"):
    formatter = FuncFormatter(lambda value, _: format_milliseconds(value))
    target = axis.xaxis if which == "x" else axis.yaxis
    target.set_major_formatter(formatter)


def use_percent_axis(axis, which="y", decimals=0):
    formatter = FuncFormatter(lambda value, _: f"{value * 100:.{decimals}f}%")
    target = axis.yaxis if which == "y" else axis.xaxis
    target.set_major_formatter(formatter)


def plot_line(axis, x_values, y_values, label, slot, markers=True, linewidth=2.0):
    return axis.plot(
        x_values, y_values,
        label=label,
        color=series_color(slot),
        linewidth=linewidth,
        marker="o" if markers else None,
        markeredgecolor=SURFACE_COLOR,
        markeredgewidth=1.2,
    )[0]


def plot_line_with_interval(axis, x_values, means, half_widths, label, slot, markers=True):
    line = plot_line(axis, x_values, means, label, slot, markers)
    lower = [mean - half for mean, half in zip(means, half_widths)]
    upper = [mean + half for mean, half in zip(means, half_widths)]
    axis.fill_between(x_values, lower, upper, color=series_color(slot), alpha=INTERVAL_ALPHA, linewidth=0)
    return line


def plot_bars(axis, labels, values, half_widths=None, slot=0, value_format="{:.3f}"):
    positions = range(len(labels))
    bars = axis.bar(
        positions, values,
        width=0.6,
        color=series_color(slot),
        edgecolor=SURFACE_COLOR,
        linewidth=2,
        yerr=half_widths,
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.2, "capsize": 4},
    )
    axis.set_xticks(list(positions), labels)
    axis.grid(axis="x", visible=False)
    for bar, value in zip(bars, values):
        axis.annotate(
            value_format.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color=TEXT_PRIMARY,
        )
    return bars


def add_horizontal_reference(axis, value, label):
    axis.axhline(value, color=REFERENCE_COLOR, linewidth=1.2, linestyle="--", label=label)


def add_vertical_reference(axis, value, label):
    axis.axvline(value, color=REFERENCE_COLOR, linewidth=1.2, linestyle="--", label=label)


def add_legend(axis, location="best"):
    axis.legend(loc=location)


def add_framed_legend(axis, location="best", fontsize=None):
    axis.legend(loc=location, fontsize=fontsize, frameon=True, facecolor=SURFACE_COLOR,
                edgecolor=GRID_COLOR, framealpha=0.95)


def save_figure(figure, filename):
    path = results_path(filename)
    figure.savefig(path)
    plt.close(figure)
    return path
