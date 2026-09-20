"""Presentation-sized scatter plots and price-selected regional dynamics."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

LABELS = {
    "liters_per_driver": "Литров на водителя",
    "fines_per_driver": "Штрафов на водителя",
    "fines_per_1000_liters": "Штрафов на 1000 наблюдаемых литров",
    "price_shock": "Рост цены относительно апреля",
}
TITLES = {
    "liters_per_driver": "Цена и наблюдаемая топливная активность",
    "fines_per_driver": "Цена и количество штрафов на водителя",
    "fines_per_1000_liters": "Цена и интенсивность штрафов на 1000 литров",
}
FILENAMES = {
    "liters_per_driver": "01_price_vs_liters_per_driver",
    "fines_per_driver": "02_price_vs_fines_per_driver",
    "fines_per_1000_liters": "03_price_vs_fines_per_1000_liters",
}
MONTH_NAMES = {"2026-04": "Апр", "2026-05": "Май", "2026-06": "Июн", "2026-07": "Июл", "2026-08": "Авг"}
MONTH_COLORS = ["#4477AA", "#EEAA44", "#228833", "#CC6677", "#8855AA"]


def p_text(value):
    return f"{value:.2e}" if value < .001 else f"{value:.4f}"


def save_figure(fig, figures: Path, name: str):
    for suffix in ["png", "svg"]:
        fig.savefig(figures / f"{name}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def draw_figures(panel, regressions, selected, distribution, national, figures: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    main = regressions.loc[regressions.block.eq("main")].set_index("outcome")
    for outcome, filename in FILENAMES.items():
        row = main.loc[outcome]
        fig, ax = plt.subplots(figsize=(11.5, 7))
        for (month, part), color in zip(panel.groupby("month", sort=True), MONTH_COLORS):
            ax.scatter(part.price_shock, part[outcome], s=56, label=MONTH_NAMES.get(month, month),
                       color=color, alpha=.82, edgecolors="white", linewidths=.5)
        x = np.linspace(panel.price_shock.min(), panel.price_shock.max(), 150)
        ax.plot(x, row.intercept + row.BETA * x, color="#1B344F", linewidth=2.5, label="OLS")
        point = panel.loc[panel.price_shock.idxmax()]
        ax.annotate(point.region_name + ", " + MONTH_NAMES.get(point.month, point.month),
                    (point.price_shock, point[outcome]), xytext=(-12, 12), ha="right",
                    textcoords="offset points", fontsize=9)
        ax.set_title(TITLES[outcome] + "\n" +
                     f"β = {row.BETA:.3f}   SE = {row.SE:.3f}   p = {p_text(row.p_value)}   "
                     f"R² = {row.R_squared:.3f}   N = {int(row.N)}", fontsize=14, pad=15)
        ax.set_xlabel("PriceShock: рост цены относительно апреля, %")
        ax.set_ylabel(LABELS[outcome])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=.18)
        ax.margins(x=.05, y=.12)
        ax.legend(ncols=6, loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=10)
        fig.text(.12, .025, "Одна точка — регион × месяц. Ошибки сгруппированы по 16 регионам. Литры — proxy активности, не пробег.", fontsize=9)
        fig.tight_layout(rect=[0, .10, 1, 1])
        save_figure(fig, figures, filename)

    # All regions are shown; none selected for favorable outcome behavior.
    rank = panel.loc[panel.month.eq(panel.month.max())].sort_values(["price_shock", "region"])
    for outcome in ["price_shock", *FILENAMES]:
        fig, axes = plt.subplots(4, 4, figsize=(18, 13), sharex=True, sharey=True)
        for ax, region in zip(axes.flat, rank.region):
            part = panel.loc[panel.region.eq(region)].sort_values("month")
            ax.plot([MONTH_NAMES.get(month, month) for month in part.month], part[outcome],
                    marker="o", markersize=4, linewidth=1.8, color="#28638B")
            ax.set_title(part.region_name.iloc[0], fontsize=10, pad=7)
            ax.grid(alpha=.18)
            if outcome == "price_shock":
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        fig.suptitle(LABELS[outcome] + " по регионам: апрель–август 2026", fontsize=18)
        fig.text(.5, .015, "Регионы упорядочены по росту цены к августу — от слабого к сильному. Масштаб Y общий для всех регионов.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=[0, .04, 1, .96])
        save_figure(fig, figures, f"regional_{outcome}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#4477AA", "#DD9944", "#AA4466"]
    group_names = {"weak": "слабый рост", "middle": "средний рост", "strong": "сильный рост"}
    for ax, outcome in zip(axes.flat, ["price_shock", *FILENAMES]):
        for region, color in zip(selected.itertuples(), colors):
            part = panel.loc[panel.region.eq(region.region)].sort_values("month")
            ax.plot([MONTH_NAMES.get(m, m) for m in part.month], part[outcome], marker="o", color=color,
                    label=f"{region.region_name} ({group_names[region.price_growth_group]})", linewidth=2)
        ax.set_title(LABELS[outcome], fontsize=12)
        ax.grid(alpha=.18)
        if outcome == "price_shock":
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=1, frameon=False, fontsize=10)
    fig.suptitle("Три региона, выбранные только по росту цены к августу", fontsize=17)
    fig.tight_layout(rect=[0, .13, 1, .95])
    save_figure(fig, figures, "04_selected_regional_dynamics")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, outcome in zip(axes, FILENAMES):
        ax.plot([MONTH_NAMES.get(m, m) for m in national.month], national[outcome], marker="o", color="#28638B", lw=2)
        ax.set_title(LABELS[outcome], fontsize=11)
        ax.grid(alpha=.18)
    fig.suptitle("Общая месячная динамика: показатели из суммарных штрафов и литров", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .90])
    save_figure(fig, figures, "05_national_monthly_dynamics")

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["min", "p1", "p5", "p10", "median", "p90", "p95", "p99", "max"]
    for row, color in zip(distribution.itertuples(index=False), ["#4477AA", "#CC6677"]):
        values = distribution.loc[distribution.population.eq(row.population), labels].iloc[0]
        ax.plot(labels, values, marker="o", label="Все клиент-месяцы" if row.population == "all_driver_months" else "Только положительные литры", color=color)
    ax.set_yscale("symlog", linthresh=10)
    for threshold, color in zip([10, 20, 40], ["#888", "#111", "#AAA"]):
        ax.axhline(threshold, linestyle="--", linewidth=.8, color=color)
        ax.text(8.2, threshold, f"{threshold} л", fontsize=9, va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set(ylabel="Литры: шкала symlog, линейная до 10 л", title="Распределение литров на клиент-месяц")
    ax.legend(fontsize=9)
    ax.grid(alpha=.18)
    fig.tight_layout()
    save_figure(fig, figures, "06_liters_distribution")
