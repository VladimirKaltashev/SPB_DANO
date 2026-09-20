"""Standalone presentation figures for all three links of the mechanism."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .prepare_data import GROUPS, ZERO

COLORS = {"LOW": "#397CB4", "MEDIUM": "#DB9A36", "HIGH": "#A74666", ZERO: "#8A969F"}
MONTHS = {"2026-04": "Апр", "2026-05": "Май", "2026-06": "Июн", "2026-07": "Июл", "2026-08": "Авг"}


def p_text(p):
    return f"{p:.2e}" if p < .001 else f"{p:.4f}"


def save(fig, folder, name):
    for ext in ["png", "svg"]:
        fig.savefig(folder / f"{name}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def draw_figures(national, baseline, slopes, interactions, baseline_results, composition, folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    for outcome, name, title, ylabel in [
        ("liters_per_driver", "01_usage_groups_liters_over_time", "Покупки топлива по фиксированным апрельским группам", "Литров на водителя в месяц"),
        ("liters_index", "02_usage_groups_normalized_activity", "Покупки топлива относительно апрельского уровня", "Индекс наблюдаемых покупок: апрель = 100"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for group in GROUPS:
            part = national.loc[national.usage_group.eq(group)].sort_values("month")
            ax.plot([MONTHS[m] for m in part.month], part[outcome], marker="o", color=COLORS[group], lw=2.5, label=group)
            for i in [0, len(part) - 1]:
                row = part.iloc[i]
                offset = 13 if group != "MEDIUM" else -19
                if outcome == "liters_index":
                    offset = -15 if group == "HIGH" else 10
                if outcome == "liters_index" and i == 0:
                    continue
                label_offset, alignment = (0, offset), "center"
                if outcome == "liters_per_driver" and i == len(part) - 1:
                    label_offset = (15, {"LOW": 0, "MEDIUM": -8, "HIGH": 8}[group])
                    alignment = "left"
                ax.annotate(f"{row[outcome]:.1f}", (MONTHS[row.month], row[outcome]), xytext=label_offset,
                            textcoords="offset points", ha=alignment, fontsize=10, color=COLORS[group])
        if outcome == "liters_index":
            ax.axhline(100, color="#666", ls="--", lw=1)
        ax.set(title=title, ylabel=ylabel, xlabel="2026 год")
        ax.legend(frameon=False, ncols=3)
        ax.grid(alpha=.18)
        ax.margins(y=.2)
        if outcome == "liters_per_driver":
            ax.margins(x=.09)
        fig.text(.12, .02, "Группа определена только по апрелю. В знаменателе остаются все её клиенты, включая месяцы с нулём покупок.", fontsize=9)
        fig.tight_layout(rect=[0, .055, 1, 1])
        save(fig, folder, name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, outcome, title in zip(axes, ["liters_per_driver", "liters_index"],
                                  ["Абсолютное изменение, л/водитель", "Относительное изменение, % апрельского объёма"]):
        part = slopes.loc[slopes.outcome.eq(outcome) & slopes.term.eq("price_shock")].set_index("usage_group").loc[GROUPS]
        for i, group in enumerate(GROUPS):
            row = part.loc[group]
            effect = .1 * row.beta
            ax.errorbar(effect, i, xerr=[[.1 * (row.beta - row.ci95_low)], [.1 * (row.ci95_high - row.beta)]],
                        fmt="o", color=COLORS[group], capsize=5, markersize=8)
            ax.annotate(f"{effect:+.1f}; p={p_text(row.p_value)}", (effect, i),
                        xytext=(0, 14), textcoords="offset points", ha="center", fontsize=10)
        ax.axvline(0, color="#666", lw=1)
        ax.set_yticks(range(3), GROUPS)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Изменение при +10 п.п. PriceShock; 95% ДИ")
        ax.set_ylim(2.6, -.6)
        ax.grid(axis="x", alpha=.18)
    contrast = interactions.loc[interactions.period.eq("all_months") & interactions.outcome.eq("liters_index")
                                & interactions.term.eq("PriceShock_x_HIGH")].iloc[0]
    fig.suptitle("Связь цены с покупками: абсолютная и относительная реакция", fontsize=17)
    fig.text(.5, .025, f"Прямой тест относительной разницы HIGH − LOW: p={p_text(contrast.p_value)}. "
             "Отдельные p-value групп не проверяют их различие.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .08, 1, .93])
    save(fig, folder, "03_price_effect_by_usage_group")

    fig, ax = plt.subplots(figsize=(9.5, 6))
    base = baseline.set_index("usage_group").loc[GROUPS]
    bars = ax.bar(GROUPS, base.baseline_fines_per_driver, color=[COLORS[g] for g in GROUPS], width=.58)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in base.baseline_fines_per_driver], padding=7, fontsize=14)
    row = baseline_results.loc[baseline_results.term.eq("HIGH")].iloc[0]
    ax.set(title="Штрафы на водителя до кризиса: апрель 2026", ylabel="Штрафов на водителя за апрель")
    ax.set_ylim(0, base.baseline_fines_per_driver.max() * 1.28)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.5, .035, f"HIGH − LOW = {row.beta:.3f} штрафа; p={p_text(row.p_value)}. Ошибки сгруппированы по регионам.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .08, 1, 1])
    save(fig, folder, "04_baseline_usage_vs_fines")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, denominator, groups, title in [
        (axes[0], "positive_baseline_tertiles", GROUPS, "Среди покупателей из фиксированных терцилей"),
        (axes[1], "eligible_including_zero", GROUPS + [ZERO], "С учётом клиентов ZERO baseline"),
    ]:
        part = composition.loc[composition.denominator_population.eq(denominator) & composition.active_threshold_liters.eq(0)].sort_values("month")
        for group in groups:
            ax.plot([MONTHS[m] for m in part.month], part["share_" + group], color=COLORS[group],
                    marker="o", lw=2, label=group.replace("_", " "))
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Доля среди наблюдаемых покупателей")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(0, .40)
        ax.grid(alpha=.18)
        ax.legend(fontsize=9, loc="lower left" if denominator == "positive_baseline_tertiles" else "best", frameon=False)
    fig.suptitle("Меняется ли состав покупателей в сторону HIGH?", fontsize=17)
    fig.text(.5, .025, "Active = положительные физические литры в месяце. Слева LOW + MEDIUM + HIGH = 100%; справа добавлена ZERO.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .065, 1, .93])
    save(fig, folder, "05_active_user_composition")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, outcome, ylabel in zip(axes, ["fines_per_driver", "total_fines"], ["Штрафов на водителя", "Всего штрафов"]):
        for group in GROUPS:
            part = national.loc[national.usage_group.eq(group)].sort_values("month")
            ax.plot([MONTHS[m] for m in part.month], part[outcome], color=COLORS[group], marker="o", label=group, lw=2)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(alpha=.18)
    fig.suptitle("Динамика штрафов по тем же фиксированным группам", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, .92])
    save(fig, folder, "06_usage_groups_fines_over_time")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (period, label) in enumerate([("all_months", "Апрель–август: основной"), ("post_baseline", "Май–август: robustness")]):
        row = interactions.loc[interactions.period.eq(period) & interactions.outcome.eq("liters_index") & interactions.term.eq("PriceShock_x_HIGH")].iloc[0]
        ax.errorbar(.1 * row.beta, i, xerr=[[.1 * (row.beta - row.ci95_low)], [.1 * (row.ci95_high - row.beta)]],
                    fmt="o", capsize=6, color="#395975", markersize=9)
        ax.annotate(f"{row.beta * .1:+.1f}; p={p_text(row.p_value)}", (.1 * row.beta, i),
                    xytext=(0, 14), textcoords="offset points", ha="center", fontsize=11)
    ax.set_yticks([0, 1], ["Апрель–август: основной", "Май–август: robustness"])
    ax.set_ylim(1.55, -.55)
    ax.axvline(0, color="#666", lw=1)
    ax.set(title="H1 чувствителен к включению месяца формирования групп",
           xlabel="HIGH − LOW: разница относительных наклонов при +10 п.п. PriceShock; 95% ДИ")
    ax.grid(axis="x", alpha=.18)
    fig.text(.5, .025, "Положительное значение соответствует меньшему сокращению HIGH относительно LOW. Группы и их границы не менялись.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .08, 1, 1])
    save(fig, folder, "07_interaction_period_sensitivity")
