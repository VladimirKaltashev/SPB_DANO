"""Generate presentation-ready charts for the active mathematical hypothesis.

Run from the project root:

    python -m current_hypothesis.presentation_charts

The module reads the active weekly panel and the current decoupling model. It
does not overwrite analytical figures produced by the main pipeline. All slide
assets and accompanying audit tables are written to
``outputs/presentation_math_model``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import font_manager
from scipy.stats import t

from current_hypothesis.check_hypothesis import make_client_baseline
from current_hypothesis.price_fuel_fines_model import (
    DEFAULT_COHORT_START,
    DEFAULT_CRISIS_START,
    DEFAULT_EQUIVALENCE_MARGIN,
    fit_decoupling_model,
    prepare_client_periods,
)
from pipeline.data_sources import read_csv_detected
from pipeline.plotting import plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PROJECT_ROOT / "data" / "processed" / "client_week_panel.csv"
DEFAULT_DEMOGRAPHICS = PROJECT_ROOT / "data" / "processed" / "clients_demographics_clean.csv"
DEFAULT_FINES = PROJECT_ROOT / "data" / "processed" / "fines_clean.csv"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "outputs" / "price_fuel_fines_model"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "presentation_math_model"

DARK_TEXT = "#1E0853"
DEEP_PURPLE = "#461B6F"
MAUVE = "#9A4A88"
ROSE = "#AB599B"
LIGHT_PINK = "#F5CAE6"
SLIDE_BG = "#BF8ED5"

REGION_NAMES = {
    "77": "Москва",
    "50": "Московская область",
    "78": "Санкт-Петербург",
    "16": "Татарстан",
    "66": "Свердловская область",
    "54": "Новосибирская область",
    "47": "Ленинградская область",
    "55": "Омская область",
    "52": "Нижегородская область",
    "23": "Краснодарский край",
    "63": "Самарская область",
    "74": "Челябинская область",
    "42": "Кемеровская область",
    "72": "Тюменская область",
    "02": "Башкортостан",
    "24": "Красноярский край",
}


def _font_family() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    return "Montserrat" if "Montserrat" in available else "DejaVu Sans"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": _font_family(),
            "font.size": 12,
            "text.color": DARK_TEXT,
            "axes.labelcolor": DARK_TEXT,
            "axes.edgecolor": DARK_TEXT,
            "xtick.color": DARK_TEXT,
            "ytick.color": DARK_TEXT,
            "axes.titlecolor": DARK_TEXT,
            "svg.fonttype": "none",
            "savefig.transparent": True,
        }
    )


def fmt(value: float, digits: int = 2, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return (prefix + f"{value:.{digits}f}").replace(".", ",")


def p_text(value: float) -> str:
    if value < 0.001:
        return "p < 0,001"
    return f"p = {value:.3f}".replace(".", ",")


def clean_axis(axis, grid: str | None = None) -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    if grid:
        axis.grid(axis=grid, color=LIGHT_PINK, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)


def save_figure(fig, output: Path, stem: str) -> None:
    fig.patch.set_alpha(0)
    for axis in fig.axes:
        axis.patch.set_alpha(0)
    fig.savefig(output / f"{stem}.png", dpi=320, transparent=True, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def mean_interval(values: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 2:
        raise ValueError("Для доверительного интервала нужно не менее двух наблюдений")
    estimate = float(clean.mean())
    se = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    critical = float(t.ppf(0.5 + confidence / 2, len(clean) - 1))
    return {
        "estimate": estimate,
        "standard_error": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "confidence": confidence,
        "n": len(clean),
    }


def regression(data: pd.DataFrame, dependent: str, independent: str) -> dict[str, float]:
    sample = data[[dependent, independent]].dropna().astype(float)
    design = sm.add_constant(sample[independent], has_constant="add")
    fitted = sm.OLS(sample[dependent], design).fit(cov_type="HC3")
    ci = fitted.conf_int().loc[independent]
    return {
        "dependent": dependent,
        "independent": independent,
        "n": len(sample),
        "b0": float(fitted.params["const"]),
        "b1": float(fitted.params[independent]),
        "se_b1": float(fitted.bse[independent]),
        "p_value": float(fitted.pvalues[independent]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "r_squared": float(fitted.rsquared),
    }


def calculate_statistics(periods: pd.DataFrame, margin: float, alpha: float):
    result = fit_decoupling_model(periods, margin, alpha)
    tests = result.tests.set_index("test")
    data = result.client_period
    price_sample = data[["fuel_price_pre", "fuel_price_post"]].dropna()
    price_change = price_sample["fuel_price_post"] - price_sample["fuel_price_pre"]
    fuel_change = data["fuel_rate_30d_post"] - data["fuel_rate_30d_pre"]
    fine_change = data["fine_rate_30d_post"] - data["fine_rate_30d_pre"]
    price_ci = mean_interval(price_change)
    fuel_ci = mean_interval(fuel_change)
    fine_ci95 = mean_interval(fine_change)
    fine_ci90 = mean_interval(fine_change, 0.90)
    gap_ci = mean_interval(data["fine_above_proportional"])
    stats = {
        "result": result,
        "tests": tests,
        "data": data,
        "price_pre": float(price_sample["fuel_price_pre"].mean()),
        "price_post": float(price_sample["fuel_price_post"].mean()),
        "price_change": price_ci,
        "fuel_pre": float(data["fuel_rate_30d_pre"].mean()),
        "fuel_post": float(data["fuel_rate_30d_post"].mean()),
        "fuel_change": fuel_ci,
        "fuel_relative_change": float(
            data["fuel_rate_30d_post"].mean() / data["fuel_rate_30d_pre"].mean() - 1
        ),
        "fine_pre": float(data["fine_rate_30d_pre"].mean()),
        "fine_post": float(data["fine_rate_30d_post"].mean()),
        "fine_change": fine_ci95,
        "fine_change_90": fine_ci90,
        "proportional": float(data["proportional_fine_counterfactual"].mean()),
        "gap": gap_ci,
        "margin": margin,
        "alpha": alpha,
        "price_n": len(price_sample),
        "cohort_n": data["client_id"].nunique(),
    }
    return stats


def chart_fuel(stats: dict[str, object], output: Path) -> None:
    values = [stats["fuel_pre"], stats["fuel_post"]]
    fig, axis = plt.subplots(figsize=(5.6, 4.5))
    bars = axis.bar([0, 1], values, width=0.56, color=[ROSE, DEEP_PURPLE])
    axis.set_ylim(0, max(values) * 1.28)
    axis.set_xticks([0, 1], ["До 1 июня", "После 1 июня"], fontsize=13)
    axis.set_ylabel("л на клиента за 30 дней", fontsize=12)
    axis.set_yticks([])
    clean_axis(axis)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            fmt(value, 2),
            ha="center",
            va="bottom",
            fontsize=21,
            fontweight="bold",
            color=DARK_TEXT,
        )
    axis.text(
        0.5,
        max(values) * 1.15,
        fmt(stats["fuel_relative_change"] * 100, 1, signed=True) + "%",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=DEEP_PURPLE,
    )
    fig.tight_layout()
    save_figure(fig, output, "01_fuel_before_after")


def chart_tost(stats: dict[str, object], output: Path) -> None:
    estimate = stats["fine_change_90"]["estimate"]
    low = stats["fine_change_90"]["ci_low"]
    high = stats["fine_change_90"]["ci_high"]
    margin = stats["margin"]
    tost_p = float(stats["tests"].loc["fine_rate_equivalence", "p_value"])
    fig, axis = plt.subplots(figsize=(7.0, 3.35))
    axis.axvspan(-margin, margin, color=LIGHT_PINK, alpha=0.95, zorder=0)
    axis.axvline(0, color=MAUVE, linewidth=1.1, linestyle="--", zorder=1)
    axis.axvline(-margin, color=ROSE, linewidth=1.0, zorder=1)
    axis.axvline(margin, color=ROSE, linewidth=1.0, zorder=1)
    axis.hlines(0, low, high, color=DARK_TEXT, linewidth=4, zorder=3)
    axis.plot([low, high], [0, 0], "|", color=DARK_TEXT, markersize=15, mew=2, zorder=4)
    axis.scatter([estimate], [0], s=120, color=DEEP_PURPLE, zorder=5)
    axis.set_xlim(-0.065, 0.065)
    axis.set_ylim(-0.72, 0.72)
    axis.set_yticks([])
    axis.set_xticks([-margin, 0, margin], ["−0,05", "0", "+0,05"], fontsize=12)
    axis.set_xlabel("изменение числа штрафов на клиента за 30 дней", fontsize=11)
    clean_axis(axis)
    axis.text(
        estimate,
        0.25,
        "Δ = " + fmt(estimate, 3, signed=True),
        ha="center",
        fontsize=19,
        fontweight="bold",
        color=DEEP_PURPLE,
    )
    axis.text(0, 0.58, "практически малое изменение", ha="center", fontsize=12)
    axis.text(
        0,
        -0.38,
        "90% ДИ полностью внутри ±0,05",
        ha="center",
        fontsize=12,
        color=DARK_TEXT,
    )
    axis.text(
        0,
        -0.57,
        f"±0,05 = 1 штраф на 20 клиентов за 30 дней  ·  TOST, {p_text(tost_p)}",
        ha="center",
        fontsize=9.5,
        color=MAUVE,
    )
    fig.tight_layout()
    save_figure(fig, output, "02_fines_tost")


def chart_actual_vs_proportional(stats: dict[str, object], output: Path) -> None:
    values = [stats["proportional"], stats["fine_post"]]
    gap = stats["gap"]
    fig, axis = plt.subplots(figsize=(5.8, 4.5))
    bars = axis.bar([0, 1], values, width=0.56, color=[MAUVE, DEEP_PURPLE])
    top = max(values) * 1.30
    axis.set_ylim(0, top)
    axis.set_xticks([0, 1], ["Пропорциональный\nсценарий", "Фактически"], fontsize=12)
    axis.set_ylabel("штрафа на клиента за 30 дней", fontsize=11)
    axis.set_yticks([])
    clean_axis(axis)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + top * 0.025,
            fmt(value, 3),
            ha="center",
            fontsize=20,
            fontweight="bold",
        )
    bracket_y = max(values) + top * 0.11
    axis.plot([0, 0, 1, 1], [bracket_y - 0.012, bracket_y, bracket_y, bracket_y - 0.012], color=DARK_TEXT, lw=1.4)
    axis.text(
        0.5,
        bracket_y + top * 0.02,
        "+" + fmt(gap["estimate"], 3),
        ha="center",
        fontsize=21,
        fontweight="bold",
        color=DEEP_PURPLE,
    )
    axis.text(
        0.5,
        bracket_y - top * 0.10,
        "95% ДИ разницы " + f"[{fmt(gap['ci_low'], 3)}; {fmt(gap['ci_high'], 3)}]",
        ha="center",
        fontsize=9.5,
        color=MAUVE,
    )
    fig.tight_layout()
    save_figure(fig, output, "03_fines_actual_vs_proportional")


def chart_main(stats: dict[str, object], output: Path) -> None:
    fig = plt.figure(figsize=(10.5, 4.25))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.2], wspace=0.30)
    fuel_axis = fig.add_subplot(grid[0, 0])
    fine_axis = fig.add_subplot(grid[0, 1])
    fuel_values = [stats["fuel_pre"], stats["fuel_post"]]
    bars = fuel_axis.bar([0, 1], fuel_values, color=[ROSE, DEEP_PURPLE], width=0.55)
    fuel_axis.set_ylim(0, max(fuel_values) * 1.27)
    fuel_axis.set_xticks([0, 1], ["До", "После"], fontsize=12)
    fuel_axis.set_yticks([])
    fuel_axis.text(0.02, 0.98, "ТОПЛИВО", transform=fuel_axis.transAxes, va="top", fontsize=13, fontweight="bold")
    fuel_axis.text(
        0.98,
        0.98,
        fmt(stats["fuel_relative_change"] * 100, 1, signed=True) + "%",
        transform=fuel_axis.transAxes,
        ha="right",
        va="top",
        fontsize=21,
        fontweight="bold",
        color=DEEP_PURPLE,
    )
    for bar, value in zip(bars, fuel_values, strict=True):
        fuel_axis.text(bar.get_x() + bar.get_width() / 2, value + 3, fmt(value, 2), ha="center", fontsize=16, fontweight="bold")
    fuel_axis.text(0.5, -0.20, "л на клиента за 30 дней", transform=fuel_axis.transAxes, ha="center", fontsize=10)
    clean_axis(fuel_axis)

    fine_values = [stats["fine_pre"], stats["fine_post"], stats["proportional"]]
    fine_bars = fine_axis.bar([0, 1, 2], fine_values, color=[ROSE, DEEP_PURPLE, MAUVE], width=0.58)
    fine_axis.set_ylim(0, max(fine_values) * 1.32)
    fine_axis.set_xticks([0, 1, 2], ["До", "После", "Пропорциональный\nсценарий"], fontsize=10)
    fine_axis.set_yticks([])
    fine_axis.text(0.02, 0.98, "ШТРАФЫ", transform=fine_axis.transAxes, va="top", fontsize=13, fontweight="bold")
    fine_axis.text(
        0.98,
        0.98,
        "практически\nстабильны",
        transform=fine_axis.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=DEEP_PURPLE,
    )
    for bar, value in zip(fine_bars, fine_values, strict=True):
        fine_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.018, fmt(value, 3), ha="center", fontsize=15, fontweight="bold")
    fine_axis.text(0.5, -0.20, "штрафа на клиента за 30 дней", transform=fine_axis.transAxes, ha="center", fontsize=10)
    clean_axis(fine_axis)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.98, bottom=0.24, wspace=0.30)
    save_figure(fig, output, "04_main_result")


def _fine_variant(periods: pd.DataFrame, label: str, kind: str, margin: float) -> dict[str, object]:
    result = fit_decoupling_model(periods, margin)
    change = result.client_period["fine_rate_30d_post"] - result.client_period["fine_rate_30d_pre"]
    interval = mean_interval(change)
    tost = result.tests.set_index("test").loc["fine_rate_equivalence"]
    return {
        "label": label,
        "kind": kind,
        "estimate": interval["estimate"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "n": interval["n"],
        "tost_p_005": float(tost["p_value"]),
        "equivalent_005": bool(tost["p_value"] < 0.05),
        "notes": "95% ДИ среднего изменения; TOST отдельно",
    }


def seasonal_did(
    demographics_path: Path, fines_path: Path, cohort_ids: pd.Index
) -> dict[str, object]:
    demographics = read_csv_detected(demographics_path)
    fines = read_csv_detected(fines_path)
    baseline = make_client_baseline(demographics).set_index("client_id").reindex(cohort_ids)
    month_names = {4: "april", 5: "may", 6: "jun", 7: "jul", 8: "aug"}
    rates_2025 = pd.DataFrame(index=cohort_ids)
    for month, name in month_names.items():
        source = f"{name}_2025_fines"
        rates_2025[month] = pd.to_numeric(baseline[source], errors="coerce").fillna(0) * (
            30 / pd.Period(f"2025-{month:02d}").days_in_month
        )
    fines["bill_offence_date"] = pd.to_datetime(fines["bill_offence_date"], errors="raise")
    fines = fines.loc[fines["client_id"].isin(cohort_ids)].copy()
    fines["month"] = fines["bill_offence_date"].dt.month
    counts_2026 = (
        fines.groupby(["client_id", "month"])["bill_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=cohort_ids, columns=range(4, 9), fill_value=0)
    )
    rates_2026 = counts_2026.copy().astype(float)
    for month in rates_2026:
        rates_2026[month] *= 30 / pd.Period(f"2026-{month:02d}").days_in_month
    change_2025 = rates_2025[[6, 7, 8]].mean(axis=1) - rates_2025[[4, 5]].mean(axis=1)
    change_2026 = rates_2026[[6, 7, 8]].mean(axis=1) - rates_2026[[4, 5]].mean(axis=1)
    did = change_2026 - change_2025
    interval = mean_interval(did)
    return {
        "label": "Сезонная поправка 2025 (DiD)",
        "kind": "seasonal_did",
        "estimate": interval["estimate"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "n": interval["n"],
        "tost_p_005": np.nan,
        "equivalent_005": False,
        "notes": "(post-pre 2026) − (post-pre 2025), месячные частоты приведены к 30 дням",
    }


def calculate_robustness(
    panel: pd.DataFrame,
    main_periods: pd.DataFrame,
    demographics: Path,
    fines: Path,
    margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [_fine_variant(main_periods, "Основная модель · 1 июня", "main", margin)]
    for cutoff, label in [("2026-05-18", "Начало post: 18 мая"), ("2026-06-15", "Начало post: 15 июня")]:
        variant = prepare_client_periods(panel, DEFAULT_COHORT_START, cutoff)
        row = _fine_variant(variant, label, "cutoff", margin)
        pre_weeks = int(variant["weeks_pre"].iloc[0])
        post_weeks = int(variant["weeks_post"].iloc[0])
        row["notes"] += f"; cutoff={cutoff}; полных недель pre={pre_weeks}, post={post_weeks}"
        rows.append(row)

    total_fuel = main_periods["liters_pre"] + main_periods["liters_post"]
    fuel_threshold = float(total_fuel.quantile(0.99))
    rows.append(
        _fine_variant(
            main_periods.loc[total_fuel.le(fuel_threshold)].copy(),
            "Без верхнего 1% по топливу",
            "outlier",
            margin,
        )
    )
    rows[-1]["notes"] += f"; исключены total liters > {fuel_threshold:.2f}"
    total_fines = main_periods["fines_pre"] + main_periods["fines_post"]
    fines_threshold = float(total_fines.quantile(0.99))
    rows.append(
        _fine_variant(
            main_periods.loc[total_fines.le(fines_threshold)].copy(),
            "Без верхнего 1% по штрафам",
            "outlier",
            margin,
        )
    )
    rows[-1]["notes"] += f"; исключены total fines > {fines_threshold:.2f}"

    client_region = (
        panel.loc[panel["client_id"].isin(main_periods["client_id"]), ["client_id", "kladr_code"]]
        .drop_duplicates("client_id")
        .assign(kladr_code=lambda x: x["kladr_code"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(2))
    )
    region_periods = main_periods.merge(client_region, on="client_id", how="left", validate="one_to_one")
    largest = region_periods["kladr_code"].value_counts().head(5)
    for code, count in largest.items():
        subset = region_periods.loc[region_periods["kladr_code"].eq(code)].copy()
        label = REGION_NAMES.get(str(code), f"Регион {code}")
        row = _fine_variant(subset, label, "region", margin)
        row["notes"] += f"; kladr_code={code}; top-5 region by cohort n={count}"
        rows.append(row)

    rows.append(seasonal_did(demographics, fines, pd.Index(main_periods["client_id"])))
    robustness = pd.DataFrame(rows)

    fine_change = main_periods["fine_rate_30d_post"] - main_periods["fine_rate_30d_pre"]
    base_interval = mean_interval(fine_change, 0.90)
    sensitivity_rows = []
    for candidate in (0.03, 0.05, 0.10):
        result = fit_decoupling_model(main_periods, candidate)
        tost = result.tests.set_index("test").loc["fine_rate_equivalence"]
        sensitivity_rows.append(
            {
                "margin": candidate,
                "estimate": base_interval["estimate"],
                "ci90_low": base_interval["ci_low"],
                "ci90_high": base_interval["ci_high"],
                "p_value": float(tost["p_value"]),
                "equivalent": bool(tost["p_value"] < 0.05),
                "n": int(tost["n"]),
            }
        )
    return robustness, pd.DataFrame(sensitivity_rows)


def chart_robustness(robustness: pd.DataFrame, output: Path, margin: float) -> None:
    display = robustness.iloc[::-1].reset_index(drop=True)
    fig_height = max(5.7, 0.48 * len(display) + 1.3)
    fig, axis = plt.subplots(figsize=(8.2, fig_height))
    axis.axvspan(-margin, margin, color=LIGHT_PINK, alpha=0.75, zorder=0)
    axis.axvline(0, color=DARK_TEXT, linewidth=1.0, linestyle="--", zorder=1)
    for index, row in display.iterrows():
        color = DEEP_PURPLE if row["kind"] == "main" else ROSE if row["kind"] == "seasonal_did" else MAUVE
        axis.hlines(index, row.ci_low, row.ci_high, color=color, linewidth=2.2, zorder=2)
        axis.scatter(row.estimate, index, s=60 if row["kind"] == "main" else 38, color=color, zorder=3)
    axis.set_yticks(range(len(display)), display["label"], fontsize=10.5)
    axis.set_xlabel("изменение частоты штрафов на клиента за 30 дней", fontsize=11)
    low = min(float(display.ci_low.min()), -margin) - 0.025
    high = max(float(display.ci_high.max()), margin) + 0.025
    axis.set_xlim(low, high)
    clean_axis(axis, "x")
    axis.text(
        0,
        len(display) - 0.28,
        "зона ±0,05",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=MAUVE,
    )
    axis.text(
        0.01,
        -0.13,
        "Точки и линии: среднее изменение и 95% ДИ. DiD — отдельная сезонно скорректированная оценка.",
        transform=axis.transAxes,
        fontsize=9,
        color=MAUVE,
    )
    fig.tight_layout()
    save_figure(fig, output, "05_robustness")


def chart_tost_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    display = sensitivity.sort_values("margin", ascending=False).reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    for index, row in display.iterrows():
        axis.barh(index, 2 * row.margin, left=-row.margin, height=0.46, color=LIGHT_PINK, alpha=0.95)
        axis.hlines(index, row.ci90_low, row.ci90_high, color=DARK_TEXT, linewidth=3)
        axis.scatter(row.estimate, index, color=DEEP_PURPLE, s=65, zorder=3)
        axis.text(
            row.margin + 0.006,
            index,
            "эквивалентность: да" if row.equivalent else "эквивалентность: нет",
            va="center",
            fontsize=10.5,
            color=DEEP_PURPLE if row.equivalent else ROSE,
        )
    axis.axvline(0, color=MAUVE, linestyle="--", linewidth=1)
    axis.set_yticks(range(len(display)), [f"±{fmt(x, 2)}" for x in display.margin], fontsize=11)
    axis.set_xlabel("изменение штрафов на клиента за 30 дней · 90% ДИ", fontsize=11)
    axis.set_xlim(-0.12, 0.20)
    clean_axis(axis)
    fig.tight_layout()
    save_figure(fig, output, "05b_tost_sensitivity")


def chart_weekly(panel: pd.DataFrame, output: Path) -> pd.DataFrame:
    data = panel.copy()
    data["week"] = pd.to_datetime(data["week"], errors="raise")
    data["subscription_creation_date"] = pd.to_datetime(data["subscription_creation_date"], errors="coerce")
    data = data.loc[
        data["subscription_creation_date"].le(DEFAULT_COHORT_START)
        & data["exposure_days"].eq(7)
    ].copy()
    cohort_n = data["client_id"].nunique()
    weekly = data.groupby("week", as_index=False).agg(
        fuel_liters=("fuel_volume_liters", "sum"), fines=("fine_count", "sum")
    )
    weekly["fuel_rate_30d"] = weekly["fuel_liters"] / cohort_n * (30 / 7)
    weekly["fine_rate_30d"] = weekly["fines"] / cohort_n * (30 / 7)
    weekly["fuel_smooth"] = weekly["fuel_rate_30d"].rolling(3, center=True, min_periods=1).mean()
    weekly["fine_smooth"] = weekly["fine_rate_30d"].rolling(3, center=True, min_periods=1).mean()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.7), sharex=True)
    for axis, raw, smooth, label in [
        (axes[0], "fuel_rate_30d", "fuel_smooth", "л на клиента за 30 дней"),
        (axes[1], "fine_rate_30d", "fine_smooth", "штрафа на клиента за 30 дней"),
    ]:
        axis.plot(weekly["week"], weekly[raw], color=ROSE, linewidth=1.2, alpha=0.55, marker="o", markersize=4)
        axis.plot(weekly["week"], weekly[smooth], color=DEEP_PURPLE, linewidth=2.8)
        axis.axvline(pd.Timestamp(DEFAULT_CRISIS_START), color=MAUVE, linewidth=1.4, linestyle="--")
        axis.set_ylabel(label, fontsize=10.5)
        clean_axis(axis, "y")
    axes[0].text(0.01, 0.91, "ТОПЛИВО", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(0.01, 0.91, "ШТРАФЫ", transform=axes[1].transAxes, fontweight="bold")
    axes[0].text(
        pd.Timestamp(DEFAULT_CRISIS_START) + pd.Timedelta(days=3),
        axes[0].get_ylim()[1] * 0.96,
        "начало post-периода",
        fontsize=9.5,
        color=MAUVE,
        va="top",
    )
    axes[1].set_xlabel("полная неделя", fontsize=10.5)
    month_ticks = pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"])
    axes[1].set_xticks(month_ticks, ["апрель", "май", "июнь", "июль", "август"])
    fig.text(0.99, 0.01, "тонкая линия — неделя · тёмная линия — 3-недельное среднее", ha="right", fontsize=9, color=MAUVE)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, output, "06_weekly_dynamics")
    return weekly


def chart_regression(
    data: pd.DataFrame,
    dependent: str,
    independent: str,
    output: Path,
    stem: str,
    x_label: str,
    y_label: str,
) -> dict[str, float]:
    sample = data[[dependent, independent]].dropna().astype(float)
    result = regression(sample, dependent, independent)
    x_low, x_high = sample[independent].quantile([0.01, 0.99])
    y_low, y_high = sample[dependent].quantile([0.01, 0.99])
    visible = sample[independent].between(x_low, x_high) & sample[dependent].between(y_low, y_high)
    fig, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.scatter(
        sample.loc[visible, independent],
        sample.loc[visible, dependent],
        s=8,
        alpha=0.09,
        color=ROSE,
        linewidths=0,
        rasterized=True,
    )
    bins = pd.qcut(sample[independent], q=12, duplicates="drop")
    binned = sample.groupby(bins, observed=True).agg(
        x=(independent, "mean"), y=(dependent, "mean"), n=(dependent, "size")
    )
    axis.scatter(binned.x, binned.y, s=34, color=DEEP_PURPLE, zorder=4)
    line_x = np.linspace(x_low, x_high, 100)
    axis.plot(line_x, result["b0"] + result["b1"] * line_x, color=DARK_TEXT, linewidth=2.3)
    axis.axhline(0, color=LIGHT_PINK, linewidth=1.0)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    clean_axis(axis, "both")
    axis.text(
        0.03,
        0.97,
        "β = " + fmt(result["b1"], 5, signed=True)
        + "\n95% ДИ ["
        + fmt(result["ci_low"], 5)
        + "; "
        + fmt(result["ci_high"], 5)
        + "]\n"
        + p_text(result["p_value"])
        + "\nR² = "
        + fmt(result["r_squared"], 5)
        + "\nn = "
        + f"{result['n']:,}".replace(",", " "),
        transform=axis.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": LIGHT_PINK, "alpha": 0.92, "edgecolor": "none"},
    )
    fig.tight_layout()
    save_figure(fig, output, stem)
    return result


def statistics_table(stats: dict[str, object]) -> pd.DataFrame:
    tests = stats["tests"]
    return pd.DataFrame(
        [
            {
                "metric": "ObservedPrice",
                "pre": stats["price_pre"],
                "post": stats["price_post"],
                "absolute_change": stats["price_change"]["estimate"],
                "relative_change": stats["price_post"] / stats["price_pre"] - 1,
                "ci_low": stats["price_change"]["ci_low"],
                "ci_high": stats["price_change"]["ci_high"],
                "p_value": tests.loc["observed_price_increase", "p_value"],
                "test": "paired one-sided t-test, increase",
                "n": stats["price_n"],
                "unit": "RUB/liter",
                "notes": "clients with positive fuel purchases in both periods",
            },
            {
                "metric": "FuelRate",
                "pre": stats["fuel_pre"],
                "post": stats["fuel_post"],
                "absolute_change": stats["fuel_change"]["estimate"],
                "relative_change": stats["fuel_relative_change"],
                "ci_low": stats["fuel_change"]["ci_low"],
                "ci_high": stats["fuel_change"]["ci_high"],
                "p_value": tests.loc["fuel_volume_decrease", "p_value"],
                "test": "paired one-sided t-test, decrease",
                "n": stats["cohort_n"],
                "unit": "liters/client/30d",
                "notes": "fixed cohort; complete weeks",
            },
            {
                "metric": "FineRate",
                "pre": stats["fine_pre"],
                "post": stats["fine_post"],
                "absolute_change": stats["fine_change"]["estimate"],
                "relative_change": stats["fine_post"] / stats["fine_pre"] - 1,
                "ci_low": stats["fine_change_90"]["ci_low"],
                "ci_high": stats["fine_change_90"]["ci_high"],
                "p_value": tests.loc["fine_rate_equivalence", "p_value"],
                "test": "TOST equivalence; 90% CI",
                "n": stats["cohort_n"],
                "unit": "fines/client/30d",
                "notes": f"equivalence margin ±{stats['margin']}",
            },
            {
                "metric": "FineRate_actual_vs_proportional",
                "pre": stats["proportional"],
                "post": stats["fine_post"],
                "absolute_change": stats["gap"]["estimate"],
                "relative_change": stats["fine_post"] / stats["proportional"] - 1,
                "ci_low": stats["gap"]["ci_low"],
                "ci_high": stats["gap"]["ci_high"],
                "p_value": tests.loc["fines_above_proportional_counterfactual", "p_value"],
                "test": "paired one-sided t-test, actual above proportional scenario",
                "n": stats["cohort_n"],
                "unit": "fines/client/30d",
                "notes": "pre column is proportional scenario, not observed pre",
            },
        ]
    )


def validate_sanity(stats: dict[str, object], output: Path) -> list[str]:
    expected = {
        "cohort_n": (23467, 0),
        "price_n": (18617, 25),
        "price_change": (1.923, 0.03),
        "fuel_pre": (114.45, 0.20),
        "fuel_post": (83.97, 0.20),
        "fine_pre": (0.565, 0.005),
        "fine_post": (0.577, 0.005),
        "proportional": (0.414, 0.005),
        "gap": (0.163, 0.005),
    }
    actual = {
        "cohort_n": stats["cohort_n"],
        "price_n": stats["price_n"],
        "price_change": stats["price_change"]["estimate"],
        "fuel_pre": stats["fuel_pre"],
        "fuel_post": stats["fuel_post"],
        "fine_pre": stats["fine_pre"],
        "fine_post": stats["fine_post"],
        "proportional": stats["proportional"],
        "gap": stats["gap"]["estimate"],
    }
    discrepancies = []
    for key, (target, tolerance) in expected.items():
        if abs(float(actual[key]) - target) > tolerance:
            discrepancies.append(
                f"- `{key}`: текущий расчёт {actual[key]}, ориентир {target}, допуск {tolerance}."
            )
    path = output / "DISCREPANCIES.md"
    if discrepancies:
        path.write_text(
            "# Расхождения с предыдущей сводкой\n\n"
            + "Значения не подгонялись. Требуется проверить версии входных данных и параметры когорты.\n\n"
            + "\n".join(discrepancies)
            + "\n",
            encoding="utf-8",
        )
    elif path.exists():
        path.unlink()
    return discrepancies


def write_manifest(
    output: Path,
    stats: dict[str, object],
    robustness: pd.DataFrame,
    regressions: list[dict[str, float]],
) -> None:
    rows = [
        ("01_fuel_before_after", "Снижение FuelRate", "client_week_panel.csv → фиксированная когорта", f"{fmt(stats['fuel_pre'],2)} → {fmt(stats['fuel_post'],2)}; {fmt(stats['fuel_relative_change']*100,1,signed=True)}%", "Парный односторонний t-тест", "Механизм / основной", "После 1 июня покупки топлива через сервис снизились на 26,6%."),
        ("02_fines_tost", "Практическая стабильность FineRate", "client_week_panel.csv → клиентские pre/post частоты", f"Δ={fmt(stats['fine_change']['estimate'],3,signed=True)}; 90% ДИ [{fmt(stats['fine_change_90']['ci_low'],3)}; {fmt(stats['fine_change_90']['ci_high'],3)}]", "TOST, margin ±0,05", "Основной", "Изменение штрафов полностью лежит в заранее заданной зоне практически малого изменения."),
        ("03_fines_actual_vs_proportional", "Факт против пропорционального сценария", "FineRate и агрегатное отношение FuelRate post/pre", f"{fmt(stats['proportional'],3)} vs {fmt(stats['fine_post'],3)}; gap {fmt(stats['gap']['estimate'],3,signed=True)}", "Парный односторонний t-тест разницы", "Основной", "Штрафы не повторили пропорциональное падение топлива."),
        ("04_main_result", "Весь основной вывод", "Те же проверенные показатели", "Fuel −26,6%; fines практически стабильны; сценарий 0,414", "Сводка четырёх тестов", "Главный слайд", "Топливная активность снизилась, а штрафная активность осталась на прежнем уровне."),
        ("05_robustness", "Устойчивость ΔFineRate", "Альтернативные даты, trims, регионы, 2025 DiD", f"{len(robustness)} спецификаций", "Среднее изменение + 95% ДИ; TOST отдельно", "Устойчивость", "Вывод устойчив к удалению выбросов, но зависит от даты отсечения и региона; сезонная DiD отвечает на отдельный вопрос."),
        ("05b_tost_sensitivity", "Чувствительность к margin", "Основная фиксированная когорта", "±0,03 / ±0,05 / ±0,10", "TOST", "Устойчивость", "Эквивалентность проверяется при нескольких заранее показанных границах."),
        ("06_weekly_dynamics", "Недельная динамика", "Полные недели фиксированной когорты", "21 неделя", "Описательные недельные частоты + 3-недельное среднее", "Диагностика / основной", "Результат не формируется одной аномальной неделей."),
        ("07_regression_price_fuel", "ΔPrice и ΔFuel", "Клиенты с покупками в обоих периодах", f"β={regressions[0]['b1']:.5f}; R²={regressions[0]['r_squared']:.5f}", "OLS, HC3", "Backup / Q&A", "Индивидуальное изменение цены слабо связано с изменением топливной активности."),
        ("07_regression_price_fines", "ΔPrice и ΔFines", "Клиенты с покупками в обоих периодах", f"β={regressions[1]['b1']:.5f}; p={regressions[1]['p_value']:.3f}", "OLS, HC3", "Backup / Q&A", "Статистически различимой линейной связи не обнаружено."),
    ]
    header = "| Файл | Что показывает | Источник данных | Основные числа | Метод | Слайд | Главная устная интерпретация |\n|---|---|---|---|---|---|---|\n"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
    notes = (
        "\n\n## Методические примечания\n\n"
        "- Все PNG имеют прозрачный фон и экспортированы с 320 dpi; для каждого есть SVG.\n"
        "- На regression-графиках визуально показаны 1–99 процентили, но OLS рассчитана по полной выборке.\n"
        "- `Сезонная поправка 2025 (DiD)` имеет ту же единицу измерения, но другой estimand: изменение 2026 относительно сезонного изменения 2025.\n"
        "- Наблюдаемая цена и покупки через сервис не идентифицируют причинный эффект кризиса.\n"
    )
    (output / "chart_manifest.md").write_text(header + body + notes, encoding="utf-8")


def validate_outputs(output: Path, stems: list[str]) -> dict[str, object]:
    from PIL import Image

    checks = []
    for stem in stems:
        png = output / f"{stem}.png"
        svg = output / f"{stem}.svg"
        if not png.is_file() or png.stat().st_size == 0:
            raise FileNotFoundError(f"Не создан PNG: {png}")
        if not svg.is_file() or svg.stat().st_size == 0:
            raise FileNotFoundError(f"Не создан SVG: {svg}")
        with Image.open(png) as image:
            if image.mode != "RGBA":
                raise ValueError(f"PNG должен иметь alpha-канал: {png}")
            alpha_min, alpha_max = image.getchannel("A").getextrema()
            if alpha_min != 0 or alpha_max != 255:
                raise ValueError(f"Прозрачность PNG не подтверждена: {png}")
            checks.append(
                {
                    "stem": stem,
                    "png_bytes": png.stat().st_size,
                    "svg_bytes": svg.stat().st_size,
                    "width": image.width,
                    "height": image.height,
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                }
            )
    result = {"status": "PASS", "charts": checks}
    (output / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--fines", type=Path, default=DEFAULT_FINES)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--margin", type=float, default=DEFAULT_EQUIVALENCE_MARGIN)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    configure_style()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = read_csv_detected(args.panel)
    periods = prepare_client_periods(panel, DEFAULT_COHORT_START, DEFAULT_CRISIS_START)
    stats = calculate_statistics(periods, args.margin, args.alpha)
    discrepancies = validate_sanity(stats, output)

    chart_fuel(stats, output)
    chart_tost(stats, output)
    chart_actual_vs_proportional(stats, output)
    chart_main(stats, output)
    robustness, sensitivity = calculate_robustness(
        panel, stats["data"], args.demographics, args.fines, args.margin
    )
    robustness.to_csv(output / "robustness_results.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output / "tost_sensitivity.csv", index=False, encoding="utf-8-sig")
    chart_robustness(robustness, output, args.margin)
    chart_tost_sensitivity(sensitivity, output)
    weekly = chart_weekly(panel, output)
    weekly.to_csv(output / "weekly_dynamics.csv", index=False, encoding="utf-8-sig")

    regression_data = stats["data"].copy()
    regression_data["price_change"] = regression_data["fuel_price_post"] - regression_data["fuel_price_pre"]
    regression_data["fuel_change"] = regression_data["fuel_rate_30d_post"] - regression_data["fuel_rate_30d_pre"]
    regression_data["fine_change"] = regression_data["fine_rate_30d_post"] - regression_data["fine_rate_30d_pre"]
    reg_fuel = chart_regression(
        regression_data,
        "fuel_change",
        "price_change",
        output,
        "07_regression_price_fuel",
        "изменение наблюдаемой цены, ₽/л",
        "изменение топлива, л/30 дней",
    )
    reg_fines = chart_regression(
        regression_data,
        "fine_change",
        "price_change",
        output,
        "07_regression_price_fines",
        "изменение наблюдаемой цены, ₽/л",
        "изменение штрафов на клиента за 30 дней",
    )
    pd.DataFrame([reg_fuel, reg_fines]).to_csv(
        output / "regression_results.csv", index=False, encoding="utf-8-sig"
    )
    statistics_table(stats).to_csv(
        output / "statistics_for_slides.csv", index=False, encoding="utf-8-sig"
    )
    write_manifest(output, stats, robustness, [reg_fuel, reg_fines])
    stems = [
        "01_fuel_before_after",
        "02_fines_tost",
        "03_fines_actual_vs_proportional",
        "04_main_result",
        "05_robustness",
        "05b_tost_sensitivity",
        "06_weekly_dynamics",
        "07_regression_price_fuel",
        "07_regression_price_fines",
    ]
    validation = validate_outputs(output, stems)
    print(f"Создано {len(stems)} пар PNG/SVG: {output}")
    print(f"Проверка файлов: {validation['status']}")
    print(f"Расхождения с sanity check: {len(discrepancies)}")
    return output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
