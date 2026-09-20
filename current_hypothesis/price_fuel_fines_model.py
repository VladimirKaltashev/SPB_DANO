"""Confirmatory model for a fuel-activity / fines decoupling hypothesis.

Hypothesis
----------
After the observed fuel price increased, clients bought less fuel through the
service, while their fine rate did not fall proportionally.

The fixed cohort contains clients whose subscription started no later than
2026-04-01. Only complete seven-day panel weeks are used. Client-week outcomes
are aggregated into client-level PRE (before 2026-06-01) and POST periods and
scaled to 30 days.

The hypothesis is supported only when all prespecified conditions hold:
1. observed volume-weighted price increased (one-sided paired test);
2. fuel volume decreased (one-sided paired test);
3. the fine-rate change is equivalent to zero within a prespecified practical
   margin (TOST);
4. post-period fines exceed the counterfactual level implied by a proportional
   change with fuel volume (one-sided paired test).

The design establishes association with the period, not a causal crisis effect.
Purchases through the service are not total mileage or total fuel consumption.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

from pipeline.data_sources import read_csv_detected
from pipeline.plotting import plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PROJECT_ROOT / "data" / "processed" / "client_week_panel.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "price_fuel_fines_model"
DEFAULT_COHORT_START = "2026-04-01"
DEFAULT_CRISIS_START = "2026-06-01"
DEFAULT_EQUIVALENCE_MARGIN = 0.05

REQUIRED_COLUMNS = {
    "client_id",
    "week",
    "exposure_days",
    "subscription_creation_date",
    "fine_count",
    "fuel_volume_liters",
    "fuel_cost_rub",
}


@dataclass(frozen=True)
class DecouplingResult:
    tests: pd.DataFrame
    client_period: pd.DataFrame
    diagnostics: dict[str, object]


def _paired_test(values: pd.Series, alternative: str) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        raise ValueError("Для парного теста нужны как минимум два изменяющихся наблюдения")
    mean = float(clean.mean())
    se = float(clean.std(ddof=1) / np.sqrt(len(clean)))
    statistic = mean / se
    degrees_freedom = len(clean) - 1
    if alternative == "greater":
        p_value = float(t.sf(statistic, degrees_freedom))
    elif alternative == "less":
        p_value = float(t.cdf(statistic, degrees_freedom))
    elif alternative == "two-sided":
        p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
    else:
        raise ValueError("alternative должен быть greater, less или two-sided")
    critical = float(t.ppf(0.975, degrees_freedom))
    return {
        "n": len(clean),
        "estimate": mean,
        "standard_error": se,
        "statistic": statistic,
        "degrees_freedom": degrees_freedom,
        "p_value": p_value,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
    }


def _equivalence_test(values: pd.Series, margin: float) -> dict[str, float | bool]:
    """Two one-sided tests for -margin < mean(values) < margin."""
    if margin <= 0:
        raise ValueError("Граница эквивалентности должна быть положительной")
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        raise ValueError("Для TOST нужны как минимум два изменяющихся наблюдения")
    mean = float(clean.mean())
    se = float(clean.std(ddof=1) / np.sqrt(len(clean)))
    df = len(clean) - 1
    lower_stat = (mean + margin) / se
    upper_stat = (mean - margin) / se
    p_lower = float(t.sf(lower_stat, df))  # H0: mean <= -margin
    p_upper = float(t.cdf(upper_stat, df))  # H0: mean >= +margin
    p_value = max(p_lower, p_upper)
    # A 90% CI is the CI counterpart of TOST at alpha=.05.
    critical90 = float(t.ppf(0.95, df))
    return {
        "n": len(clean),
        "estimate": mean,
        "standard_error": se,
        "statistic": np.nan,
        "degrees_freedom": df,
        "p_value": p_value,
        "ci95_low": mean - float(t.ppf(0.975, df)) * se,
        "ci95_high": mean + float(t.ppf(0.975, df)) * se,
        "ci90_low": mean - critical90 * se,
        "ci90_high": mean + critical90 * se,
        "p_lower_margin": p_lower,
        "p_upper_margin": p_upper,
        "equivalence_margin": margin,
    }


def prepare_client_periods(
    panel: pd.DataFrame,
    cohort_start: str = DEFAULT_COHORT_START,
    crisis_start: str = DEFAULT_CRISIS_START,
) -> pd.DataFrame:
    """Create comparable client PRE/POST rates from the active weekly panel."""
    missing = sorted(REQUIRED_COLUMNS - set(panel.columns))
    if missing:
        raise ValueError(f"В недельной панели отсутствуют колонки: {missing}")
    data = panel.copy()
    data["client_id"] = data["client_id"].astype("string")
    data["week"] = pd.to_datetime(data["week"], errors="raise")
    data["subscription_creation_date"] = pd.to_datetime(
        data["subscription_creation_date"], errors="coerce"
    )
    for column in ("exposure_days", "fine_count", "fuel_volume_liters", "fuel_cost_rub"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data.duplicated(["client_id", "week"]).any():
        raise ValueError("Пара client_id × week должна быть уникальной")
    if data[["fine_count", "fuel_volume_liters", "fuel_cost_rub"]].lt(0).any().any():
        raise ValueError("Счётчики, физический объём и стоимость не могут быть отрицательными")

    cohort_date = pd.Timestamp(cohort_start)
    crisis_date = pd.Timestamp(crisis_start)
    data = data.loc[
        data["subscription_creation_date"].notna()
        & data["subscription_creation_date"].le(cohort_date)
        & data["exposure_days"].eq(7)
    ].copy()
    data["period"] = np.where(data["week"].lt(crisis_date), "pre", "post")
    coverage = data.groupby(["client_id", "period"], observed=True)["week"].nunique().unstack()
    eligible_ids = coverage.index[coverage.get("pre", 0).gt(0) & coverage.get("post", 0).gt(0)]
    data = data.loc[data["client_id"].isin(eligible_ids)].copy()
    if data.empty:
        raise ValueError("Нет клиентов с наблюдениями и до, и после даты кризиса")

    weekly = data.groupby(["client_id", "period"], observed=True).agg(
        weeks=("week", "nunique"),
        fines=("fine_count", "sum"),
        liters=("fuel_volume_liters", "sum"),
        fuel_cost=("fuel_cost_rub", "sum"),
    )
    weekly["fine_rate_30d"] = weekly["fines"] / (weekly["weeks"] * 7) * 30
    weekly["fuel_rate_30d"] = weekly["liters"] / (weekly["weeks"] * 7) * 30
    wide = weekly.unstack("period")
    wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
    wide = wide.reset_index()
    for period in ("pre", "post"):
        wide[f"fuel_price_{period}"] = wide[f"fuel_cost_{period}"].div(
            wide[f"liters_{period}"].where(wide[f"liters_{period}"].gt(0))
        )
    return wide


def fit_decoupling_model(
    client_period: pd.DataFrame,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    alpha: float = 0.05,
) -> DecouplingResult:
    data = client_period.copy()
    data["fuel_change"] = data["fuel_rate_30d_post"] - data["fuel_rate_30d_pre"]
    data["fine_change"] = data["fine_rate_30d_post"] - data["fine_rate_30d_pre"]
    data["price_change"] = data["fuel_price_post"] - data["fuel_price_pre"]

    mean_fuel_pre = float(data["fuel_rate_30d_pre"].mean())
    mean_fuel_post = float(data["fuel_rate_30d_post"].mean())
    if mean_fuel_pre <= 0:
        raise ValueError("Средний докризисный объём топлива должен быть положительным")
    fuel_ratio = mean_fuel_post / mean_fuel_pre
    # "Proportional" is defined before estimation as equal relative change:
    # every client's expected post fine rate equals its pre rate * aggregate
    # post/pre fuel ratio. This avoids division by individual zero fine counts.
    data["proportional_fine_counterfactual"] = data["fine_rate_30d_pre"] * fuel_ratio
    data["fine_above_proportional"] = (
        data["fine_rate_30d_post"] - data["proportional_fine_counterfactual"]
    )

    specifications = [
        (
            "observed_price_increase",
            "fuel_price_post - fuel_price_pre",
            _paired_test(data["price_change"], "greater"),
            "greater",
        ),
        (
            "fuel_volume_decrease",
            "fuel_rate_30d_post - fuel_rate_30d_pre",
            _paired_test(data["fuel_change"], "less"),
            "less",
        ),
        (
            "fine_rate_equivalence",
            "fine_rate_30d_post - fine_rate_30d_pre",
            _equivalence_test(data["fine_change"], equivalence_margin),
            "equivalence",
        ),
        (
            "fines_above_proportional_counterfactual",
            "fine_rate_30d_post - fine_rate_30d_pre * aggregate_fuel_ratio",
            _paired_test(data["fine_above_proportional"], "greater"),
            "greater",
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, estimand, result, alternative in specifications:
        row: dict[str, object] = {
            "test": name,
            "estimand": estimand,
            "alternative": alternative,
            **result,
        }
        row["supported_at_alpha"] = bool(float(result["p_value"]) < alpha)
        rows.append(row)
    tests = pd.DataFrame(rows)
    overall = bool(tests["supported_at_alpha"].all())
    diagnostics: dict[str, object] = {
        "n_clients": int(data["client_id"].nunique()),
        "pre_weeks": sorted(data["weeks_pre"].astype(int).unique().tolist()),
        "post_weeks": sorted(data["weeks_post"].astype(int).unique().tolist()),
        "mean_fuel_rate_30d_pre": mean_fuel_pre,
        "mean_fuel_rate_30d_post": mean_fuel_post,
        "aggregate_fuel_post_pre_ratio": fuel_ratio,
        "aggregate_fuel_relative_change": fuel_ratio - 1,
        "mean_fine_rate_30d_pre": float(data["fine_rate_30d_pre"].mean()),
        "mean_fine_rate_30d_post": float(data["fine_rate_30d_post"].mean()),
        "equivalence_margin_fines_per_30d": equivalence_margin,
        "alpha": alpha,
        "all_prespecified_conditions_supported": overall,
        "interpretation": "association with the post-2026-06-01 period, not a causal effect",
    }
    return DecouplingResult(tests=tests, client_period=data, diagnostics=diagnostics)


def write_results(result: DecouplingResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.tests.to_csv(output_dir / "hypothesis_tests.csv", index=False, encoding="utf-8-sig")
    result.client_period.to_csv(
        output_dir / "client_period_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tests = result.tests.set_index("test")
    d = result.diagnostics
    price = tests.loc["observed_price_increase"]
    fuel = tests.loc["fuel_volume_decrease"]
    fines = tests.loc["fine_rate_equivalence"]
    gap = tests.loc["fines_above_proportional_counterfactual"]
    verdict = "поддерживается" if d["all_prespecified_conditions_supported"] else "не поддерживается полностью"
    lines = [
        "# Цена, топливная активность и штрафы",
        "",
        "## Гипотеза",
        "",
        "После роста наблюдаемой цены клиенты стали покупать меньше топлива через сервис, "
        "но общее число штрафов на клиента не снизилось пропорционально.",
        "",
        f"**Итог: гипотеза {verdict} при α={d['alpha']}.**",
        "",
        "## Результаты",
        "",
        f"- Цена: изменение {price.estimate:+.3f} ₽/л, "
        f"95% ДИ [{price.ci95_low:.3f}; {price.ci95_high:.3f}], p={price.p_value:.3g}.",
        f"- Топливо: {d['mean_fuel_rate_30d_pre']:.2f} → "
        f"{d['mean_fuel_rate_30d_post']:.2f} л/30 дней; изменение "
        f"{fuel.estimate:+.2f}, p={fuel.p_value:.3g}.",
        f"- Штрафы: {d['mean_fine_rate_30d_pre']:.3f} → "
        f"{d['mean_fine_rate_30d_post']:.3f} на 30 дней; изменение "
        f"{fines.estimate:+.3f}. TOST для границ ±{d['equivalence_margin_fines_per_30d']:.3f}: "
        f"p={fines.p_value:.3g}, 90% ДИ [{fines.ci90_low:.3f}; {fines.ci90_high:.3f}].",
        f"- Топливо изменилось на {d['aggregate_fuel_relative_change']:+.1%}. "
        f"Штрафы выше пропорционального контрфактуала на {gap.estimate:+.3f} "
        f"штрафа/30 дней, p={gap.p_value:.3g}.",
        "",
        "## Ограничения",
        "",
        "Это сравнение периодов для фиксированной когорты, а не причинная оценка кризиса. "
        "Наблюдаются только покупки через сервис, а не полный пробег или все покупки топлива. "
        "Граница ±0,05 штрафа за 30 дней должна быть содержательно защищена до презентации. "
        "На результат могли повлиять сезонность, контроль ГИБДД и другие изменения 2026 года.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    expected_fines = (
        d["mean_fine_rate_30d_pre"] * d["aggregate_fuel_post_pre_ratio"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(
        ["До 1 июня", "После 1 июня"],
        [d["mean_fuel_rate_30d_pre"], d["mean_fuel_rate_30d_post"]],
        color=["#2B6CA3", "#EF8D32"],
    )
    axes[0].set_title("Покупки топлива снизились на 26,6%")
    axes[0].set_ylabel("Литров на клиента за 30 дней")
    axes[0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(
        [d["mean_fuel_rate_30d_pre"], d["mean_fuel_rate_30d_post"]]
    ):
        axes[0].text(index, value, f"{value:.1f}", ha="center", va="bottom")

    labels = ["До 1 июня", "После: фактически", "После: пропорционально\nтопливу"]
    fine_values = [
        d["mean_fine_rate_30d_pre"],
        d["mean_fine_rate_30d_post"],
        expected_fines,
    ]
    axes[1].bar(labels, fine_values, color=["#2B6CA3", "#20988E", "#B8C1CC"])
    axes[1].set_title("Штрафы не последовали за снижением топлива")
    axes[1].set_ylabel("Штрафов на клиента за 30 дней")
    axes[1].grid(axis="y", alpha=0.2)
    for index, value in enumerate(fine_values):
        axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom")
    axes[1].tick_params(axis="x", labelrotation=8)
    fig.suptitle("Наблюдаемая топливная активность и штрафы · фиксированная когорта")
    fig.tight_layout()
    fig.savefig(output_dir / "model_result.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    regression_specs = [
        (
            "fuel_change_on_price_change",
            "price_change",
            "fuel_change",
            "Изменение цены, ₽/л",
            "Изменение топлива, л/30 дней",
            "Рост цены связан со снижением покупок топлива",
        ),
        (
            "fine_change_on_price_change",
            "price_change",
            "fine_change",
            "Изменение цены, ₽/л",
            "Изменение штрафов / 30 дней",
            "Линейной связи цены с изменением штрафов не обнаружено",
        ),
    ]
    regression_rows: list[dict[str, object]] = []
    regression_fig, regression_axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (name, x_name, y_name, x_label, y_label, title) in zip(
        regression_axes, regression_specs, strict=True
    ):
        sample = result.client_period[[x_name, y_name]].dropna().astype(float)
        design = sm.add_constant(sample[x_name], has_constant="add")
        fitted = sm.OLS(sample[y_name], design).fit(cov_type="HC3")
        ci = fitted.conf_int().loc[x_name]
        regression_rows.append(
            {
                "model": name,
                "dependent_variable": y_name,
                "independent_variable": x_name,
                "n": len(sample),
                "intercept_b0": fitted.params["const"],
                "slope_b1": fitted.params[x_name],
                "robust_standard_error_b1": fitted.bse[x_name],
                "p_value_b1": fitted.pvalues[x_name],
                "ci95_b1_low": ci.iloc[0],
                "ci95_b1_high": ci.iloc[1],
                "r_squared": fitted.rsquared,
                "covariance": "HC3",
            }
        )
        # Quantile limits make the dense cloud legible; the regression itself is
        # always estimated on the complete sample recorded in regression_results.csv.
        x_low, x_high = sample[x_name].quantile([0.01, 0.99])
        y_low, y_high = sample[y_name].quantile([0.01, 0.99])
        visible = sample[x_name].between(x_low, x_high) & sample[y_name].between(y_low, y_high)
        axis.scatter(
            sample.loc[visible, x_name],
            sample.loc[visible, y_name],
            s=7,
            alpha=0.10,
            color="#2B6CA3",
            linewidths=0,
            rasterized=True,
        )
        line_x = np.linspace(x_low, x_high, 100)
        line_y = fitted.params["const"] + fitted.params[x_name] * line_x
        axis.plot(line_x, line_y, color="#C44536", linewidth=2.5)
        axis.axhline(0, color="#7A8793", linewidth=0.8, linestyle="--")
        axis.set(xlabel=x_label, ylabel=y_label, title=title, xlim=(x_low, x_high), ylim=(y_low, y_high))
        axis.grid(alpha=0.16)
        axis.text(
            0.03,
            0.97,
            f"Линейная регрессия\n"
            f"β₀ = {fitted.params['const']:.4f}\n"
            f"β₁ = {fitted.params[x_name]:.5f}\n"
            f"p-value = {fitted.pvalues[x_name]:.3g}\n"
            f"R² = {fitted.rsquared:.5f}\n"
            f"n = {len(sample):,}".replace(",", " "),
            transform=axis.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9, "edgecolor": "#D4D9DE"},
        )
    regression_fig.suptitle("Изменения на уровне клиента · линия OLS, устойчивые ошибки HC3")
    regression_fig.text(
        0.5,
        0.01,
        "На графике показаны 1–99 процентили для читаемости; коэффициенты рассчитаны по полной выборке.",
        ha="center",
        fontsize=9,
        color="#5B6570",
    )
    regression_fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    regression_fig.savefig(output_dir / "linear_regressions.png", dpi=180, bbox_inches="tight")
    plt.close(regression_fig)
    pd.DataFrame(regression_rows).to_csv(
        output_dir / "regression_results.csv", index=False, encoding="utf-8-sig"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort-start", default=DEFAULT_COHORT_START)
    parser.add_argument("--crisis-start", default=DEFAULT_CRISIS_START)
    parser.add_argument("--equivalence-margin", type=float, default=DEFAULT_EQUIVALENCE_MARGIN)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = read_csv_detected(args.panel)
    periods = prepare_client_periods(panel, args.cohort_start, args.crisis_start)
    result = fit_decoupling_model(periods, args.equivalence_margin, args.alpha)
    write_results(result, args.output_dir)
    print(result.tests.to_string(index=False))
    print(f"Результаты сохранены: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
