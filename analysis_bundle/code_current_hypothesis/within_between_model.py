"""Within-between model for the stable-driver-component hypothesis.

The unit of observation is a client-week.  The model separates observed fuel
activity into a client's period average (between component) and the weekly
deviation from that average (within component):

    fines_it = a + b_within * (fuel_it - mean_fuel_i)
                 + b_between * mean_fuel_i + week FE + controls_i + error_it.

The confirmatory test is H0: b_between <= b_within against
H1: b_between > b_within.  Both coefficients are estimated in one regression,
so the standard error of their difference includes their covariance.

This is an associational model.  Fuel purchases in the service are not total
mileage, and the model does not identify a causal effect of the fuel crisis.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PROJECT_ROOT / "data" / "processed" / "client_week_panel.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "within_between_model"

REQUIRED_COLUMNS = {
    "client_id",
    "week",
    "exposure_days",
    "fine_count",
    "fuel_volume_liters",
}


@dataclass(frozen=True)
class ModelFit:
    """Regression output needed for the hypothesis decision and reporting."""

    coefficients: pd.DataFrame
    comparison: pd.DataFrame
    diagnostics: dict[str, object]


def _numeric_with_missing_indicator(
    frame: pd.DataFrame, column: str, *, log1p: bool = False
) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce")
    result = pd.DataFrame(index=frame.index)
    if values.isna().any():
        result[f"{column}_missing"] = values.isna().astype(float)
    fill_value = float(values.median()) if values.notna().any() else 0.0
    values = values.fillna(fill_value).clip(lower=0)
    result[f"log1p_{column}" if log1p else column] = (
        np.log1p(values) if log1p else values
    )
    return result


def prepare_model_frame(panel: pd.DataFrame, min_weeks: int = 12) -> pd.DataFrame:
    """Validate and prepare a client-week panel without deleting event zeros."""
    missing = sorted(REQUIRED_COLUMNS - set(panel.columns))
    if missing:
        raise ValueError(f"В недельной панели отсутствуют колонки: {missing}")

    data = panel.copy()
    data["client_id"] = data["client_id"].astype("string")
    data["week"] = pd.to_datetime(data["week"], errors="raise")
    for column in ("exposure_days", "fine_count", "fuel_volume_liters"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data.duplicated(["client_id", "week"]).any():
        raise ValueError("Пара client_id × week должна быть уникальной")
    if not data["exposure_days"].between(1, 7).all():
        raise ValueError("exposure_days должен находиться в диапазоне 1–7")
    if data[["fine_count", "fuel_volume_liters"]].lt(0).any().any():
        raise ValueError("Штрафы и очищенный физический объём топлива не могут быть отрицательными")

    # Boundary weeks have unequal exposure. Full weeks make coefficients directly
    # comparable and avoid treating five observed days as seven.
    data = data.loc[data["exposure_days"].eq(7)].copy()
    observed_weeks = data.groupby("client_id")["week"].transform("nunique")
    data = data.loc[observed_weeks.ge(min_weeks)].copy()
    if data.empty:
        raise ValueError("После ограничения по длительности наблюдения не осталось данных")

    data["fuel_100l"] = data["fuel_volume_liters"] / 100.0
    data["fuel_between_100l"] = data.groupby("client_id")["fuel_100l"].transform("mean")
    data["fuel_within_100l"] = data["fuel_100l"] - data["fuel_between_100l"]
    if not np.allclose(
        data.groupby("client_id")["fuel_within_100l"].mean().to_numpy(),
        0.0,
        atol=1e-12,
    ):
        raise AssertionError("Внутриклиентские отклонения должны иметь нулевое среднее")
    return data.sort_values(["client_id", "week"]).reset_index(drop=True)


def build_design_matrix(data: pd.DataFrame, include_controls: bool = True) -> pd.DataFrame:
    """Create the joint Mundlak design with calendar-week fixed effects."""
    design = pd.DataFrame(
        {
            "fuel_within_100l": data["fuel_within_100l"],
            "fuel_between_100l": data["fuel_between_100l"],
        },
        index=data.index,
    )
    week = pd.get_dummies(
        data["week"].dt.strftime("%Y-%m-%d"), prefix="week", drop_first=True, dtype=float
    )
    design = pd.concat([design, week], axis=1)

    if include_controls:
        numeric_controls = {
            "fines_2025": True,
            "car_count": True,
            "median_car_price": True,
        }
        for column, log1p in numeric_controls.items():
            if column in data:
                design = pd.concat(
                    [design, _numeric_with_missing_indicator(data, column, log1p=log1p)],
                    axis=1,
                )
        for column in ("gender", "age_type_code", "kladr_code"):
            if column in data:
                values = data[column].astype("string").fillna("missing")
                dummies = pd.get_dummies(
                    values, prefix=column, drop_first=True, dtype=float
                )
                design = pd.concat([design, dummies], axis=1)

    design = sm.add_constant(design, has_constant="add").astype(float)
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError("Матрица модели содержит NaN или бесконечные значения")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Матрица модели не имеет полного ранга; проверьте контрольные переменные")
    return design


def _fit_with_cluster(
    y: pd.Series, design: pd.DataFrame, groups: pd.Series, label: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    group_values = groups.astype("string").fillna("missing")
    group_count = int(group_values.nunique())
    if group_count < 2:
        raise ValueError(f"Для кластеризации {label} нужно не менее двух кластеров")

    fitted = sm.OLS(y.astype(float), design).fit()
    robust = fitted.get_robustcov_results(
        cov_type="cluster",
        groups=group_values,
        use_correction=True,
        df_correction=True,
        use_t=True,
    )
    names = list(design.columns)
    covariance = np.asarray(robust.cov_params())
    ci = np.asarray(robust.conf_int())
    coefficients = pd.DataFrame(
        {
            "cluster": label,
            "term": names,
            "beta": robust.params,
            "standard_error": robust.bse,
            "p_value_two_sided": robust.pvalues,
            "ci95_low": ci[:, 0],
            "ci95_high": ci[:, 1],
        }
    )

    within_index = names.index("fuel_within_100l")
    between_index = names.index("fuel_between_100l")
    contrast = np.zeros(len(names))
    contrast[between_index] = 1.0
    contrast[within_index] = -1.0
    difference = float(contrast @ robust.params)
    difference_se = float(np.sqrt(contrast @ covariance @ contrast))
    degrees_freedom = group_count - 1
    statistic = difference / difference_se
    critical = float(t.ppf(0.975, degrees_freedom))
    comparison = pd.DataFrame(
        [
            {
                "cluster": label,
                "beta_between": robust.params[between_index],
                "beta_within": robust.params[within_index],
                "difference_between_minus_within": difference,
                "standard_error_difference": difference_se,
                "t_statistic": statistic,
                "degrees_freedom": degrees_freedom,
                "p_value_one_sided_between_greater": float(t.sf(statistic, degrees_freedom)),
                "p_value_two_sided_equality": float(2 * t.sf(abs(statistic), degrees_freedom)),
                "ci95_difference_low": difference - critical * difference_se,
                "ci95_difference_high": difference + critical * difference_se,
                "n_clusters": group_count,
            }
        ]
    )
    diagnostics = {
        "cluster": label,
        "n_clusters": group_count,
        "r_squared": float(fitted.rsquared),
        "condition_number": float(np.linalg.cond(design.to_numpy())),
    }
    return coefficients, comparison, diagnostics


def fit_within_between(data: pd.DataFrame, include_controls: bool = True) -> ModelFit:
    """Estimate the joint model and its prespecified coefficient contrast."""
    design = build_design_matrix(data, include_controls=include_controls)
    client_result = _fit_with_cluster(
        data["fine_count"], design, data["client_id"], "client"
    )
    results = [client_result]
    if "kladr_code" in data and data["kladr_code"].nunique(dropna=True) >= 2:
        results.append(
            _fit_with_cluster(data["fine_count"], design, data["kladr_code"], "region")
        )

    coefficients = pd.concat([item[0] for item in results], ignore_index=True)
    comparison = pd.concat([item[1] for item in results], ignore_index=True)
    diagnostics: dict[str, object] = {
        "n_observations": len(data),
        "n_clients": int(data["client_id"].nunique()),
        "n_weeks": int(data["week"].nunique()),
        "share_zero_fines": float(data["fine_count"].eq(0).mean()),
        "share_zero_fuel": float(data["fuel_volume_liters"].eq(0).mean()),
        "mean_weekly_fines": float(data["fine_count"].mean()),
        "mean_weekly_fuel_liters": float(data["fuel_volume_liters"].mean()),
        "include_controls": include_controls,
        "fits": [item[2] for item in results],
    }
    return ModelFit(coefficients, comparison, diagnostics)


def write_results(result: ModelFit, output_dir: Path, alpha: float = 0.05) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.coefficients.to_csv(output_dir / "coefficients.csv", index=False, encoding="utf-8-sig")
    result.comparison.to_csv(output_dir / "hypothesis_test.csv", index=False, encoding="utf-8-sig")
    (output_dir / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary = result.comparison.loc[result.comparison["cluster"].eq("client")].iloc[0]
    supported = bool(
        primary["difference_between_minus_within"] > 0
        and primary["p_value_one_sided_between_greater"] < alpha
    )
    lines = [
        "# Within–between модель штрафной активности",
        "",
        "## Проверяемая гипотеза",
        "",
        "H0: β_between ≤ β_within; H1: β_between > β_within.",
        "",
        "Единица наблюдения — клиент-неделя. Зависимая переменная — число штрафов; "
        "топливная активность измеряется наблюдаемыми покупками через сервис.",
        "",
        "## Основной результат",
        "",
        f"- β_between = {primary['beta_between']:.6f} штрафа на 100 л в неделю.",
        f"- β_within = {primary['beta_within']:.6f} штрафа на 100 л в неделю.",
        f"- Разность = {primary['difference_between_minus_within']:.6f}; "
        f"95% ДИ [{primary['ci95_difference_low']:.6f}; "
        f"{primary['ci95_difference_high']:.6f}].",
        f"- Односторонний p-value = {primary['p_value_one_sided_between_greater']:.6g}.",
        f"- Гипотеза при α={alpha}: **{'поддерживается' if supported else 'не получила достаточной поддержки'}**.",
        "",
        "## Интерпретация и ограничения",
        "",
        "Коэффициент within описывает связь краткосрочного изменения покупок топлива "
        "с изменением штрафов у того же клиента. Between описывает различия между "
        "клиентами после учёта календарной недели и доступных постоянных контролей.",
        "",
        "Результат является статистической ассоциацией. Покупки топлива через сервис "
        "не равны полному пробегу; ненаблюдаемые профессия, стиль использования "
        "автомобиля и покупки на других АЗС могут объяснять часть between-различий. "
        "Модель не доказывает причинное влияние топливного кризиса.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-weeks", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-controls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = read_csv_detected(args.panel)
    data = prepare_model_frame(panel, min_weeks=args.min_weeks)
    result = fit_within_between(data, include_controls=not args.no_controls)
    write_results(result, args.output_dir, alpha=args.alpha)
    print(result.comparison.to_string(index=False))
    print(f"Результаты сохранены: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
