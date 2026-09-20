r"""Бинарная кластеризация водителей по client_week_panel_v2.csv.

Кластеры строятся на апреле–мае 2026 года и проверяются на июне–августе.
Запуск: .venv/bin/python archive_insurance_driver_clustering/run_analysis.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

BASELINE_START, BASELINE_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-01")
VALIDATION_START, VALIDATION_END = BASELINE_END, pd.Timestamp("2026-09-01")
RANDOM_STATE = 42


@dataclass(frozen=True)
class ConsumptionScenario:
    name: str
    liters_per_100km: float


SCENARIOS = (
    ConsumptionScenario("economy", 6.5),
    ConsumptionScenario("base", 8.0),
    ConsumptionScenario("high", 10.0),
)
CLUSTER_FEATURES = [
    "log_estimated_km", "log_fuel_transactions", "smoothed_fines_per_1000_km",
    "average_fine_amount_rub", "fuel_active_week_share", "fine_active_week_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path,
        default=Path(__file__).resolve().parents[1] / "client_week_panel_v2.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/insurance_clustering")
    )
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найдена недельная панель: {path}")
    panel = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {
        "client_id", "observed_start", "observed_end", "exposure_days",
        "fine_count", "fine_amount_rub", "fuel_transaction_count",
        "fuel_volume_physical_teammate", "gender", "age_type_code",
        "kladr_code", "car_count", "median_car_price", "fines_2025",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"В {path.name} отсутствуют колонки: {missing}")
    panel["observed_start"] = pd.to_datetime(panel["observed_start"], errors="coerce")
    panel["observed_end"] = pd.to_datetime(panel["observed_end"], errors="coerce")
    if panel[["observed_start", "observed_end"]].isna().any().any():
        raise ValueError("В панели есть непарсящиеся даты периода наблюдения")
    return panel


def first_known(values: pd.Series, default: str = "unknown") -> object:
    known = values.dropna()
    return known.iloc[0] if len(known) else default


def aggregate_period(
    panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    scenario: ConsumptionScenario,
) -> pd.DataFrame:
    part = panel[
        panel["observed_start"].ge(start) & panel["observed_end"].lt(end)
    ].copy()
    if part.empty:
        raise ValueError(f"Нет строк панели в периоде {start.date()} — {end.date()}")
    numeric = [
        "exposure_days", "fine_count", "fine_amount_rub", "fuel_transaction_count",
        "fuel_volume_physical_teammate", "car_count", "median_car_price", "fines_2025",
    ]
    for column in numeric:
        part[column] = pd.to_numeric(part[column], errors="coerce")
    part["fuel_active_week"] = part["fuel_volume_physical_teammate"].fillna(0).gt(0)
    part["fine_active_week"] = part["fine_count"].fillna(0).gt(0)
    result = part.groupby("client_id", as_index=False).agg(
        observed_days=("exposure_days", "sum"), fine_count=("fine_count", "sum"),
        fine_amount_rub=("fine_amount_rub", "sum"),
        fuel_transactions=("fuel_transaction_count", "sum"),
        fuel_liters=("fuel_volume_physical_teammate", "sum"),
        observed_weeks=("observed_start", "size"),
        fuel_active_weeks=("fuel_active_week", "sum"),
        fine_active_weeks=("fine_active_week", "sum"),
        gender=("gender", first_known), age_type_code=("age_type_code", first_known),
        region_code=("kladr_code", first_known), cars_count=("car_count", "max"),
        vehicle_price_median=("median_car_price", "max"), fines_2025=("fines_2025", "max"),
    )
    result["fuel_liters"] = result["fuel_liters"].clip(lower=0)
    result["estimated_km"] = 100 * result["fuel_liters"] / scenario.liters_per_100km
    result["average_fine_amount_rub"] = (
        result["fine_amount_rub"] / result["fine_count"].clip(lower=1)
    )
    result["fuel_active_week_share"] = result["fuel_active_weeks"] / result["observed_weeks"].clip(lower=1)
    result["fine_active_week_share"] = result["fine_active_weeks"] / result["observed_weeks"].clip(lower=1)
    return result


def add_smoothed_rate(data: pd.DataFrame, prior_exposure_1000km: float = 0.5) -> pd.DataFrame:
    result = data.copy()
    exposure = result["estimated_km"].div(1000)
    observed = exposure.gt(0)
    global_rate = result.loc[observed, "fine_count"].sum() / exposure[observed].sum()
    result["smoothed_fines_per_1000_km"] = (
        result["fine_count"] + global_rate * prior_exposure_1000km
    ) / (exposure + prior_exposure_1000km)
    result["log_estimated_km"] = np.log1p(result["estimated_km"])
    result["log_fuel_transactions"] = np.log1p(result["fuel_transactions"])
    return result


def cluster_baseline(baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    eligible = add_smoothed_rate(baseline[baseline["estimated_km"].gt(0)].copy())
    cluster_input = eligible.copy()
    for column in CLUSTER_FEATURES:
        low, high = cluster_input[column].quantile([0.01, 0.99])
        cluster_input[column] = cluster_input[column].clip(low, high)
    transformer = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler()),
        ]), CLUSTER_FEATURES)], remainder="drop",
    )
    matrix = transformer.fit_transform(cluster_input)
    model = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=30).fit(matrix)
    eligible["raw_cluster_id"] = model.labels_
    ranking = eligible.groupby("raw_cluster_id")["smoothed_fines_per_1000_km"].mean().sort_values()
    rank_map = {raw: rank for rank, raw in enumerate(ranking.index)}
    eligible["cluster_id"] = eligible["raw_cluster_id"].map(rank_map)
    eligible["is_cautious_cluster"] = eligible["cluster_id"].eq(0)
    eligible["driver_type"] = np.where(
        eligible["is_cautious_cluster"], "аккуратный", "неаккуратный"
    )
    shares = eligible["cluster_id"].value_counts(normalize=True)
    diagnostics = {
        "silhouette": float(silhouette_score(
            matrix, model.labels_, sample_size=min(10_000, len(eligible)),
            random_state=RANDOM_STATE,
        )),
        "cautious_share": float(shares.get(0, 0)),
        "not_cautious_share": float(shares.get(1, 0)),
    }
    return eligible, diagnostics


def validate_clusters(
    assignments: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["client_id", "cluster_id", "is_cautious_cluster", "driver_type"]
    validation = future.merge(assignments[columns], on="client_id", how="inner")
    validation = validation[validation["estimated_km"].gt(0)].copy()
    validation["log_future_km"] = np.log(validation["estimated_km"].clip(lower=1))
    summary = validation.groupby(["cluster_id", "driver_type"], as_index=False).agg(
        clients=("client_id", "nunique"), future_km=("estimated_km", "sum"),
        future_fines=("fine_count", "sum"),
        clients_with_fine=("fine_count", lambda x: int(x.gt(0).sum())),
    )
    summary["future_fines_per_1000_km"] = 1000 * summary["future_fines"] / summary["future_km"]
    formula = (
        "fine_count ~ C(cluster_id, Treatment(reference=0)) + cars_count + "
        "C(age_type_code) + C(region_code)"
    )
    poisson = smf.glm(
        formula, data=validation, family=sm.families.Poisson(),
        offset=validation["log_future_km"],
    ).fit(cov_type="HC1")
    negative_binomial = smf.glm(
        formula, data=validation, family=sm.families.NegativeBinomial(alpha=1.0),
        offset=validation["log_future_km"],
    ).fit(cov_type="HC1")
    any_fine = validation.assign(any_future_fine=validation["fine_count"].gt(0).astype(int))
    logistic = smf.glm(
        "any_future_fine ~ C(cluster_id, Treatment(reference=0)) + log_future_km + "
        "cars_count + C(age_type_code) + C(region_code)",
        data=any_fine, family=sm.families.Binomial(),
    ).fit(cov_type="HC1")
    rows = []
    for name, model in [
        ("Poisson future fines with km offset", poisson),
        ("Negative Binomial future fines with km offset", negative_binomial),
        ("Logit any future fine", logistic),
    ]:
        term = next(item for item in model.params.index if "C(cluster_id" in item)
        low, high = model.conf_int().loc[term]
        rows.append({
            "model": name, "term": term, "coefficient": model.params[term],
            "std_error": model.bse[term], "p_value": model.pvalues[term],
            "effect_ratio": np.exp(model.params[term]),
            "effect_ratio_ci_low": np.exp(low), "effect_ratio_ci_high": np.exp(high),
            "poisson_dispersion": poisson.pearson_chi2 / poisson.df_resid
            if name.startswith("Poisson") else np.nan,
        })
    return summary, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.panel)
    assignments_by_scenario, summaries, models, diagnostics_rows = {}, [], [], []
    report = [
        "# Бинарная кластеризация водителей по client_week_panel_v2.csv\n",
        f"Источник: `{args.panel.resolve()}`. Кластеры строятся на апреле–мае "
        "и проверяются на июне–августе 2026 года.\n",
    ]
    for scenario in SCENARIOS:
        baseline = aggregate_period(panel, BASELINE_START, BASELINE_END, scenario)
        future = aggregate_period(panel, VALIDATION_START, VALIDATION_END, scenario)
        assignments, diagnostics = cluster_baseline(baseline)
        summary, model_results = validate_clusters(assignments, future)
        assignments.insert(1, "scenario", scenario.name)
        assignments["assumed_liters_per_100km"] = scenario.liters_per_100km
        summary.insert(0, "scenario", scenario.name)
        model_results.insert(0, "scenario", scenario.name)
        diagnostics_rows.append({"scenario": scenario.name, **diagnostics})
        summaries.append(summary)
        models.append(model_results)
        assignments.to_csv(args.output_dir / f"client_clusters_{scenario.name}.csv", index=False)
        assignments_by_scenario[scenario.name] = assignments[["client_id", "cluster_id"]]
        cautious = summary.loc[summary["cluster_id"].eq(0)].iloc[0]
        unsafe = summary.loc[summary["cluster_id"].eq(1)].iloc[0]
        report.append(
            f"## {scenario.name}: {scenario.liters_per_100km:g} л/100 км\n\n"
            f"Аккуратные: **{int(cautious.clients):,}**, "
            f"{cautious.future_fines_per_1000_km:.3f} будущих штрафа на 1000 км.  "
            f"Неаккуратные: **{int(unsafe.clients):,}**, "
            f"{unsafe.future_fines_per_1000_km:.3f} штрафа на 1000 км.\n"
        )
    stability, names = [], list(assignments_by_scenario)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            paired = assignments_by_scenario[left_name].merge(
                assignments_by_scenario[right_name], on="client_id", suffixes=("_left", "_right")
            )
            stability.append({
                "scenario_left": left_name, "scenario_right": right_name,
                "clients_compared": len(paired),
                "adjusted_rand_index": adjusted_rand_score(
                    paired["cluster_id_left"], paired["cluster_id_right"]
                ),
                "same_driver_type_share": paired["cluster_id_left"].eq(paired["cluster_id_right"]).mean(),
            })
    pd.DataFrame(diagnostics_rows).to_csv(args.output_dir / "cluster_diagnostics.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / "future_cluster_validation.csv", index=False)
    pd.concat(models, ignore_index=True).to_csv(args.output_dir / "future_risk_models.csv", index=False)
    pd.DataFrame(stability).to_csv(args.output_dir / "scenario_stability.csv", index=False)
    report.extend([
        "## Ограничения\n",
        "Панель не содержит характеристики двигателя, поэтому расход задан сценарно "
        "одним коэффициентом для всех клиентов. Покупки вне сервиса не наблюдаются, "
        "а штрафы являются proxy риска, но не страховыми убытками.\n",
    ])
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_dir / "consumption_scenarios.json").write_text(
        json.dumps([asdict(item) for item in SCENARIOS], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Результаты сохранены в {args.output_dir}")


if __name__ == "__main__":
    main()
