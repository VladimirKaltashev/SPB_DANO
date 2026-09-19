"""Клиентская витрина и сегментация водителей по данным до топливного кризиса.

Единица кластеризации — клиент. Сырые таблицы с автомобилями, штрафами и
заправками сначала независимо агрегируются, поэтому при объединении не возникает
декартова размножения строк.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

SEED = 42
PRE_START = pd.Timestamp("2026-03-20")
CRISIS_START = pd.Timestamp("2026-06-01")
AS_OF_DATE = CRISIS_START - pd.Timedelta(days=1)

HISTORICAL_FINE_COLUMNS = [
    "april_2025_fines",
    "may_2025_fines",
    "jun_2025_fines",
    "jul_2025_fines",
    "aug_2025_fines",
    "fines_last_6_month",
    "fines_last_12_month",
    "fines_last_24_month",
    "fines_last_36_month",
]

LOG_MODEL_COLUMNS = [
    "cars_count",
    "car_value_total",
    "car_price_median",
    "car_price_max",
    "car_price_range",
    "historical_fines_5m_per_car",
    "historical_fines_12m_per_car",
    "fuel_orders_per_week_pre",
    "fuel_liters_per_week_pre",
    "fuel_spend_per_week_pre",
    "fines_per_week_pre",
    "fine_amount_per_week_pre",
]

LINEAR_MODEL_COLUMNS = [
    "car_age_mean",
    "car_age_max",
    "engine_l_mean",
    "horsepower_max",
    "service_tenure_years",
    "fuel_active_days_per_week_pre",
    "fuel_weighted_price_pre",
    "fuel_median_fill_liters_pre",
    "fuel_refund_share_pre",
    "car_price_missing_share",
]

CATEGORICAL_MODEL_COLUMNS = ["gender", "age_type_code", "home_region_code"]

PROFILE_COLUMNS = {
    "cars_count": "Автомобилей",
    "car_value_total": "Стоимость автопарка",
    "car_price_median": "Медианная стоимость авто",
    "car_age_mean": "Средний возраст авто",
    "horsepower_max": "Максимальная мощность",
    "historical_fines_12m_per_car": "Штрафов за 12 мес. на авто",
    "fuel_orders_per_week_pre": "Заправок в неделю",
    "fuel_liters_per_week_pre": "Литров в неделю",
    "fuel_weighted_price_pre": "Средневзвешенная цена литра",
    "fines_per_week_pre": "Штрафов в неделю до кризиса",
    "fine_amount_per_week_pre": "Сумма штрафов в неделю",
}

AGE_LABELS = {
    "A00": "до 20 лет",
    "A20": "21–40 лет",
    "A40": "41–60 лет",
    "A60": "61 год и старше",
}

REGION_LABELS = {
    "77": "Москва",
    "50": "Московская область",
    "78": "Санкт-Петербург",
    "16": "Республика Татарстан",
    "66": "Свердловская область",
    "54": "Новосибирская область",
    "47": "Ленинградская область",
    "52": "Нижегородская область",
    "23": "Краснодарский край",
    "63": "Самарская область",
    "74": "Челябинская область",
    "42": "Кемеровская область",
    "72": "Тюменская область",
    "02": "Республика Башкортостан",
    "24": "Красноярский край",
    "55": "Омская область",
}


@dataclass
class ClusterResult:
    features: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: pd.DataFrame
    category_profiles: pd.DataFrame
    preprocessor: ColumnTransformer
    model: KMeans
    label_remap: dict[int, int]
    transformed: np.ndarray
    selected_k: int


def stable_mode(series: pd.Series):
    """Детерминированная мода без пропусков."""
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    modes = clean.mode(dropna=True)
    return modes.iloc[0]


def locate_data_dir(explicit: Path | None, project_root: Path) -> Path:
    candidates = [
        explicit,
        project_root / "data" / "raw",
        Path("/Users/vlad/DANO/SPB"),
    ]
    required = {"clients_demographics.csv", "fines_2026.csv", "fuel_transaction.csv"}
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            if required.issubset({path.name for path in candidate.iterdir()}):
                return candidate.resolve()
    raise FileNotFoundError(
        "Не найдены три CSV. Передайте папку через --data-dir или положите файлы в data/raw/."
    )


def load_sources(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demographics = pd.read_csv(
        data_dir / "clients_demographics.csv",
        sep=";",
        decimal=",",
        dtype={"kladr_code": "string"},
    )
    fines = pd.read_csv(data_dir / "fines_2026.csv", sep=";", decimal=",")
    fuel = pd.read_csv(data_dir / "fuel_transaction.csv", sep=";", decimal=",")
    demographics["kladr_code"] = demographics["kladr_code"].str.zfill(2)
    demographics["subscription_creation_date"] = pd.to_datetime(
        demographics["subscription_creation_date"], errors="coerce"
    )
    fines["bill_offence_date"] = pd.to_datetime(fines["bill_offence_date"], errors="coerce")
    fuel["order_datetime"] = pd.to_datetime(fuel["order_datetime"], errors="coerce")
    return demographics, fines, fuel


def validate_sources(
    demographics: pd.DataFrame, fines: pd.DataFrame, fuel: pd.DataFrame
) -> dict[str, int | float | str]:
    required_demo = {
        "client_id",
        "auto_document_id",
        "price",
        "auto_year",
        "engine_type",
        "subscription_creation_date",
        *HISTORICAL_FINE_COLUMNS,
    }
    required_fines = {
        "client_id",
        "bill_id",
        "total_fine_amount",
        "offence_short_statement",
        "bill_offence_date",
        "auto_document_id",
    }
    required_fuel = {
        "client_id",
        "order_id",
        "order_datetime",
        "order_fuel_volume",
        "order_fuel_price_1liter",
    }
    for name, frame, required in [
        ("clients_demographics", demographics, required_demo),
        ("fines_2026", fines, required_fines),
        ("fuel_transaction", fuel, required_fuel),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"В {name} отсутствуют столбцы: {missing}")

    demo_pairs = set(
        map(tuple, demographics[["client_id", "auto_document_id"]].drop_duplicates().to_numpy())
    )
    fine_pairs = fines[["client_id", "auto_document_id"]].drop_duplicates().to_numpy()
    unmatched_fine_pairs = sum(tuple(pair) not in demo_pairs for pair in fine_pairs)
    if unmatched_fine_pairs:
        raise ValueError(
            f"В штрафах найдено {unmatched_fine_pairs} пар client_id + auto_document_id "
            "без соответствия в демографии."
        )

    cars_per_client = demographics.groupby("client_id")["auto_document_id"].nunique()
    return {
        "demographics_rows": len(demographics),
        "fines_rows": len(fines),
        "fuel_rows": len(fuel),
        "clients": demographics["client_id"].nunique(),
        "clients_with_multiple_cars": int((cars_per_client > 1).sum()),
        "duplicate_client_car_rows": int(
            demographics.duplicated(["client_id", "auto_document_id"]).sum()
        ),
        "negative_fuel_transactions": int((fuel["order_fuel_volume"] < 0).sum()),
        "pre_crisis_start": PRE_START.date().isoformat(),
        "pre_crisis_end": AS_OF_DATE.date().isoformat(),
    }


def build_vehicle_features(demographics: pd.DataFrame) -> pd.DataFrame:
    demo = demographics.copy()
    engine_parts = demo["engine_type"].astype("string").str.extract(
        r"^\s*(\d+(?:[.,]\d+)?)\s*\((\d+(?:[.,]\d+)?)"
    )
    demo["engine_l"] = pd.to_numeric(
        engine_parts[0].str.replace(",", ".", regex=False), errors="coerce"
    )
    demo["horsepower"] = pd.to_numeric(
        engine_parts[1].str.replace(",", ".", regex=False), errors="coerce"
    )

    pair_aggregation = {
        "auto_mark": stable_mode,
        "auto_year": "median",
        "price": "median",
        "engine_l": "median",
        "horsepower": "median",
        "subscription_creation_date": "min",
        **{column: "max" for column in HISTORICAL_FINE_COLUMNS},
    }
    cars = (
        demo.groupby(["client_id", "auto_document_id"], as_index=False, sort=False)
        .agg(pair_aggregation)
        .copy()
    )
    cars["car_age_2026"] = (2026 - cars["auto_year"]).clip(lower=0)
    monthly_2025 = [
        "april_2025_fines",
        "may_2025_fines",
        "jun_2025_fines",
        "jul_2025_fines",
        "aug_2025_fines",
    ]
    cars["fines_2025_5m"] = cars[monthly_2025].sum(axis=1)
    cars["fines_7_12_month"] = (
        cars["fines_last_12_month"] - cars["fines_last_6_month"]
    ).clip(lower=0)
    cars["fines_13_24_month"] = (
        cars["fines_last_24_month"] - cars["fines_last_12_month"]
    ).clip(lower=0)
    cars["fines_25_36_month"] = (
        cars["fines_last_36_month"] - cars["fines_last_24_month"]
    ).clip(lower=0)

    grouped = cars.groupby("client_id", sort=False)
    result = grouped.agg(
        cars_count=("auto_document_id", "nunique"),
        brands_count=("auto_mark", "nunique"),
        car_price_mean=("price", "mean"),
        car_price_median=("price", "median"),
        car_price_min=("price", "min"),
        car_price_max=("price", "max"),
        car_price_std=("price", "std"),
        car_price_missing_count=("price", lambda values: int(values.isna().sum())),
        car_age_mean=("car_age_2026", "mean"),
        car_age_min=("car_age_2026", "min"),
        car_age_max=("car_age_2026", "max"),
        engine_l_mean=("engine_l", "mean"),
        engine_l_max=("engine_l", "max"),
        horsepower_mean=("horsepower", "mean"),
        horsepower_max=("horsepower", "max"),
        april_2025_fines_total=("april_2025_fines", "sum"),
        may_2025_fines_total=("may_2025_fines", "sum"),
        jun_2025_fines_total=("jun_2025_fines", "sum"),
        jul_2025_fines_total=("jul_2025_fines", "sum"),
        aug_2025_fines_total=("aug_2025_fines", "sum"),
        fines_2025_5m_total=("fines_2025_5m", "sum"),
        fines_last_6m_total=("fines_last_6_month", "sum"),
        fines_7_12m_total=("fines_7_12_month", "sum"),
        fines_13_24m_total=("fines_13_24_month", "sum"),
        fines_25_36m_total=("fines_25_36_month", "sum"),
        fines_last_12m_total=("fines_last_12_month", "sum"),
        fines_last_24m_total=("fines_last_24_month", "sum"),
        fines_last_36m_total=("fines_last_36_month", "sum"),
        first_subscription_date=("subscription_creation_date", "min"),
        last_subscription_date=("subscription_creation_date", "max"),
    )
    # sum(min_count=1) не превращает полностью неизвестную стоимость в ложный ноль.
    result["car_value_total"] = grouped["price"].sum(min_count=1)
    result["car_price_std"] = result["car_price_std"].fillna(0)
    result["car_price_range"] = result["car_price_max"] - result["car_price_min"]
    result["car_age_range"] = result["car_age_max"] - result["car_age_min"]
    result["car_price_missing_share"] = (
        result["car_price_missing_count"] / result["cars_count"]
    )
    result["historical_fines_5m_per_car"] = (
        result["fines_2025_5m_total"] / result["cars_count"]
    )
    result["historical_fines_12m_per_car"] = (
        result["fines_last_12m_total"] / result["cars_count"]
    )
    result["service_tenure_years"] = (
        (AS_OF_DATE - result["first_subscription_date"]).dt.days.clip(lower=0) / 365.25
    )
    result["subscription_after_cutoff"] = (
        result["first_subscription_date"] > AS_OF_DATE
    ).astype("int8")

    for source, target in [
        ("gender", "gender"),
        ("age_type_code", "age_type_code"),
        ("kladr_code", "home_region_code"),
    ]:
        result[target] = demo.groupby("client_id")[source].agg(stable_mode)
    return result


def build_fuel_features(fuel: pd.DataFrame) -> pd.DataFrame:
    pre = fuel[
        fuel["order_datetime"].ge(PRE_START)
        & fuel["order_datetime"].lt(CRISIS_START)
    ].copy()
    weeks = (CRISIS_START - PRE_START).days / 7
    pre["is_positive"] = pre["order_fuel_volume"] > 0
    positive = pre[pre["is_positive"]].copy()
    positive["fuel_spend"] = (
        positive["order_fuel_volume"] * positive["order_fuel_price_1liter"]
    )
    positive["weighted_price_numerator"] = positive["fuel_spend"]

    all_orders = pre.groupby("client_id").agg(
        fuel_transactions_pre=("order_id", "nunique"),
        fuel_refund_orders_pre=("is_positive", lambda values: int((~values).sum())),
        fuel_refund_volume_pre=(
            "order_fuel_volume",
            lambda values: float(-values[values < 0].sum()),
        ),
    )
    positive_agg = positive.groupby("client_id").agg(
        fuel_orders_pre=("order_id", "nunique"),
        fuel_active_days_pre=("order_datetime", lambda values: values.dt.date.nunique()),
        fuel_liters_pre=("order_fuel_volume", "sum"),
        fuel_spend_pre=("fuel_spend", "sum"),
        fuel_mean_fill_liters_pre=("order_fuel_volume", "mean"),
        fuel_median_fill_liters_pre=("order_fuel_volume", "median"),
        weighted_price_numerator=("weighted_price_numerator", "sum"),
    )
    positive_agg["fuel_weighted_price_pre"] = (
        positive_agg["weighted_price_numerator"] / positive_agg["fuel_liters_pre"]
    )
    positive_agg = positive_agg.drop(columns="weighted_price_numerator")
    result = all_orders.join(positive_agg, how="outer")
    result["fuel_refund_share_pre"] = (
        result["fuel_refund_orders_pre"] / result["fuel_transactions_pre"]
    ).fillna(0)
    for target, source in [
        ("fuel_orders_per_week_pre", "fuel_orders_pre"),
        ("fuel_active_days_per_week_pre", "fuel_active_days_pre"),
        ("fuel_liters_per_week_pre", "fuel_liters_pre"),
        ("fuel_spend_per_week_pre", "fuel_spend_pre"),
    ]:
        result[target] = result[source] / weeks
    result["has_pre_fuel"] = (result["fuel_orders_pre"].fillna(0) > 0).astype("int8")
    return result


def build_fine_features(fines: pd.DataFrame) -> pd.DataFrame:
    pre = fines[
        fines["bill_offence_date"].ge(PRE_START)
        & fines["bill_offence_date"].lt(CRISIS_START)
    ].copy()
    weeks = (CRISIS_START - PRE_START).days / 7
    pre["fine_amount_rub"] = pre["total_fine_amount"] / 100
    result = pre.groupby("client_id").agg(
        fines_pre=("bill_id", "nunique"),
        fine_amount_pre=("fine_amount_rub", "sum"),
        offence_types_pre=("offence_short_statement", "nunique"),
        fine_active_days_pre=("bill_offence_date", lambda values: values.dt.date.nunique()),
    )
    result["fines_per_week_pre"] = result["fines_pre"] / weeks
    result["fine_amount_per_week_pre"] = result["fine_amount_pre"] / weeks
    result["has_pre_fine"] = (result["fines_pre"] > 0).astype("int8")
    return result


def build_client_features(
    demographics: pd.DataFrame, fines: pd.DataFrame, fuel: pd.DataFrame
) -> pd.DataFrame:
    vehicles = build_vehicle_features(demographics)
    fuel_features = build_fuel_features(fuel)
    fine_features = build_fine_features(fines)
    features = vehicles.join(fuel_features, how="left").join(fine_features, how="left")

    zero_fuel_columns = [
        "fuel_transactions_pre",
        "fuel_refund_orders_pre",
        "fuel_refund_volume_pre",
        "fuel_orders_pre",
        "fuel_active_days_pre",
        "fuel_liters_pre",
        "fuel_spend_pre",
        "fuel_mean_fill_liters_pre",
        "fuel_median_fill_liters_pre",
        "fuel_refund_share_pre",
        "fuel_orders_per_week_pre",
        "fuel_active_days_per_week_pre",
        "fuel_liters_per_week_pre",
        "fuel_spend_per_week_pre",
        "has_pre_fuel",
    ]
    zero_fine_columns = [
        "fines_pre",
        "fine_amount_pre",
        "offence_types_pre",
        "fine_active_days_pre",
        "fines_per_week_pre",
        "fine_amount_per_week_pre",
        "has_pre_fine",
    ]
    features[zero_fuel_columns + zero_fine_columns] = features[
        zero_fuel_columns + zero_fine_columns
    ].fillna(0)
    features["fines_per_1000_liters_pre"] = np.where(
        features["fuel_liters_pre"] > 0,
        features["fines_pre"] / features["fuel_liters_pre"] * 1000,
        np.nan,
    )
    features.index.name = "client_id"
    return features.reset_index()


def make_preprocessor() -> ColumnTransformer:
    log_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("scale", RobustScaler(quantile_range=(10, 90))),
        ]
    )
    linear_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler(quantile_range=(10, 90))),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("log", log_pipeline, LOG_MODEL_COLUMNS),
            ("linear", linear_pipeline, LINEAR_MODEL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_MODEL_COLUMNS),
        ],
        transformer_weights={"log": 1.0, "linear": 1.0, "categorical": 0.6},
        sparse_threshold=0,
    )


def _metric_normalized(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    spread = series.max() - series.min()
    if np.isclose(spread, 0):
        return pd.Series(0.5, index=series.index)
    normalized = (series - series.min()) / spread
    return normalized if higher_is_better else 1 - normalized


def evaluate_cluster_counts(
    transformed: np.ndarray,
    candidates: Iterable[int] = range(3, 9),
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_size = min(5_000, len(transformed))
    sample_index = rng.choice(len(transformed), size=sample_size, replace=False)
    rows: list[dict[str, float | int]] = []
    for k in candidates:
        base = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=15,
            max_iter=500,
        )
        labels = base.fit_predict(transformed)
        stability_scores = []
        for alternate_seed in (7, 19, 73):
            alternate = KMeans(
                n_clusters=k,
                random_state=alternate_seed,
                n_init=8,
                max_iter=500,
            ).fit_predict(transformed)
            stability_scores.append(adjusted_rand_score(labels, alternate))
        shares = np.bincount(labels, minlength=k) / len(labels)
        sample_labels = labels[sample_index]
        sample_matrix = transformed[sample_index]
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(sample_matrix, sample_labels),
                "calinski_harabasz": calinski_harabasz_score(
                    sample_matrix, sample_labels
                ),
                "davies_bouldin": davies_bouldin_score(sample_matrix, sample_labels),
                "stability_ari": float(np.mean(stability_scores)),
                "min_cluster_share": float(shares.min()),
                "max_cluster_share": float(shares.max()),
                "inertia": float(base.inertia_),
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics["selection_score"] = (
        0.45 * _metric_normalized(diagnostics["silhouette"])
        + 0.15 * _metric_normalized(diagnostics["calinski_harabasz"])
        + 0.15 * _metric_normalized(diagnostics["davies_bouldin"], False)
        + 0.20 * _metric_normalized(diagnostics["stability_ari"])
        + 0.05 * _metric_normalized(diagnostics["min_cluster_share"])
        - 0.01 * (diagnostics["k"] - diagnostics["k"].min())
    )
    eligible = (diagnostics["min_cluster_share"] >= 0.03) & (
        diagnostics["stability_ari"] >= 0.70
    )
    diagnostics["eligible"] = eligible
    return diagnostics


def select_cluster_count(diagnostics: pd.DataFrame) -> int:
    eligible = diagnostics[diagnostics["eligible"]]
    pool = eligible if not eligible.empty else diagnostics
    return int(pool.loc[pool["selection_score"].idxmax(), "k"])


def make_cluster_labels(features: pd.DataFrame) -> dict[int, str]:
    descriptor_specs = {
        "cars_count": ("несколько автомобилей", "один автомобиль"),
        "car_value_total": ("дорогой автопарк", "доступный автопарк"),
        "car_age_mean": ("возрастные автомобили", "новые автомобили"),
        "historical_fines_12m_per_car": (
            "много исторических штрафов",
            "мало исторических штрафов",
        ),
        "fuel_liters_per_week_pre": (
            "высокий расход топлива",
            "низкая топливная активность",
        ),
        "fines_per_week_pre": (
            "частые штрафы до кризиса",
            "редкие штрафы до кризиса",
        ),
        "service_tenure_years": ("давние пользователи", "новые пользователи"),
    }
    overall = features[list(descriptor_specs)].mean(numeric_only=True)
    spread = features[list(descriptor_specs)].std(numeric_only=True).replace(0, np.nan)
    labels: dict[int, str] = {}
    for cluster_id, group in features.groupby("cluster_id"):
        z_scores = (group[list(descriptor_specs)].mean() - overall) / spread
        strongest = z_scores.abs().sort_values(ascending=False).head(2)
        parts = []
        for column in strongest.index:
            high, low = descriptor_specs[column]
            parts.append(high if z_scores[column] >= 0 else low)
        labels[int(cluster_id)] = f"Кластер {cluster_id}: " + ", ".join(parts)
    return labels


def build_profiles(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cluster_id, group in features.groupby("cluster_id", sort=True):
        row: dict[str, float | int | str] = {
            "cluster_id": int(cluster_id),
            "cluster_label": group["cluster_label"].iloc[0],
            "clients": len(group),
            "share": len(group) / len(features),
            "multi_car_share": float((group["cars_count"] > 1).mean()),
            "has_pre_fuel_share": float(group["has_pre_fuel"].mean()),
            "has_pre_fine_share": float(group["has_pre_fine"].mean()),
        }
        for column in PROFILE_COLUMNS:
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"median_{column}"] = float(group[column].median())
        rows.append(row)
    summary = pd.DataFrame(rows)

    category_rows = []
    for cluster_id, group in features.groupby("cluster_id", sort=True):
        for column in CATEGORICAL_MODEL_COLUMNS:
            distribution = group[column].fillna("__missing__").value_counts(normalize=True)
            for rank, (value, share) in enumerate(distribution.head(5).items(), start=1):
                category_rows.append(
                    {
                        "cluster_id": int(cluster_id),
                        "cluster_label": group["cluster_label"].iloc[0],
                        "feature": column,
                        "rank": rank,
                        "value": value,
                        "share": float(share),
                    }
                )
    return summary, pd.DataFrame(category_rows)


def fit_clustering(features: pd.DataFrame, requested_k: int | None = None) -> ClusterResult:
    model_columns = LOG_MODEL_COLUMNS + LINEAR_MODEL_COLUMNS + CATEGORICAL_MODEL_COLUMNS
    model_input = features[model_columns].copy()
    preprocessor = make_preprocessor()
    transformed = np.asarray(preprocessor.fit_transform(model_input), dtype=float)
    if not np.isfinite(transformed).all():
        raise ValueError("После подготовки признаков остались NaN или бесконечности.")

    diagnostics = evaluate_cluster_counts(transformed)
    selected_k = requested_k or select_cluster_count(diagnostics)
    if selected_k not in diagnostics["k"].tolist():
        raise ValueError("Число кластеров должно быть от 3 до 8.")
    diagnostics["selected"] = diagnostics["k"].eq(selected_k)

    model = KMeans(
        n_clusters=selected_k,
        random_state=SEED,
        n_init=30,
        max_iter=600,
    )
    result_features = features.copy()
    result_features["cluster_id"] = model.fit_predict(transformed)

    # Перенумеровываем сегменты по медиане недельного расхода топлива: так ID
    # остаются интерпретируемыми и стабильнее между запусками.
    order = (
        result_features.groupby("cluster_id")["fuel_liters_per_week_pre"]
        .median()
        .sort_values()
        .index
    )
    remap = {old: new for new, old in enumerate(order)}
    result_features["cluster_id"] = result_features["cluster_id"].map(remap).astype(int)
    labels = make_cluster_labels(result_features)
    result_features["cluster_label"] = result_features["cluster_id"].map(labels)
    summary, category_profiles = build_profiles(result_features)
    return ClusterResult(
        features=result_features,
        diagnostics=diagnostics,
        summary=summary,
        category_profiles=category_profiles,
        preprocessor=preprocessor,
        model=model,
        label_remap=remap,
        transformed=transformed,
        selected_k=selected_k,
    )


def save_plots(result: ClusterResult, output_dir: Path) -> None:
    cache_dir = output_dir.parent.parent / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    sizes = result.features["cluster_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(sizes.index.astype(str), sizes.values, color="#F2B705")
    ax.bar_label(bars, labels=[f"{value:,}".replace(",", " ") for value in sizes.values])
    ax.set(title="Размер кластеров", xlabel="Кластер", ylabel="Клиентов")
    fig.tight_layout()
    fig.savefig(figures_dir / "cluster_sizes.png", dpi=180)
    plt.close(fig)

    profile = result.features.groupby("cluster_id")[list(PROFILE_COLUMNS)].mean()
    overall_mean = result.features[list(PROFILE_COLUMNS)].mean()
    overall_std = result.features[list(PROFILE_COLUMNS)].std().replace(0, np.nan)
    standardized = ((profile - overall_mean) / overall_std).clip(-3, 3).astype(float)
    standardized.columns = [PROFILE_COLUMNS[column] for column in standardized.columns]
    fig, ax = plt.subplots(figsize=(14, max(4, result.selected_k * 0.8)))
    sns.heatmap(
        standardized,
        cmap="RdYlBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Отклонение от среднего, σ"},
        ax=ax,
    )
    ax.set(title="Профили кластеров", xlabel="Признак", ylabel="Кластер")
    fig.tight_layout()
    fig.savefig(figures_dir / "cluster_profiles_heatmap.png", dpi=180)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=SEED)
    coordinates = pca.fit_transform(result.transformed)
    rng = np.random.default_rng(SEED)
    sample_size = min(8_000, len(coordinates))
    sample = rng.choice(len(coordinates), size=sample_size, replace=False)
    scatter_data = pd.DataFrame(
        {
            "PC1": coordinates[sample, 0],
            "PC2": coordinates[sample, 1],
            "cluster": result.features.iloc[sample]["cluster_id"].astype(str).to_numpy(),
        }
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=scatter_data,
        x="PC1",
        y="PC2",
        hue="cluster",
        hue_order=[str(value) for value in sorted(result.features["cluster_id"].unique())],
        palette="tab10",
        alpha=0.45,
        s=18,
        linewidth=0,
        ax=ax,
    )
    ax.set(
        title=(
            "Кластеры в проекции PCA "
            f"({100 * pca.explained_variance_ratio_.sum():.1f}% дисперсии)"
        )
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "clusters_pca.png", dpi=180)
    plt.close(fig)


def _format_number(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "нет данных"
    return f"{value:,.{digits}f}".replace(",", " ")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Небольшая Markdown-таблица без необязательной зависимости tabulate."""
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def write_report(
    result: ClusterResult,
    validation: dict[str, int | float | str],
    data_dir: Path,
    output_dir: Path,
) -> None:
    diagnostics = result.diagnostics.copy()
    diagnostics_view = diagnostics[
        [
            "k",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "stability_ari",
            "min_cluster_share",
            "selection_score",
            "selected",
        ]
    ].round(4)
    lines = [
        "# Кластеризация водителей по докризисным данным",
        "",
        f"Источник: `{data_dir}`.",
        f"Окно признаков: {PRE_START.date()} — {AS_OF_DATE.date()}.",
        "Данные с 1 июня 2026 года и позже при обучении кластеров не использовались.",
        "",
        "## Контроль данных",
        "",
        f"- Клиентов: {validation['clients']:,}.",
        f"- Клиентов с несколькими автомобилями: "
        f"{validation['clients_with_multiple_cars']:,}.",
        f"- Повторяющихся строк клиент–автомобиль, свёрнутых до одной записи: "
        f"{validation['duplicate_client_car_rows']:,}.",
        f"- Отрицательных топливных операций, учтённых как возвраты/корректировки: "
        f"{validation['negative_fuel_transactions']:,}.",
        "",
        "## Выбор числа кластеров",
        "",
        "Кандидаты сравнивались по silhouette, Calinski–Harabasz, Davies–Bouldin, "
        "устойчивости ARI при смене seed и минимальной доле кластера.",
        "",
        dataframe_to_markdown(diagnostics_view),
        "",
        f"Выбрано кластеров: **{result.selected_k}**.",
        "",
        "## Профили кластеров",
        "",
    ]
    category_profiles = result.category_profiles
    for row in result.summary.itertuples(index=False):
        lines.extend(
            [
                f"### {row.cluster_label}",
                "",
                f"- Клиентов: {row.clients:,} ({row.share:.1%}).",
                f"- В среднем автомобилей: {row.mean_cars_count:.2f}; "
                f"несколько машин имеют {row.multi_car_share:.1%} клиентов.",
                f"- Медианная суммарная стоимость автопарка: "
                f"{_format_number(row.median_car_value_total, 0)} руб.",
                f"- Средний возраст автомобиля: {row.mean_car_age_mean:.1f} года; "
                f"максимальная мощность: {row.mean_horsepower_max:.0f} л.с.",
                f"- До кризиса: {row.mean_fuel_orders_per_week_pre:.2f} заправки и "
                f"{row.mean_fuel_liters_per_week_pre:.1f} л в неделю; "
                f"средневзвешенная цена "
                f"{_format_number(row.mean_fuel_weighted_price_pre)} руб./л.",
                f"- Покупки топлива в докризисном окне есть у "
                f"{row.has_pre_fuel_share:.1%} клиентов.",
                f"- Штрафы: {row.mean_historical_fines_12m_per_car:.2f} за предыдущие "
                f"12 месяцев на автомобиль и {row.mean_fines_per_week_pre:.2f} в неделю "
                f"непосредственно перед кризисом; хотя бы один докризисный штраф есть "
                f"у {row.has_pre_fine_share:.1%} клиентов.",
            ]
        )
        top_categories = category_profiles[
            (category_profiles["cluster_id"] == row.cluster_id)
            & (category_profiles["rank"] == 1)
        ]
        readable_names = {
            "gender": "пол",
            "age_type_code": "возрастная группа",
            "home_region_code": "домашний регион",
        }
        def readable_value(feature: str, value: str) -> str:
            if feature == "gender":
                return {"M": "мужской", "F": "женский"}.get(value, value)
            if feature == "age_type_code":
                return AGE_LABELS.get(value, value)
            if feature == "home_region_code":
                return REGION_LABELS.get(str(value).zfill(2), value)
            return value

        category_text = "; ".join(
            f"{readable_names[item.feature]} — "
            f"{readable_value(item.feature, str(item.value))} ({item.share:.1%})"
            for item in top_categories.itertuples(index=False)
        )
        lines.extend([f"- Наиболее частые категории: {category_text}.", ""])

    lines.extend(
        [
            "## Как использовать результат",
            "",
            "`client_features_precrisis.csv` содержит рассчитанные признаки и "
            "кластер каждого клиента. `client_clusters.csv` — компактное соответствие "
            "client_id → cluster_id. Для анализа кризиса эти группы следует соединить "
            "с клиентско-недельной панелью июня–августа, не переобучая кластеры.",
            "",
            "Кластеры являются описательной сегментацией, а не доказательством "
            "причинного влияния кризиса.",
        ]
    )
    (output_dir / "cluster_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    result: ClusterResult,
    validation: dict[str, int | float | str],
    data_dir: Path,
    output_dir: Path,
) -> None:
    tables_dir = output_dir / "tables"
    models_dir = output_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    result.features.to_csv(
        tables_dir / "client_features_precrisis.csv", index=False, encoding="utf-8-sig"
    )
    result.features[["client_id", "cluster_id", "cluster_label"]].to_csv(
        tables_dir / "client_clusters.csv", index=False, encoding="utf-8-sig"
    )
    result.summary.to_csv(
        tables_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig"
    )
    result.category_profiles.to_csv(
        tables_dir / "cluster_category_profiles.csv", index=False, encoding="utf-8-sig"
    )
    result.diagnostics.to_csv(
        tables_dir / "cluster_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(
        {
            "preprocessor": result.preprocessor,
            "model": result.model,
            "label_remap": result.label_remap,
            "selected_k": result.selected_k,
            "log_columns": LOG_MODEL_COLUMNS,
            "linear_columns": LINEAR_MODEL_COLUMNS,
            "categorical_columns": CATEGORICAL_MODEL_COLUMNS,
            "pre_start": PRE_START,
            "crisis_start": CRISIS_START,
        },
        models_dir / "client_clustering.joblib",
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(result, output_dir)
    write_report(result, validation, data_dir, output_dir)


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    requested_k: int | None = None,
) -> ClusterResult:
    demographics, fines, fuel = load_sources(data_dir)
    validation = validate_sources(demographics, fines, fuel)
    features = build_client_features(demographics, fines, fuel)
    if len(features) != demographics["client_id"].nunique():
        raise AssertionError("В клиентской витрине должна быть ровно одна строка на клиента.")
    result = fit_clustering(features, requested_k=requested_k)
    save_outputs(result, validation, data_dir, output_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="По умолчанию outputs/clustering внутри проекта.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Фиксированное число кластеров от 3 до 8; без параметра выбирается автоматически.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_dir = locate_data_dir(args.data_dir, project_root)
    output_dir = args.output_dir or project_root / "outputs" / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(data_dir, output_dir, requested_k=args.clusters)
    print(f"Клиентов: {len(result.features):,}")
    print(f"Выбрано кластеров: {result.selected_k}")
    print(result.summary[["cluster_id", "cluster_label", "clients", "share"]].to_string(index=False))
    print(f"Результаты: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
