"""Клиентская витрина без обучения кластеров по данным до топливного кризиса.

Единица кластеризации — клиент. Сырые таблицы с автомобилями, штрафами и
заправками сначала независимо агрегируются, поэтому при объединении не возникает
декартова размножения строк.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import (
    DataSources,
    SourceMode,
    read_csv_detected,
    resolve_data_sources,
)

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


def stable_mode(series: pd.Series):
    """Детерминированная мода без пропусков."""
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    modes = clean.mode(dropna=True)
    return modes.iloc[0]


def locate_data_dir(
    explicit: Path | None,
    project_root: Path,
    source: SourceMode = "auto",
) -> Path:
    """Обратимо-совместимый поиск каталога с приоритетом очищенных данных."""
    return resolve_data_sources(explicit, project_root, source=source).directory


def load_sources(
    data_dir: Path,
    sources: DataSources | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = sources or resolve_data_sources(
        data_dir,
        Path(__file__).resolve().parent,
    )
    demographics = read_csv_detected(
        sources.demographics_path,
        dtype={"kladr_code": "string"},
    )
    fines = read_csv_detected(sources.fines_path)
    fuel = read_csv_detected(sources.fuel_path)
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
        map(
            tuple,
            demographics[["client_id", "auto_document_id"]].drop_duplicates().to_numpy(),
        )
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
    engine_parts = (
        demo["engine_type"]
        .astype("string")
        .str.extract(r"^\s*(\d+(?:[.,]\d+)?)\s*\((\d+(?:[.,]\d+)?)")
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
    cars["fines_7_12_month"] = (cars["fines_last_12_month"] - cars["fines_last_6_month"]).clip(
        lower=0
    )
    cars["fines_13_24_month"] = (cars["fines_last_24_month"] - cars["fines_last_12_month"]).clip(
        lower=0
    )
    cars["fines_25_36_month"] = (cars["fines_last_36_month"] - cars["fines_last_24_month"]).clip(
        lower=0
    )

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
    result["car_price_missing_share"] = result["car_price_missing_count"] / result["cars_count"]
    result["historical_fines_5m_per_car"] = result["fines_2025_5m_total"] / result["cars_count"]
    result["historical_fines_12m_per_car"] = result["fines_last_12m_total"] / result["cars_count"]
    result["service_tenure_years"] = (AS_OF_DATE - result["first_subscription_date"]).dt.days.clip(
        lower=0
    ) / 365.25
    result["subscription_after_cutoff"] = (result["first_subscription_date"] > AS_OF_DATE).astype(
        "int8"
    )

    for source, target in [
        ("gender", "gender"),
        ("age_type_code", "age_type_code"),
        ("kladr_code", "home_region_code"),
    ]:
        result[target] = demo.groupby("client_id")[source].agg(stable_mode)
    return result


def build_fuel_features(fuel: pd.DataFrame) -> pd.DataFrame:
    pre = fuel[
        fuel["order_datetime"].ge(PRE_START) & fuel["order_datetime"].lt(CRISIS_START)
    ].copy()
    weeks = (CRISIS_START - PRE_START).days / 7
    raw_volume = pd.to_numeric(pre["order_fuel_volume"], errors="coerce")
    if "physical_fuel_volume_main" in pre:
        analysis_volume = pd.to_numeric(pre["physical_fuel_volume_main"], errors="coerce").fillna(0)
    else:
        analysis_volume = raw_volume
    if "physical_fuel_transaction_main" in pre:
        analysis_transaction = pd.to_numeric(
            pre["physical_fuel_transaction_main"], errors="coerce"
        ).fillna(0)
    else:
        analysis_transaction = analysis_volume.gt(0).astype("int8")
    if "is_negative_volume" in pre:
        is_refund = (
            pre["is_negative_volume"].astype("string").str.lower().isin({"true", "1", "yes"})
        )
    else:
        is_refund = raw_volume.lt(0)

    pre["analysis_fuel_volume"] = analysis_volume
    pre["analysis_fuel_transaction"] = analysis_transaction
    pre["is_refund"] = is_refund
    pre["refund_volume"] = raw_volume.where(is_refund, 0).abs()
    pre["is_positive"] = analysis_volume.gt(0) & analysis_transaction.gt(0)
    positive = pre[pre["is_positive"]].copy()
    calculated_spend = positive["analysis_fuel_volume"] * positive["order_fuel_price_1liter"]
    if "fuel_cost_rub" in positive:
        positive["fuel_spend"] = pd.to_numeric(positive["fuel_cost_rub"], errors="coerce").fillna(
            calculated_spend
        )
    else:
        positive["fuel_spend"] = calculated_spend
    positive["weighted_price_numerator"] = positive["fuel_spend"]

    all_orders = pre.groupby("client_id").agg(
        fuel_transactions_pre=("order_id", "nunique"),
        fuel_refund_orders_pre=("is_refund", "sum"),
        fuel_refund_volume_pre=("refund_volume", "sum"),
    )
    positive_agg = positive.groupby("client_id").agg(
        fuel_orders_pre=("order_id", "nunique"),
        fuel_active_days_pre=(
            "order_datetime",
            lambda values: values.dt.date.nunique(),
        ),
        fuel_liters_pre=("analysis_fuel_volume", "sum"),
        fuel_spend_pre=("fuel_spend", "sum"),
        fuel_mean_fill_liters_pre=("analysis_fuel_volume", "mean"),
        fuel_median_fill_liters_pre=("analysis_fuel_volume", "median"),
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
        fines["bill_offence_date"].ge(PRE_START) & fines["bill_offence_date"].lt(CRISIS_START)
    ].copy()
    weeks = (CRISIS_START - PRE_START).days / 7
    pre["fine_amount_rub"] = pre["total_fine_amount"] / 100
    result = pre.groupby("client_id").agg(
        fines_pre=("bill_id", "nunique"),
        fine_amount_pre=("fine_amount_rub", "sum"),
        offence_types_pre=("offence_short_statement", "nunique"),
        fine_active_days_pre=(
            "bill_offence_date",
            lambda values: values.dt.date.nunique(),
        ),
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


if __name__ == "__main__":
    from behavior_clusters import main

    main()
