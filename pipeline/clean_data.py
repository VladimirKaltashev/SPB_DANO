"""Единственная воспроизводимая очистка raw CSV и построение недельной панели."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.preprocessing import (
    PAIR_KEYS,
    build_cleaning_report,
    clean_fuel_data,
    load_clients_data,
    load_fuel_data,
    resolve_client_auto_duplicates,
    resolve_field_conflict,
    save_cleaning_outputs,
    validate_cleaned_data,
)

START_DATE = pd.Timestamp("2026-04-01")
END_EXCLUSIVE = pd.Timestamp("2026-09-01")
CRISIS_START = pd.Timestamp("2026-06-01")
V2_DERIVED_COLUMNS = {
    "fuel_all_transaction_count",
    "fuel_positive_transaction_count",
    "fuel_volume_physical_teammate",
    "fuel_price_weighted",
    "client_has_multiple_autos",
    "is_after_subscription",
}


@dataclass(frozen=True)
class CleaningResult:
    output_dir: Path
    weekly_panel: Path
    clients: int
    panel_rows: int
    reference_matches: bool | None


def _read_fines(path: Path) -> pd.DataFrame:
    fines = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8",
        dtype={column: "string" for column in ["client_id", "bill_id", "auto_document_id"]},
        low_memory=False,
    )
    required = {
        "client_id",
        "bill_id",
        "auto_document_id",
        "bill_offence_date",
        "total_fine_amount",
    }
    missing = sorted(required - set(fines.columns))
    if missing:
        raise ValueError(f"В fines отсутствуют столбцы: {missing}")
    fines["bill_offence_date"] = pd.to_datetime(fines["bill_offence_date"], errors="coerce")
    if fines["bill_id"].isna().any() or not fines["bill_id"].is_unique:
        raise ValueError("bill_id в raw-штрафах должен быть заполнен и уникален")
    return fines


def build_client_level_features(client_auto: pd.DataFrame) -> pd.DataFrame:
    """Свернуть таблицу клиент×автомобиль до одной строки на клиента."""

    def aggregate(group: pd.DataFrame) -> pd.Series:
        result: dict[str, object] = {"client_id": group.name}
        for column in ["gender", "age_type_code", "kladr_code"]:
            result[column], _ = resolve_field_conflict(group[column])
        result["subscription_creation_date"] = group[
            "subscription_creation_date_first"
        ].min()
        result["car_count"] = group["auto_document_id"].nunique()
        result["median_car_price"] = group["price"].median()
        result["fines_2025"] = group["fines_last_12_month"].sum(min_count=1)
        for flag in [
            "has_duplicate_source_pair",
            "has_any_data_conflict",
            "fines_history_conflict",
            "vehicle_data_conflict",
            "subscription_date_conflict",
        ]:
            result[flag] = group[flag].max()
        return pd.Series(result)

    return (
        client_auto.groupby("client_id", sort=False)
        .apply(aggregate, include_groups=False)
        .reset_index(drop=True)
    )


def build_weekly_panel(
    client_features: pd.DataFrame,
    fines: pd.DataFrame,
    fuel: pd.DataFrame,
    start: pd.Timestamp = START_DATE,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
    crisis_start: pd.Timestamp = CRISIS_START,
) -> pd.DataFrame:
    """Построить полную панель клиент×неделя с нулевыми неделями."""
    fines = fines.loc[
        fines["bill_offence_date"].ge(start)
        & fines["bill_offence_date"].lt(end_exclusive)
    ].copy()
    fuel = fuel.loc[
        fuel["order_datetime"].ge(start) & fuel["order_datetime"].lt(end_exclusive)
    ].copy()
    fines["fine_amount_rub"] = pd.to_numeric(
        fines["total_fine_amount"], errors="coerce"
    ) / 100
    fines["week"] = fines["bill_offence_date"].dt.to_period("W-SUN").dt.start_time
    fuel["week"] = fuel["order_datetime"].dt.to_period("W-SUN").dt.start_time
    fuel["positive_price_for_mean"] = fuel["order_fuel_price_1liter"].where(
        fuel["order_fuel_volume"].gt(0)
    )
    fuel["is_positive_transaction"] = fuel["order_fuel_volume"].gt(0).astype(int)

    fines_weekly = fines.groupby(["client_id", "week"], as_index=False).agg(
        fine_count=("bill_id", "count"),
        fine_amount_rub=("fine_amount_rub", "sum"),
    )
    fuel_weekly = fuel.groupby(["client_id", "week"], as_index=False).agg(
        fuel_transaction_count=("physical_fuel_transaction_main", "sum"),
        fuel_volume_liters=("physical_fuel_volume_main", "sum"),
        fuel_volume_positive_only=("fuel_volume_positive_only", "sum"),
        fuel_volume_signed_net=("fuel_volume_signed_net", "sum"),
        fuel_price_mean=("positive_price_for_mean", "mean"),
        fuel_cost_rub=("fuel_cost_rub", "sum"),
        fuel_all_transaction_count=("order_id", "size"),
        fuel_positive_transaction_count=("is_positive_transaction", "sum"),
    )
    fuel_weekly["fuel_volume_physical_teammate"] = fuel_weekly["fuel_volume_liters"]
    fuel_weekly["fuel_price_weighted"] = fuel_weekly["fuel_cost_rub"].div(
        fuel_weekly["fuel_volume_liters"].where(fuel_weekly["fuel_volume_liters"].gt(0))
    )

    weeks = pd.date_range(
        start.to_period("W-SUN").start_time,
        (end_exclusive - pd.Timedelta(days=1)).to_period("W-SUN").start_time,
        freq="W-MON",
    )
    panel = pd.MultiIndex.from_product(
        [client_features["client_id"], weeks], names=["client_id", "week"]
    ).to_frame(index=False)
    panel["week_end"] = panel["week"] + pd.Timedelta(days=6)
    panel["observed_start"] = panel["week"].clip(lower=start)
    panel["observed_end"] = panel["week_end"].clip(
        upper=(end_exclusive - pd.Timedelta(days=1)).normalize()
    )
    panel["exposure_days"] = (panel["observed_end"] - panel["observed_start"]).dt.days + 1
    panel = panel.merge(fines_weekly, on=["client_id", "week"], how="left")
    panel = panel.merge(fuel_weekly, on=["client_id", "week"], how="left")
    panel = panel.merge(client_features, on="client_id", how="left")

    zero_columns = [
        "fine_count",
        "fine_amount_rub",
        "fuel_transaction_count",
        "fuel_volume_liters",
        "fuel_volume_positive_only",
        "fuel_volume_signed_net",
        "fuel_cost_rub",
        "fuel_all_transaction_count",
        "fuel_positive_transaction_count",
        "fuel_volume_physical_teammate",
    ]
    panel[zero_columns] = panel[zero_columns].fillna(0)
    panel["after_crisis"] = panel["week"].ge(crisis_start).astype(int)
    panel["client_has_multiple_autos"] = panel["car_count"].gt(1)
    panel["is_after_subscription"] = panel["observed_end"].ge(
        panel["subscription_creation_date"]
    )
    return panel


def compare_reference_panel(generated: Path, reference: Path) -> bool:
    """Проверить 27 базовых колонок прежнего v2-снимка по всем строкам."""
    generated_columns = pd.read_csv(generated, nrows=0).columns.tolist()
    reference_columns = pd.read_csv(reference, nrows=0).columns.tolist()
    common = [
        column
        for column in generated_columns
        if column in reference_columns and column not in V2_DERIVED_COLUMNS
    ]
    generated_chunks = pd.read_csv(generated, usecols=common, chunksize=100_000)
    reference_chunks = pd.read_csv(reference, usecols=common, chunksize=100_000)
    generated_rows = reference_rows = 0
    for generated_chunk, reference_chunk in zip(
        generated_chunks, reference_chunks, strict=True
    ):
        generated_rows += len(generated_chunk)
        reference_rows += len(reference_chunk)
        if len(generated_chunk) != len(reference_chunk):
            return False
        for column in common:
            left, right = generated_chunk[column], reference_chunk[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                equal = np.isclose(left, right, equal_nan=True, rtol=1e-10, atol=1e-8)
            else:
                equal = left.astype("string").eq(right.astype("string")) | (
                    left.isna() & right.isna()
                )
            if not bool(np.asarray(equal).all()):
                return False
    return generated_rows == reference_rows


def run_cleaning(
    raw_dir: Path,
    output_dir: Path,
    reference_panel: Path | None = None,
) -> CleaningResult:
    """Очистить три raw CSV, сохранить row-level таблицы и недельную панель."""
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    required = {
        "clients": raw_dir / "clients_demographics.csv",
        "fines": raw_dir / "fines_2026.csv",
        "fuel": raw_dir / "fuel_transaction.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Не найдены raw-файлы:\n- " + "\n- ".join(missing))

    clients = load_clients_data(required["clients"])
    fines = _read_fines(required["fines"])
    fuel = load_fuel_data(required["fuel"])
    client_auto_conflicts = clients.loc[
        clients.duplicated(PAIR_KEYS, keep=False)
    ].sort_values(PAIR_KEYS)
    client_auto, resolved_conflicts = resolve_client_auto_duplicates(clients)
    fuel_clean, matches, unresolved = clean_fuel_data(fuel)
    validate_cleaned_data(fuel, fuel_clean, matches, unresolved, client_auto)

    report = build_cleaning_report(
        fuel,
        fuel_clean,
        matches,
        unresolved,
        clients,
        client_auto,
        resolved_conflicts,
    )
    save_cleaning_outputs(
        output_dir,
        fuel_clean,
        matches,
        unresolved,
        client_auto_conflicts,
        resolved_conflicts,
        client_auto,
        report,
    )
    fines_clean = fines.loc[
        fines["bill_offence_date"].ge(START_DATE)
        & fines["bill_offence_date"].lt(END_EXCLUSIVE)
    ].copy()
    fines_clean["fine_amount_rub"] = pd.to_numeric(
        fines_clean["total_fine_amount"], errors="coerce"
    ) / 100
    fines_clean.to_csv(output_dir / "fines_clean.csv", index=False)

    features = build_client_level_features(client_auto)
    panel = build_weekly_panel(features, fines_clean, fuel_clean)
    weekly_panel = output_dir / "client_week_panel.csv"
    panel.to_csv(weekly_panel, index=False)

    reference_matches: bool | None = None
    if reference_panel is not None and reference_panel.is_file():
        reference_matches = compare_reference_panel(weekly_panel, reference_panel)
    metadata = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "period": [str(START_DATE.date()), str((END_EXCLUSIVE - pd.Timedelta(days=1)).date())],
        "clients": int(features["client_id"].nunique()),
        "fines": int(len(fines_clean)),
        "fuel_rows": int(len(fuel_clean)),
        "panel_rows": int(len(panel)),
        "reference_panel": str(reference_panel) if reference_panel is not None else None,
        "reference_matches_on_27_base_columns": reference_matches,
    }
    (output_dir / "cleaning_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return CleaningResult(
        output_dir=output_dir,
        weekly_panel=weekly_panel,
        clients=metadata["clients"],
        panel_rows=metadata["panel_rows"],
        reference_matches=reference_matches,
    )
