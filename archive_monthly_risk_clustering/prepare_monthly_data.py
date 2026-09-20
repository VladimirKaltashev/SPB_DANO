"""Calendar-month reconstruction, reconciled to the cleaned v2 weekly panel."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

REGION_NAMES = {
    "77": "Москва", "50": "Московская область", "78": "Санкт-Петербург",
    "16": "Республика Татарстан", "66": "Свердловская область",
    "54": "Новосибирская область", "47": "Ленинградская область",
    "52": "Нижегородская область", "23": "Краснодарский край",
    "63": "Самарская область", "74": "Челябинская область",
    "42": "Кемеровская область", "72": "Тюменская область",
    "02": "Республика Башкортостан", "24": "Красноярский край", "55": "Омская область",
}


def read_source(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig") as stream:
        if stream.readline().startswith("version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{path}: Git LFS pointer; run git lfs pull first")
    return pd.read_csv(path, dtype={"client_id": "string", "kladr_code": "string"},
                       low_memory=False)


def require_unique(data: pd.DataFrame, keys: list[str], name: str) -> None:
    if data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError(f"{name}: missing or duplicate keys {keys}")


def parse_engines(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Parse recorded engine strings; no inferred specifications or imputation."""
    result = vehicles.copy()
    parsed = result.engine_type.astype("string").str.extract(
        r"^\s*([\d.,]+)\s*\(\s*([\d.,]+)\s*л\.\s*с\.\s*\)\s*$"
    )
    result["engine_volume"] = pd.to_numeric(parsed[0].str.replace(",", "."), errors="coerce")
    result["horsepower"] = pd.to_numeric(parsed[1].str.replace(",", "."), errors="coerce")
    invalid = result.engine_volume.lt(0) | result.horsepower.le(0)
    result.loc[invalid, ["engine_volume", "horsepower"]] = np.nan
    result["engine_parse_missing"] = result[["engine_volume", "horsepower"]].isna().any(axis=1)
    # Flags only: retain recorded outliers rather than silently winsorizing KMeans input.
    result["extreme_engine"] = result.engine_volume.gt(8) | result.horsepower.gt(1000)
    return result


def reconcile_weekly(panel: pd.DataFrame, fines: pd.DataFrame,
                     fuel: pd.DataFrame) -> pd.DataFrame:
    """Every client-week must match, not just the grand totals."""
    keys = ["client_id", "week"]
    fine_week = fines.assign(week=fines.bill_offence_date.dt.to_period("W-SUN").dt.start_time)
    fuel_week = fuel.assign(week=fuel.order_datetime.dt.to_period("W-SUN").dt.start_time)
    original = panel.set_index(keys)[["fine_count", "fuel_volume_physical_teammate"]]
    rebuilt = fine_week.groupby(keys).size().rename("fines_rebuilt").to_frame().join(
        fuel_week.groupby(keys).physical_fuel_volume_main.sum().rename("liters_rebuilt"), how="outer")
    if len(rebuilt.index.difference(original.index)):
        raise ValueError("Detailed operations contain client-weeks absent from v2")
    joined = original.join(rebuilt).fillna({"fines_rebuilt": 0, "liters_rebuilt": 0})
    rows = []
    for left, right in [("fine_count", "fines_rebuilt"),
                        ("fuel_volume_physical_teammate", "liters_rebuilt")]:
        diff = (joined[left] - joined[right]).abs()
        rows.append(dict(metric=left, panel_total=joined[left].sum(),
                         rebuilt_total=joined[right].sum(), max_abs_difference=diff.max(),
                         mismatched_client_weeks=int(diff.gt(1e-6).sum())))
    audit = pd.DataFrame(rows)
    if audit.mismatched_client_weeks.any():
        raise ValueError(f"Detail does not reproduce v2:\n{audit.to_string(index=False)}")
    return audit


def aggregate_calendar_months(cohort: pd.DataFrame, fines: pd.DataFrame,
                              fuel: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Include zero-event drivers from v2; assign events by their actual dates."""
    grid = cohort.merge(pd.DataFrame({"month": months.astype(str)}), how="cross")
    fine_month = fines.assign(month=fines.bill_offence_date.dt.to_period("M").astype(str))
    fuel_month = fuel.assign(month=fuel.order_datetime.dt.to_period("M").astype(str))
    counts = fine_month.groupby(["client_id", "month"]).size().rename("fines_count")
    liters = fuel_month.groupby(["client_id", "month"]).physical_fuel_volume_main.sum().rename("fuel_liters")
    grid = grid.join(counts, on=["client_id", "month"]).join(liters, on=["client_id", "month"])
    grid[["fines_count", "fuel_liters"]] = grid[["fines_count", "fuel_liters"]].fillna(0)
    grid["fines_count"] = grid.fines_count.astype(int)
    return grid


def prepare_data(root: Path, panel_path: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = {"panel_v2": panel_path,
             **{name: root / "data" / "processed" / filename for name, filename in {
                 "vehicles": "clients_demographics_clean.csv", "fines": "fines_clean.csv",
                 "fuel": "fuel_clean.csv"}.items()}}
    panel, vehicles, fines, fuel = [read_source(paths[name]) for name in paths]
    for column in ["week", "observed_start", "observed_end", "subscription_creation_date"]:
        panel[column] = pd.to_datetime(panel[column], errors="raise")
    fines["bill_offence_date"] = pd.to_datetime(fines.bill_offence_date, errors="raise")
    fuel["order_datetime"] = pd.to_datetime(fuel.order_datetime, errors="raise")
    require_unique(panel, ["client_id", "week"], "v2")
    require_unique(vehicles, ["client_id", "auto_document_id"], "vehicles")
    require_unique(fines, ["bill_id"], "fines")
    require_unique(fuel, ["order_id"], "fuel")
    start, end = panel.observed_start.min(), panel.observed_end.max()
    if start.day != 1 or end != end.to_period("M").end_time.normalize():
        raise ValueError("v2 observation window must cover whole calendar months")
    months = pd.period_range(start, end, freq="M")
    if panel[["observed_start", "observed_end"]].isna().any().any():
        raise ValueError("v2 has missing observation dates")
    if not panel.groupby("client_id").exposure_days.sum().eq((end - start).days + 1).all():
        raise ValueError("v2 has incomplete client exposure; cannot assume a balanced monthly cohort")
    intervals = panel[["week", "observed_start", "observed_end", "exposure_days"]].drop_duplicates()
    require_unique(intervals, ["week"], "weekly intervals")
    intervals = intervals.sort_values("observed_start")
    if not intervals.observed_start.iloc[1:].reset_index(drop=True).eq(
        intervals.observed_end.iloc[:-1].reset_index(drop=True) + pd.Timedelta(days=1)
    ).all():
        raise ValueError("v2 weekly intervals contain gaps or overlaps")
    fuel_all = fuel
    fines = fines[fines.bill_offence_date.between(start, end + pd.Timedelta(days=1), inclusive="left")].copy()
    fuel = fuel[fuel.order_datetime.between(start, end + pd.Timedelta(days=1), inclusive="left")].copy()
    if not np.isfinite(fuel.physical_fuel_volume_main).all() or fuel.physical_fuel_volume_main.lt(0).any():
        raise ValueError("Missing/nonfinite/negative cleaned physical fuel volume")
    reconciliation = reconcile_weekly(panel, fines, fuel)
    reconciliation.to_csv(tables / "weekly_reconciliation.csv", index=False)

    panel["region"] = panel.kladr_code.str.replace(r"\.0$", "", regex=True).str.zfill(2)
    if panel.groupby("client_id").region.nunique(dropna=False).gt(1).any():
        raise ValueError("Client registration region changes within v2")
    cohort = panel[["client_id", "region", "subscription_creation_date", "car_count"]].drop_duplicates()
    require_unique(cohort, ["client_id"], "cohort")
    if cohort.region.isna().any():
        raise ValueError("Missing client region")
    cohort["region_name"] = cohort.region.map(REGION_NAMES).fillna(cohort.region)
    engines = parse_engines(vehicles)
    engines.to_csv(tables / "vehicle_features_audit.csv", index=False)
    characteristics = engines.groupby("client_id").agg(
        engine_volume=("engine_volume", "median"), horsepower=("horsepower", "median"),
        n_cars_with_missing_engine=("engine_parse_missing", "sum"),
        has_extreme_engine=("extreme_engine", "max"))
    cohort = cohort.join(characteristics, on="client_id")
    require_unique(cohort, ["client_id"], "enriched cohort")
    monthly = aggregate_calendar_months(cohort, fines, fuel, months)
    monthly["eligible"] = monthly[["engine_volume", "horsepower"]].notna().all(axis=1)
    monthly["exclusion_reason"] = np.where(monthly.eligible, "", "missing_engine_features")
    require_unique(monthly, ["client_id", "region", "month"], "monthly")
    monthly = monthly.sort_values(["region", "month", "client_id"]).reset_index(drop=True)
    monthly.to_csv(tables / "driver_region_month.csv", index=False)
    monthly.loc[~monthly.eligible].to_csv(tables / "excluded_driver_months.csv", index=False)

    sales = fuel.loc[fuel.physical_fuel_volume_main.gt(0)].merge(
        cohort[["client_id", "region", "region_name"]], on="client_id", validate="many_to_one")
    invalid_price = ~np.isfinite(sales.order_fuel_price_1liter) | sales.order_fuel_price_1liter.le(0)
    if invalid_price.any():
        raise ValueError(f"{int(invalid_price.sum())} physical sales have invalid prices")
    sales["month"] = sales.order_datetime.dt.to_period("M").astype(str)
    sales["cost"] = sales.physical_fuel_volume_main * sales.order_fuel_price_1liter
    prices = sales.groupby(["region", "region_name", "month"], as_index=False).agg(
        fuel_liters=("physical_fuel_volume_main", "sum"), fuel_cost_rub=("cost", "sum"),
        n_transactions=("order_id", "size"), mean_transaction_price=("order_fuel_price_1liter", "mean"))
    prices["avg_fuel_price"] = prices.fuel_cost_rub / prices.fuel_liters
    prices = prices.sort_values(["region", "month"])
    prices["price_change_rub"] = prices.groupby("region").avg_fuel_price.diff()
    prices["price_change_pct"] = prices.groupby("region").avg_fuel_price.pct_change(fill_method=None)
    fine_sub = fines.merge(cohort[["client_id", "subscription_creation_date"]], on="client_id", validate="many_to_one")
    metadata = {
        "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in paths.items()},
        "panel_rows": len(panel), "start": str(start.date()), "end": str(end.date()),
        "fuel_source_start": str(fuel_all.order_datetime.min()), "fuel_source_end": str(fuel_all.order_datetime.max()),
        "unique_clients": cohort.client_id.nunique(), "driver_months": len(monthly),
        "eligible_clients": monthly.loc[monthly.eligible, "client_id"].nunique(),
        "eligible_driver_months": int(monthly.eligible.sum()),
        "region_month_cells": monthly.groupby(["region", "month"]).ngroups,
        "multi_car_clients": int(cohort.car_count.gt(1).sum()),
        "unparsed_vehicle_rows": int(engines.engine_parse_missing.sum()),
        "extreme_vehicle_rows": int(engines.extreme_engine.sum()),
        "fines_before_subscription": int(fine_sub.bill_offence_date.lt(fine_sub.subscription_creation_date).sum()),
        "clients_subscription_after_window": int(cohort.subscription_creation_date.gt(end).sum()),
        "negative_fuel_rows_in_window": int(fuel.order_fuel_volume.lt(0).sum()),
        "unresolved_negative_fuel_rows": int(fuel.refund_match_status.eq("unresolved").sum()),
        "month_crossing_week_rows": int(panel.observed_start.dt.month.ne(panel.observed_end.dt.month).sum()),
        "total_fines": int(monthly.fines_count.sum()), "total_liters": float(monthly.fuel_liters.sum()),
    }
    return monthly, prices, metadata
