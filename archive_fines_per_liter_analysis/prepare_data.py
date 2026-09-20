"""Reuse the archived calendar-month cohort and verify against cleaned sources."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig") as stream:
        if stream.readline().startswith("version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{path}: Git LFS pointer, download the object first")
    return pd.read_csv(path, dtype={"client_id": "string", "region": "string"},
                       low_memory=False, **kwargs)


def add_price_shock(prices: pd.DataFrame, baseline_month: str) -> pd.DataFrame:
    """Same formula/join as archive_monthly_risk_clustering.regression.add_price_shock."""
    baseline = prices.loc[prices.month.eq(baseline_month), ["region", "avg_fuel_price"]].rename(
        columns={"avg_fuel_price": "baseline_price_region"})
    if baseline.empty or baseline.region.duplicated().any():
        raise ValueError(f"Missing/duplicate baseline for {baseline_month}")
    result = prices.merge(baseline, on="region", how="left", validate="many_to_one")
    result["baseline_month"] = baseline_month
    result["price_shock"] = result.avg_fuel_price / result.baseline_price_region - 1
    return result


def prepare_data(root: Path, archive: Path, tables: Path):
    meta_path = archive / "outputs" / "metadata.json"
    archived_meta = json.loads(meta_path.read_text())
    input_paths = {
        "driver_month": archive / "outputs/tables/driver_region_month.csv",
        "archived_prices": archive / "outputs/tables/region_month_prices.csv",
        "risk_no_liters": archive / "outputs/tables/driver_clusters_no_liters.csv",
        "archive_metadata": meta_path,
        "panel_v2": root / "client_week_panel_v2.csv",
        "fines": root / "data/processed/fines_clean.csv",
        "fuel": root / "data/processed/fuel_clean.csv",
        "vehicles": root / "data/processed/clients_demographics_clean.csv",
    }
    # Refuse to silently combine current data with risk labels fitted on different inputs.
    for name, old in archived_meta["sources"].items():
        if sha256(input_paths[name]) != old["sha256"]:
            raise ValueError(f"{name} changed since the archived fit; reconcile inputs before analysis")
    columns = ["client_id", "region", "region_name", "month", "fines_count", "fuel_liters"]
    drivers = read_csv(input_paths["driver_month"], usecols=columns)
    keys = ["client_id", "region", "month"]
    if drivers[keys].isna().any().any() or drivers.duplicated(keys).any():
        raise ValueError("Driver-month keys must be nonmissing and unique")
    if not np.isfinite(drivers[["fines_count", "fuel_liters"]]).all().all():
        raise ValueError("Nonfinite driver fines/liters")
    if drivers[["fines_count", "fuel_liters"]].lt(0).any().any() or not drivers.fines_count.mod(1).eq(0).all():
        raise ValueError("Counts must be nonnegative integers and physical liters nonnegative")
    cohort = drivers[["client_id", "region", "region_name"]].drop_duplicates()
    if cohort.client_id.duplicated().any():
        raise ValueError("More than one registration region per driver")
    panel = read_csv(input_paths["panel_v2"], usecols=["client_id", "kladr_code", "observed_start", "observed_end"])
    panel["region"] = panel.kladr_code.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(2)
    v2_keys = panel[["client_id", "region"]].drop_duplicates()
    if set(map(tuple, cohort[["client_id", "region"]].to_numpy())) != set(map(tuple, v2_keys.to_numpy())):
        raise ValueError("Archived driver-month table does not cover the complete v2 cohort")
    start, end = pd.Timestamp(panel.observed_start.min()), pd.Timestamp(panel.observed_end.max())
    months = pd.period_range(start, end, freq="M").astype(str)
    if set(drivers.month.unique()) != set(months) or not drivers.groupby("client_id").size().eq(len(months)).all():
        raise ValueError("Incomplete calendar-month cohort")

    fines = read_csv(input_paths["fines"])
    fuel = read_csv(input_paths["fuel"])
    if fines.bill_id.isna().any() or fines.bill_id.duplicated().any() or fuel.order_id.isna().any() or fuel.order_id.duplicated().any():
        raise ValueError("Duplicate/missing operation IDs")
    fines["date"] = pd.to_datetime(fines.bill_offence_date, errors="raise")
    fuel["date"] = pd.to_datetime(fuel.order_datetime, errors="raise")
    fines = fines[fines.date.between(start, end + pd.Timedelta(days=1), inclusive="left")].copy()
    fuel = fuel[fuel.date.between(start, end + pd.Timedelta(days=1), inclusive="left")].copy()
    for name, data in [("fines", fines), ("fuel", fuel)]:
        if not data.client_id.isin(cohort.client_id).all():
            raise ValueError(f"{name}: operations outside v2 cohort")
        data["month"] = data.date.dt.to_period("M").astype(str)
    fine_counts = fines.groupby(["client_id", "month"]).size().rename("rebuilt_fines")
    fuel_sums = fuel.groupby(["client_id", "month"]).physical_fuel_volume_main.sum().rename("rebuilt_liters")
    verified = drivers.join(fine_counts, on=["client_id", "month"]).join(fuel_sums, on=["client_id", "month"])
    verified[["rebuilt_fines", "rebuilt_liters"]] = verified[["rebuilt_fines", "rebuilt_liters"]].fillna(0)
    audit = []
    for original, rebuilt in [("fines_count", "rebuilt_fines"), ("fuel_liters", "rebuilt_liters")]:
        difference = verified[original].sub(verified[rebuilt]).abs()
        audit.append(dict(metric=original, saved_total=verified[original].sum(),
                          rebuilt_total=verified[rebuilt].sum(), max_abs_difference=difference.max(),
                          mismatched_driver_months=int(difference.gt(1e-6).sum())))
    if any(row["mismatched_driver_months"] for row in audit):
        raise ValueError("Archived calendar-month values differ from cleaned operations")
    pd.DataFrame(audit).to_csv(tables / "source_reconciliation.csv", index=False)

    sales = fuel.loc[fuel.physical_fuel_volume_main.gt(0)].merge(
        cohort, on="client_id", how="left", validate="many_to_one")
    if not np.isfinite(sales.order_fuel_price_1liter).all() or sales.order_fuel_price_1liter.le(0).any():
        raise ValueError("Invalid positive-sale price")
    sales["cost_rub"] = sales.physical_fuel_volume_main * sales.order_fuel_price_1liter
    prices = sales.groupby(["region", "region_name", "month"], as_index=False).agg(
        price_liters=("physical_fuel_volume_main", "sum"), total_fuel_cost_rub=("cost_rub", "sum"),
        n_fuel_transactions=("order_id", "size"))
    prices["avg_fuel_price"] = prices.total_fuel_cost_rub / prices.price_liters
    prices = add_price_shock(prices, archived_meta["baseline_month"])
    old_prices = read_csv(input_paths["archived_prices"])
    compare = prices.merge(old_prices[["region", "month", "avg_fuel_price", "price_shock"]],
                           on=["region", "month"], suffixes=("", "_archived"), validate="one_to_one", how="outer")
    if compare.isna().any().any() or not np.allclose(compare.price_shock, compare.price_shock_archived, atol=1e-12, rtol=0):
        raise ValueError("PriceShock differs from the archived regional baseline")
    if not np.allclose(compare.avg_fuel_price, compare.avg_fuel_price_archived, atol=1e-10, rtol=0):
        raise ValueError("Regional prices differ from archived prices")
    compare.to_csv(tables / "price_reconciliation.csv", index=False)
    prices.to_csv(tables / "region_month_prices.csv", index=False)
    drivers.to_csv(tables / "driver_month.csv", index=False)

    labels = read_csv(input_paths["risk_no_liters"])
    if labels.duplicated(keys).any():
        raise ValueError("Duplicate archived risk keys")
    risk_check = labels.merge(drivers, on=keys, how="left", suffixes=("_label", "_actual"), validate="one_to_one")
    for column in ["fines_count", "fuel_liters"]:
        if not np.allclose(risk_check[f"{column}_label"], risk_check[f"{column}_actual"], atol=1e-6, rtol=0):
            raise ValueError("Archived risk labels refer to different driver-month outcomes")
    labels = labels.loc[labels.cell_status.eq("ok"), keys + ["risk_label"]]
    if not labels.risk_label.isin(["high_risk", "low_risk"]).all():
        raise ValueError("Unknown risk labels")
    metadata = {
        "sources": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in input_paths.items()},
        "start": str(start.date()), "end": str(end.date()), "baseline_month": archived_meta["baseline_month"],
        "n_drivers": cohort.client_id.nunique(), "n_driver_months": len(drivers),
        "n_regions": cohort.region.nunique(), "n_months": len(months), "n_cells": len(prices),
        "total_fines": int(drivers.fines_count.sum()), "total_liters": float(drivers.fuel_liters.sum()),
        "zero_liter_driver_months": int(drivers.fuel_liters.eq(0).sum()),
        "zero_liters_with_fines": int((drivers.fuel_liters.eq(0) & drivers.fines_count.gt(0)).sum()),
        "fines_in_zero_liter_months": int(drivers.loc[drivers.fuel_liters.eq(0), "fines_count"].sum()),
        "unclassified_drivers": cohort.client_id.nunique() - labels.client_id.nunique(),
        "unclassified_driver_months": len(drivers) - len(labels),
        "fines_before_subscription": archived_meta["fines_before_subscription"],
        "clients_subscription_after_window": archived_meta["clients_subscription_after_window"],
        "archived_negative_fuel_rows": archived_meta["negative_fuel_rows_in_window"],
    }
    return drivers, prices, labels, metadata
