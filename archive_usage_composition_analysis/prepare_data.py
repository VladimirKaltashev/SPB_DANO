"""Define fixed April groups without using future purchases or fines."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["LOW", "MEDIUM", "HIGH"]
ZERO = "ZERO_UNOBSERVED"
INSUFFICIENT = "INSUFFICIENT_BASELINE"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assign_baseline_groups(baseline, baseline_month="2026-04"):
    """Ties stay together. All classification inputs must be baseline observations."""
    result = baseline.copy()
    start = pd.Period(baseline_month, freq="M").start_time
    result["baseline_eligible"] = result.baseline_observed_days.eq(start.days_in_month) & result.subscription_creation_date.le(start)
    positive = result.baseline_eligible & result.baseline_fuel_liters.gt(0)
    q1, q2 = result.loc[positive, "baseline_fuel_liters"].quantile([1 / 3, 2 / 3])
    if not np.isfinite([q1, q2]).all() or q1 >= q2:
        raise ValueError("Cannot form three meaningful baseline tertiles without splitting ties")
    result["usage_group"] = INSUFFICIENT
    result.loc[result.baseline_eligible & result.baseline_fuel_liters.eq(0), "usage_group"] = ZERO
    result.loc[positive & result.baseline_fuel_liters.le(q1), "usage_group"] = "LOW"
    result.loc[positive & result.baseline_fuel_liters.gt(q1) & result.baseline_fuel_liters.le(q2), "usage_group"] = "MEDIUM"
    result.loc[positive & result.baseline_fuel_liters.gt(q2), "usage_group"] = "HIGH"
    if result.client_id.duplicated().any():
        raise ValueError("Baseline must contain one record per client")
    return result, {"q1_liters": float(q1), "q2_liters": float(q2)}


def prepare_data(root: Path, tables: Path, baseline_month="2026-04"):
    archive = root / "archive_monthly_risk_clustering/outputs"
    previous = root / "archive_fines_per_liter_analysis/outputs"
    paths = {
        "driver_month": previous / "tables/driver_month.csv",
        "prices": previous / "tables/region_month_prices.csv",
        "client_attributes": archive / "tables/driver_region_month.csv",
        "panel_v2": root / "data/reference/client_week_panel_v2.csv",
        "fines": root / "data/processed/fines_clean.csv",
        "fuel": root / "data/processed/fuel_clean.csv",
        "vehicles": root / "data/processed/clients_demographics_clean.csv",
    }
    prior = json.loads((previous / "metadata.json").read_text())
    for name in ["panel_v2", "fines", "fuel", "vehicles"]:
        if digest(paths[name]) != prior["sources"][name]["sha256"]:
            raise ValueError(f"{name} changed since the verified previous analysis")
    if digest(paths["client_attributes"]) != prior["sources"]["driver_month"]["sha256"]:
        raise ValueError("Archived calendar-month data changed")
    dtypes = {"client_id": "string", "region": "string"}
    drivers = pd.read_csv(paths["driver_month"], dtype=dtypes)
    prices = pd.read_csv(paths["prices"], dtype={"region": "string"})
    attributes = pd.read_csv(paths["client_attributes"], dtype=dtypes)
    keys = ["client_id", "region", "month"]
    if drivers.duplicated(keys).any() or drivers[keys].isna().any().any():
        raise ValueError("Driver-month keys must be unique and nonmissing")
    matched = drivers.merge(attributes[keys + ["fines_count", "fuel_liters"]], on=keys,
                            how="outer", suffixes=("", "_archive"), validate="one_to_one")
    if not np.allclose(matched.fines_count, matched.fines_count_archive) or not np.allclose(matched.fuel_liters, matched.fuel_liters_archive):
        raise ValueError("Working monthly observations differ from the verified archive")
    if prices.duplicated(["region", "month"]).any() or prices.price_shock.isna().any():
        raise ValueError("Invalid regional price panel")
    if not np.allclose(prices.price_shock, prices.avg_fuel_price / prices.baseline_price_region - 1):
        raise ValueError("PriceShock does not match the original formula")
    if set(prices.baseline_month) != {baseline_month}:
        raise ValueError("Usage baseline and existing regional price baseline must agree")

    panel = pd.read_csv(paths["panel_v2"], usecols=["client_id", "week", "observed_start", "observed_end"], dtype={"client_id": "string"})
    if panel.duplicated(["client_id", "week"]).any():
        raise ValueError("Duplicate v2 client-week")
    start, end = pd.Period(baseline_month, freq="M").start_time, pd.Period(baseline_month, freq="M").end_time.normalize()
    left = pd.to_datetime(panel.observed_start).clip(lower=start)
    right = pd.to_datetime(panel.observed_end).clip(upper=end)
    panel["baseline_observed_days"] = ((right - left).dt.days + 1).clip(lower=0)
    coverage = panel.groupby("client_id").baseline_observed_days.sum()
    if coverage.gt(start.days_in_month).any():
        raise ValueError("Overlapping baseline observation days")
    baseline = attributes.loc[attributes.month.eq(baseline_month),
                              ["client_id", "region", "region_name", "subscription_creation_date", "fuel_liters", "fines_count"]].copy()
    baseline = baseline.rename(columns={"fuel_liters": "baseline_fuel_liters", "fines_count": "baseline_fines"})
    baseline["subscription_creation_date"] = pd.to_datetime(baseline.subscription_creation_date, errors="raise")
    baseline = baseline.join(coverage, on="client_id")
    groups, cutoffs = assign_baseline_groups(baseline, baseline_month)
    groups.to_csv(tables / "fixed_baseline_groups.csv", index=False)
    drivers = drivers.merge(groups[["client_id", "usage_group", "baseline_eligible", "baseline_fuel_liters"]],
                            on="client_id", how="left", validate="many_to_one")
    if drivers.usage_group.isna().any() or drivers.groupby("client_id").usage_group.nunique().gt(1).any():
        raise ValueError("Usage group must be defined and fixed for every driver")
    drivers.to_csv(tables / "driver_month_fixed_groups.csv", index=False)
    summary = groups.groupby("usage_group", as_index=False).agg(
        n_drivers=("client_id", "size"), minimum_baseline_liters=("baseline_fuel_liters", "min"),
        maximum_baseline_liters=("baseline_fuel_liters", "max"), mean_baseline_liters=("baseline_fuel_liters", "mean"),
        baseline_total_fines=("baseline_fines", "sum"), baseline_fines_per_driver=("baseline_fines", "mean"),
        n_with_baseline_fines=("baseline_fines", lambda x: x.gt(0).sum()))
    summary["share_of_all_drivers"] = summary.n_drivers / len(groups)
    summary.to_csv(tables / "baseline_group_summary.csv", index=False)
    zero_rows = []
    for population, zero in [("all_clients_zero_baseline", groups.loc[groups.baseline_fuel_liters.eq(0)]),
                              ("eligible_zero_baseline", groups.loc[groups.usage_group.eq(ZERO)])]:
        future = drivers.loc[drivers.client_id.isin(zero.client_id) & drivers.month.gt(baseline_month)]
        n_future_buyers = future.loc[future.fuel_liters.gt(0), "client_id"].nunique()
        denominator = len(groups) if population.startswith("all_") else int(groups.baseline_eligible.sum())
        zero_rows.append(dict(population=population, n_drivers=len(zero), share_of_population=len(zero) / denominator,
                              baseline_fines=int(zero.baseline_fines.sum()), baseline_drivers_with_fines=int(zero.baseline_fines.gt(0).sum()),
                              future_buyers=n_future_buyers, share_buying_later=n_future_buyers / len(zero),
                              future_fines=int(future.fines_count.sum()), future_drivers_with_fines=future.loc[future.fines_count.gt(0), "client_id"].nunique()))
    zero_audit = pd.DataFrame(zero_rows)
    zero_audit.to_csv(tables / "zero_baseline_audit.csv", index=False)
    metadata = {"baseline_month": baseline_month, **cutoffs, "n_all_drivers": len(groups),
                "n_eligible_drivers": int(groups.baseline_eligible.sum()),
                "n_tertile_drivers": int(groups.usage_group.isin(GROUPS).sum()),
                "n_insufficient_baseline": int(groups.usage_group.eq(INSUFFICIENT).sum()),
                "months": sorted(drivers.month.unique()), "n_regions": int(drivers.region.nunique()),
                "sources": {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in paths.items()},
                "actual_driving_measure": "No odometer, GPS, trip count or observed mileage field found. Fuel transaction counts are another service-purchase proxy; fines are the outcome."}
    return drivers, prices, groups, summary, zero_audit, metadata
