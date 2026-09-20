"""Driver labels, cell shares and within-cell diagnostics."""

import numpy as np
import pandas as pd


def aggregate_results(assignments: pd.DataFrame, cells: pd.DataFrame):
    labeled = assignments[assignments.risk_label.notna()]
    profiles = labeled.groupby(["region", "month", "risk_label"], as_index=False).agg(
        n_drivers=("client_id", "size"), mean_fines=("fines_count", "mean"),
        median_fines=("fines_count", "median"), mean_liters=("fuel_liters", "mean"),
        mean_engine_volume=("engine_volume", "mean"), mean_horsepower=("horsepower", "mean"))
    counts = labeled.groupby(["region", "month", "risk_label"]).size().unstack("risk_label")
    counts = counts.reindex(columns=["high_risk", "low_risk"]).rename(
        columns={"high_risk": "n_high_risk", "low_risk": "n_low_risk"})
    region_month = cells.merge(counts, on=["region", "month"], how="left", validate="one_to_one")
    region_month["share_high_risk"] = region_month.n_high_risk / region_month.n_drivers.replace(0, np.nan)
    region_month.loc[region_month.status.ne("ok"), "share_high_risk"] = np.nan
    good = assignments[assignments.cell_status.eq("ok")]
    pooled = good.groupby("risk_label", as_index=False).agg(
        n_driver_months=("client_id", "size"), n_unique_drivers=("client_id", "nunique"),
        mean_fines=("fines_count", "mean"), median_fines=("fines_count", "median"),
        mean_liters=("fuel_liters", "mean"), mean_engine_volume=("engine_volume", "mean"),
        mean_horsepower=("horsepower", "mean"))
    monthly = good.assign(is_high=good.risk_label.eq("high_risk")).groupby("month", as_index=False).agg(
        n_drivers=("client_id", "size"), n_high_risk=("is_high", "sum"),
        n_regions=("region", "nunique"))
    monthly["n_low_risk"] = monthly.n_drivers - monthly.n_high_risk
    monthly["share_high_risk"] = monthly.n_high_risk / monthly.n_drivers
    return region_month, profiles, pooled, monthly
