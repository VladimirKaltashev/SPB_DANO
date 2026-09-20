"""Use fixed membership in denominators, including subsequent zero purchases."""

import numpy as np
import pandas as pd

from .prepare_data import GROUPS, ZERO


def group_panels(drivers, prices, baseline_month):
    panel = drivers.groupby(["region", "region_name", "month", "usage_group"], as_index=False).agg(
        n_drivers=("client_id", "size"), total_liters=("fuel_liters", "sum"),
        total_fines=("fines_count", "sum"), n_positive_buyers=("fuel_liters", lambda x: x.gt(0).sum()))
    panel["liters_per_driver"] = panel.total_liters / panel.n_drivers
    panel["fines_per_driver"] = panel.total_fines / panel.n_drivers
    base = panel.loc[panel.month.eq(baseline_month), ["region", "usage_group", "liters_per_driver"]].rename(
        columns={"liters_per_driver": "baseline_liters_per_driver"})
    panel = panel.merge(base, on=["region", "usage_group"], validate="many_to_one")
    panel["liters_index"] = 100 * panel.liters_per_driver / panel.baseline_liters_per_driver.replace(0, np.nan)
    panel["liters_change_percent"] = panel.liters_index - 100
    panel["liters_change_absolute"] = panel.liters_per_driver - panel.baseline_liters_per_driver
    panel = panel.merge(prices[["region", "month", "price_shock", "avg_fuel_price", "baseline_price_region"]],
                        on=["region", "month"], validate="many_to_one")
    if panel.groupby(["region", "usage_group"]).n_drivers.nunique().gt(1).any():
        raise ValueError("Fixed-cohort group counts changed across months")
    national = panel.groupby(["month", "usage_group"], as_index=False)[["n_drivers", "total_liters", "total_fines", "n_positive_buyers"]].sum()
    national["liters_per_driver"] = national.total_liters / national.n_drivers
    national["fines_per_driver"] = national.total_fines / national.n_drivers
    national_base = national.loc[national.month.eq(baseline_month)].set_index("usage_group").liters_per_driver
    national["baseline_liters_per_driver"] = national.usage_group.map(national_base)
    national["liters_index"] = 100 * national.liters_per_driver / national.baseline_liters_per_driver.replace(0, np.nan)
    national["liters_change_percent"] = national.liters_index - 100
    national["liters_change_absolute"] = national.liters_per_driver - national.baseline_liters_per_driver
    return panel, national


def composition(drivers, prices, threshold=0, include_zero=False):
    group_names = GROUPS + ([ZERO] if include_zero else [])
    sample = drivers.loc[drivers.usage_group.isin(group_names)].copy()
    sample["active"] = sample.fuel_liters.gt(0) if threshold == 0 else sample.fuel_liters.ge(threshold)
    counts = sample.groupby(["region", "month", "usage_group"]).active.sum().unstack("usage_group", fill_value=0)
    counts = counts.reindex(columns=group_names, fill_value=0)
    counts.columns = ["active_" + name for name in counts.columns]
    result = counts.reset_index()
    result["n_active_total"] = result[["active_" + name for name in group_names]].sum(axis=1)
    for group in group_names:
        result["share_" + group] = result["active_" + group] / result.n_active_total.replace(0, np.nan)
    result["active_threshold_liters"] = threshold
    result["denominator_population"] = "eligible_including_zero" if include_zero else "positive_baseline_tertiles"
    result = result.merge(prices[["region", "month", "price_shock"]], on=["region", "month"], validate="one_to_one")
    national = result.groupby("month", as_index=False)[["active_" + name for name in group_names] + ["n_active_total"]].sum()
    for group in group_names:
        national["share_" + group] = national["active_" + group] / national.n_active_total.replace(0, np.nan)
    national["active_threshold_liters"] = threshold
    national["denominator_population"] = result.denominator_population.iloc[0]
    return result, national
