"""Ratio of totals is primary; individual ratios are a separate robustness check."""

import numpy as np
import pandas as pd

OUTCOMES = ["liters_per_driver", "fines_per_driver", "fines_per_1000_liters"]


def aggregate_metrics(drivers: pd.DataFrame, prices: pd.DataFrame, by_risk=False):
    keys = ["region", "month"] + (["risk_label"] if by_risk else [])
    data = drivers.assign(has_fine=drivers.fines_count.gt(0),
                          has_fuel=drivers.fuel_liters.gt(0),
                          fine_without_fuel=drivers.fines_count.where(drivers.fuel_liters.eq(0), 0))
    result = data.groupby(keys, as_index=False).agg(
        n_drivers=("client_id", "nunique"), total_fines=("fines_count", "sum"),
        total_liters=("fuel_liters", "sum"), n_fined_drivers=("has_fine", "sum"),
        n_fuel_buyers=("has_fuel", "sum"), fines_without_observed_fuel=("fine_without_fuel", "sum"))
    result["fines_per_driver"] = result.total_fines / result.n_drivers
    result["liters_per_driver"] = result.total_liters / result.n_drivers
    result["fines_per_1000_liters"] = 1000 * result.total_fines / result.total_liters.replace(0, np.nan)
    result["share_fuel_buyers"] = result.n_fuel_buyers / result.n_drivers
    result["intensity_status"] = np.where(result.total_liters.gt(0), "ok", "zero_total_liters")
    return result.merge(prices, on=["region", "month"], how="left", validate="many_to_one")


def attach_risk(drivers: pd.DataFrame, labels: pd.DataFrame, lag=False):
    keys = ["client_id", "region", "month"]
    labels = labels.copy()
    labels["risk_source_month"] = labels.month
    if lag:
        labels["month"] = (pd.PeriodIndex(labels.month, freq="M") + 1).astype(str)
    return drivers.merge(labels, on=keys, how="inner", validate="one_to_one")


def liters_distribution(drivers: pd.DataFrame):
    quantiles = {"min": 0, "p1": .01, "p5": .05, "p10": .1, "median": .5,
                 "p90": .9, "p95": .95, "p99": .99, "max": 1}
    rows = []
    for name, values in [("all_driver_months", drivers.fuel_liters),
                         ("positive_liters_only", drivers.loc[drivers.fuel_liters.gt(0), "fuel_liters"])]:
        rows.append({"population": name, "N": len(values),
                     **{label: values.quantile(q) for label, q in quantiles.items()}})
    return pd.DataFrame(rows)


def individual_ratios(drivers: pd.DataFrame, prices: pd.DataFrame, threshold: float):
    if threshold <= 0:
        raise ValueError("Individual ratio requires a strictly positive liters threshold")
    eligible = drivers.loc[drivers.fuel_liters.ge(threshold)].copy()
    eligible["individual_fines_per_100_liters"] = 100 * eligible.fines_count / eligible.fuel_liters
    return eligible.merge(prices[["region", "month", "price_shock"]], on=["region", "month"],
                          how="left", validate="many_to_one")


def select_regions(prices: pd.DataFrame):
    """Predeclared selection uses only endpoint price growth, never outcomes."""
    endpoint = prices.loc[prices.month.eq(prices.month.max())].sort_values(["price_shock", "region"])
    indexes = [0, len(endpoint) // 2, len(endpoint) - 1]
    selected = endpoint.iloc[indexes][["region", "region_name", "month", "price_shock"]].copy()
    selected["price_growth_group"] = ["weak", "middle", "strong"]
    selected["selection_rule"] = "minimum / upper median rank / maximum final-month price shock; ties by region"
    return selected
