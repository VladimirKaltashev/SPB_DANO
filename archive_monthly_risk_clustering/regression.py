"""Descriptive pooled OLS; inference accounts for repeated observations by region."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .clustering import FEATURES


def add_price_shock(prices: pd.DataFrame, baseline_month: str) -> pd.DataFrame:
    prices = prices.copy()
    baseline = prices.loc[prices.month.eq(baseline_month), ["region", "avg_fuel_price"]].rename(
        columns={"avg_fuel_price": "baseline_price_region"})
    if baseline.empty or baseline.region.duplicated().any():
        raise ValueError(f"Missing/duplicate baseline for {baseline_month}")
    prices = prices.merge(baseline, on="region", how="left", validate="many_to_one")
    prices["baseline_month"] = baseline_month
    prices["price_shock"] = prices.avg_fuel_price / prices.baseline_price_region - 1
    return prices


def fit_regression(region_month: pd.DataFrame, model: str):
    sample = region_month.loc[region_month.status.eq("ok")].dropna(
        subset=["price_shock", "share_high_risk"]).copy()
    if len(sample) < 4 or sample.region.nunique() < 3 or sample.price_shock.nunique() < 2:
        raise ValueError(f"{model}: insufficient cells/regions/price variation for regression")
    X = sm.add_constant(sample[["price_shock"]], has_constant="add")
    ols = sm.OLS(sample.share_high_risk, X).fit()
    robust = ols.get_robustcov_results(
        cov_type="cluster", groups=sample.region, use_correction=True,
        df_correction=True, use_t=True)
    low, high = robust.conf_int()[1]
    row = dict(model=model.upper(), features=" + ".join(FEATURES[model]),
               beta=float(ols.params.price_shock), p_value=float(robust.pvalues[1]),
               R_squared=float(ols.rsquared), N=int(ols.nobs), n_regions=sample.region.nunique(),
               intercept=float(ols.params.const), std_error=float(robust.bse[1]),
               ci95_low=float(low), ci95_high=float(high),
               p_value_iid=float(ols.pvalues.price_shock),
               beta_pp_per_10pct_price=float(ols.params.price_shock * 10),
               inference="region-clustered SE, small-sample correction, t(G-1)",
               specification="share_high_risk ~ 1 + price_shock; unweighted pooled OLS")
    sample["fitted_share_high_risk"] = ols.predict(X)
    sample["residual"] = ols.resid
    return row, sample
