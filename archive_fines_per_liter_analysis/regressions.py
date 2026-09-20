"""OLS with region-clustered inference, matching the archived specification."""

import numpy as np
import statsmodels.api as sm

from .metrics import OUTCOMES


def fit_ols(data, outcome, population="ALL", block="main"):
    sample = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome, "price_shock", "region"]).copy()
    regions = sample.region.nunique()
    if len(sample) < 4 or regions < 3 or sample.price_shock.nunique() < 2:
        raise ValueError(f"{block}/{population}/{outcome}: insufficient observations, regions or price variation")
    X = sm.add_constant(sample[["price_shock"]], has_constant="add")
    ols = sm.OLS(sample[outcome], X).fit()
    robust = ols.get_robustcov_results(cov_type="cluster", groups=sample.region,
                                     use_correction=True, df_correction=True, use_t=True)
    low, high = robust.conf_int()[1]
    row = dict(TEST=f"Price → {outcome} {population}", block=block, population=population,
               outcome=outcome, BETA=float(ols.params.price_shock), SE=float(robust.bse[1]),
               p_value=float(robust.pvalues[1]), R_squared=float(ols.rsquared), N=int(ols.nobs),
               n_regions=int(regions), n_region_months=int(sample.groupby(["region", "month"]).ngroups),
               n_omitted=int(len(data) - len(sample)), ci95_low=float(low), ci95_high=float(high),
               intercept=float(ols.params.const), iid_p_value=float(ols.pvalues.price_shock),
               change_per_10pct_price=float(ols.params.price_shock * .1),
               specification=f"{outcome} ~ 1 + price_shock; unweighted pooled OLS",
               inference="SE clustered by region; small-sample correction; t(G-1)")
    sample["fitted"] = ols.predict(X)
    sample["residual"] = ols.resid
    return row, sample


def fit_three(panel, population="ALL", block="main"):
    rows, samples = [], {}
    for outcome in OUTCOMES:
        row, sample = fit_ols(panel, outcome, population, block)
        rows.append(row)
        samples[outcome] = sample
    return rows, samples
