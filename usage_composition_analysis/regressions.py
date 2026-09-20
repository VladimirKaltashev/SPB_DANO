"""Direct slope contrasts account for covariance across groups in each region."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def design(data, interactions=False, group_effects=False):
    X = pd.DataFrame({"const": 1.}, index=data.index)
    if not group_effects:
        X["price_shock"] = data.price_shock
    if interactions or group_effects:
        for group in ["MEDIUM", "HIGH"]:
            X[group] = data.usage_group.eq(group).astype(float)
            if interactions:
                X["PriceShock_x_" + group] = data.price_shock * X[group]
    return X


def fit(data, outcome, label, interactions=False, group_effects=False):
    data = data.copy().reset_index(drop=True)
    X = design(data, interactions, group_effects)
    if not np.isfinite(X).all().all() or not np.isfinite(data[outcome]).all():
        raise ValueError(f"{label}: nonfinite model input; do not silently drop cells")
    if np.linalg.matrix_rank(X) < len(X.columns) or data.region.nunique() < 3:
        raise ValueError(f"{label}: underidentified model or insufficient region clusters")
    ols = sm.OLS(data[outcome], X).fit()
    robust = ols.get_robustcov_results(cov_type="cluster", groups=data.region, use_correction=True,
                                     df_correction=True, use_t=True)
    rows = []
    for i, term in enumerate(X.columns):
        lower, upper = robust.conf_int()[i]
        rows.append(dict(model=label, outcome=outcome, term=term, beta=float(ols.params.iloc[i]),
                         SE=float(robust.bse[i]), p_value=float(robust.pvalues[i]), R_squared=float(ols.rsquared),
                         N=int(ols.nobs), n_regions=int(data.region.nunique()), ci95_low=float(lower), ci95_high=float(upper),
                         effect_per_10pp_shock=float(ols.params.iloc[i] * .1) if "price" in term.lower() else np.nan))
    joint = None
    if interactions or group_effects:
        terms = ["PriceShock_x_MEDIUM", "PriceShock_x_HIGH"] if interactions else ["MEDIUM", "HIGH"]
        R = np.zeros((2, len(X.columns)))
        for i, term in enumerate(terms):
            R[i, X.columns.get_loc(term)] = 1
        test = robust.f_test(R)
        joint = dict(model=label, hypothesis=" = ".join(terms) + " = 0", F=float(np.asarray(test.fvalue).item()),
                     p_value=float(test.pvalue), df_num=float(test.df_num), df_denom=float(test.df_denom))
    return pd.DataFrame(rows), joint
