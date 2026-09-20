"""Exactly two feature sets, each fitted independently inside region × month."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = {
    "full": ["fines_count", "fuel_liters", "engine_volume", "horsepower"],
    "no_liters": ["fines_count", "engine_volume", "horsepower"],
}


def fit_cells(monthly: pd.DataFrame, model: str, min_drivers: int = 50,
              min_cluster_size: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_drivers < 2 or min_cluster_size < 1:
        raise ValueError("Minimum cell size must be >= 2 and minimum cluster size >= 1")
    features = FEATURES[model]
    output, diagnostics = [], []
    for (region, month), all_drivers in monthly.groupby(["region", "month"], sort=True):
        part = all_drivers.loc[all_drivers.eligible].sort_values("client_id").copy()
        part["cluster"] = pd.Series(pd.NA, index=part.index, dtype="Int64")
        part["risk_label"] = pd.Series(pd.NA, index=part.index, dtype="string")
        audit = dict(model=model, region=region, month=month, n_input=len(all_drivers),
                     n_drivers=len(part), n_missing_features=len(all_drivers) - len(part))
        status = "ok"
        if len(part) < min_drivers:
            status = "too_few_drivers"
        elif len(part[features].drop_duplicates()) < 2:
            status = "identical_features"
        else:
            matrix = StandardScaler().fit_transform(part[features])
            if not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite feature values in {region}, {month}, {model}")
            with threadpool_limits(limits=1):
                fitted = KMeans(n_clusters=2, random_state=42, n_init=30).fit(matrix)
            part["cluster"] = fitted.labels_
            means = part.groupby("cluster").fines_count.mean()
            counts = part.cluster.value_counts()
            audit.update(inertia=float(fitted.inertia_), min_cluster_n=int(counts.min()))
            if len(means) != 2:
                status = "single_cluster"
            elif np.isclose(means.iloc[0], means.iloc[1], rtol=0, atol=1e-12):
                status = "equal_mean_fines"
            else:
                high = means.idxmax()
                part["risk_label"] = np.where(part.cluster.eq(high), "high_risk", "low_risk")
                low_mean = float(means.drop(high).iloc[0])
                audit.update(mean_fines_high=float(means[high]), mean_fines_low=low_mean,
                             fines_ratio=float(means[high] / low_mean) if low_mean > 0 else np.inf,
                             fines_difference=float(means[high] - low_mean))
                if counts.min() < min_cluster_size:
                    status = "too_small_cluster"
        part["cell_status"] = status
        part["model"] = model
        audit["status"] = status
        output.append(part)
        diagnostics.append(audit)
    if not output:
        raise ValueError("No region-month cells")
    return pd.concat(output, ignore_index=True), pd.DataFrame(diagnostics)
