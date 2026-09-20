"""Check economic definitions, calendar lagging and clustered inference."""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

from .metrics import aggregate_metrics, attach_risk, individual_ratios, select_regions
from .prepare_data import add_price_shock
from .regressions import fit_ols


class AnalysisTests(unittest.TestCase):
    def prices(self):
        return pd.DataFrame({"region": ["77"], "month": ["2026-04"], "price_shock": [0.]})

    def drivers(self):
        return pd.DataFrame({"client_id": ["a", "b", "c"], "region": "77", "month": "2026-04",
                             "fines_count": [1, 1, 2], "fuel_liters": [10., 90., 0.]})

    def test_ratio_of_sums_includes_zero_purchase_fines_and_all_drivers(self):
        row = aggregate_metrics(self.drivers(), self.prices()).iloc[0]
        self.assertEqual(row.n_drivers, 3)
        self.assertEqual(row.total_fines, 4)
        self.assertEqual(row.total_liters, 100.)
        self.assertEqual(row.fines_per_1000_liters, 40.)
        self.assertEqual(row.n_fuel_buyers, 2)
        self.assertAlmostEqual(row.fines_per_driver, 4 / 3)
        mean_ratio = (1000 * self.drivers().iloc[:2].fines_count / self.drivers().iloc[:2].fuel_liters).mean()
        self.assertNotAlmostEqual(row.fines_per_1000_liters, mean_ratio)

    def test_zero_total_liters_is_missing_not_infinity_or_zero(self):
        row = aggregate_metrics(self.drivers().assign(fuel_liters=0.), self.prices()).iloc[0]
        self.assertTrue(pd.isna(row.fines_per_1000_liters))
        self.assertEqual(row.intensity_status, "zero_total_liters")
        self.assertEqual(row.total_fines, 4)

    def test_current_groups_reconcile_to_full_classified_population(self):
        drivers = self.drivers()
        labels = drivers[["client_id", "region", "month"]].assign(risk_label=["low_risk", "high_risk", "high_risk"])
        groups = aggregate_metrics(attach_risk(drivers, labels), self.prices(), by_risk=True)
        overall = aggregate_metrics(drivers, self.prices()).iloc[0]
        for col in ["n_drivers", "total_fines", "total_liters"]:
            self.assertEqual(groups[col].sum(), overall[col])

    def test_lag_uses_exact_previous_calendar_month_and_previous_label(self):
        drivers = pd.DataFrame({"client_id": ["a"] * 3, "region": ["77"] * 3,
                                "month": ["2026-05", "2026-06", "2026-07"],
                                "fines_count": [100, 200, 300], "fuel_liters": [10, 20, 30]})
        labels = pd.DataFrame({"client_id": ["a", "a"], "region": ["77", "77"],
                               "month": ["2026-04", "2026-06"], "risk_label": ["low_risk", "high_risk"]})
        lagged = attach_risk(drivers, labels, lag=True)
        self.assertEqual(lagged.month.tolist(), ["2026-05", "2026-07"])
        self.assertEqual(lagged.risk_label.tolist(), ["low_risk", "high_risk"])
        self.assertEqual(lagged.fines_count.tolist(), [100, 300])
        self.assertEqual(labels.month.tolist(), ["2026-04", "2026-06"])

    def test_threshold_boundaries_and_individual_unit(self):
        source = self.drivers().assign(fuel_liters=[19.99, 20., 40.])
        result = individual_ratios(source, self.prices(), 20)
        self.assertEqual(result.client_id.tolist(), ["b", "c"])
        np.testing.assert_allclose(result.individual_fines_per_100_liters, [5., 5.])
        with self.assertRaises(ValueError):
            individual_ratios(source, self.prices(), 0)

    def test_regional_price_baseline_is_not_pooled(self):
        prices = pd.DataFrame({"region": ["77", "77", "78", "78"],
                               "month": ["2026-04", "2026-05", "2026-04", "2026-05"],
                               "avg_fuel_price": [50., 55., 100., 120.]})
        result = add_price_shock(prices, "2026-04")
        np.testing.assert_allclose(result.price_shock, [0., .1, 0., .2])

    def test_region_selection_uses_price_only_and_handles_ties_deterministically(self):
        prices = pd.DataFrame({"region": ["02", "03", "04", "05"], "region_name": ["a", "b", "c", "d"],
                               "month": "2026-08", "price_shock": [.02, .05, .05, .15],
                               "fines_per_driver": [1000, -1000, 50, 1]})
        self.assertEqual(select_regions(prices).region.tolist(), ["02", "04", "05"])
        self.assertEqual(select_regions(prices.assign(fines_per_driver=0)).region.tolist(), ["02", "04", "05"])

    def test_clustered_se_against_manual_sandwich_and_t_distribution(self):
        rng = np.random.default_rng(10)
        region = np.repeat(np.arange(6), 5)
        x = rng.uniform(0, .2, 30)
        y = 3 + 2 * x + np.repeat(rng.normal(size=6), 5) + rng.normal(scale=.1, size=30)
        data = pd.DataFrame({"region": region.astype(str), "month": np.tile(np.arange(5), 6),
                             "price_shock": x, "y": y})
        row, _ = fit_ols(data, "y")
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta
        bread = np.linalg.inv(X.T @ X)
        meat = np.zeros((2, 2))
        for g in range(6):
            score = X[region == g].T @ residual[region == g]
            meat += np.outer(score, score)
        covariance = bread @ meat @ bread * (6 / 5) * (29 / 28)
        se = np.sqrt(covariance[1, 1])
        p = 2 * t.sf(abs(beta[1] / se), df=5)
        self.assertAlmostEqual(row["BETA"], beta[1])
        self.assertAlmostEqual(row["SE"], se)
        self.assertAlmostEqual(row["p_value"], p)


if __name__ == "__main__":
    unittest.main()
