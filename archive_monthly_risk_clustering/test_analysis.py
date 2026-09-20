"""Methodology checks: cell isolation, labels, calendar boundaries and provenance."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from .clustering import fit_cells
from .prepare_monthly_data import aggregate_calendar_months, parse_engines, reconcile_weekly
from .regression import add_price_shock
from .visualization import summarize_no_liters_regions


def sample_cell(region="77", month="2026-04"):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "client_id": [f"c{i:03}" for i in range(100)], "region": region, "month": month,
        "eligible": True, "fines_count": np.r_[np.zeros(50), np.full(50, 8)],
        "fuel_liters": np.r_[np.full(50, 40), np.full(50, 180)] + rng.uniform(0, 5, 100),
        "engine_volume": np.r_[np.full(50, 1.2), np.full(50, 3.5)],
        "horsepower": np.r_[np.full(50, 80), np.full(50, 280)],
    })


class MethodologyTests(unittest.TestCase):
    def test_region_month_isolation_and_risk_order(self):
        one = sample_cell()
        different_month = sample_cell(month="2026-05")
        different_month["fines_count"] = different_month.fines_count * 100
        different_region = sample_cell(region="78")
        different_region["horsepower"] *= 1000
        for model in ["full", "no_liters"]:
            solo, _ = fit_cells(one, model)
            combined, audit = fit_cells(pd.concat([one, different_month, different_region]), model)
            selected = combined[combined.region.eq("77") & combined.month.eq("2026-04")]
            assert_frame_equal(solo[["client_id", "risk_label"]].reset_index(drop=True),
                               selected[["client_id", "risk_label"]].reset_index(drop=True))
            self.assertEqual(len(audit), 3)
            self.assertTrue(audit.mean_fines_high.gt(audit.mean_fines_low).all())
            self.assertTrue(audit.status.eq("ok").all())

    def test_no_liters_invariant_to_fuel_and_row_order(self):
        source = sample_cell()
        before, _ = fit_cells(source, "no_liters")
        source["fuel_liters"] = np.arange(100) ** 5
        after, _ = fit_cells(source.sample(frac=1, random_state=7), "no_liters")
        assert_frame_equal(before[["client_id", "risk_label"]], after[["client_id", "risk_label"]])

    def test_small_identical_and_equal_fines_cells_not_silently_labeled(self):
        small, audit = fit_cells(sample_cell().iloc[:20], "full")
        self.assertEqual(audit.status.iloc[0], "too_few_drivers")
        self.assertTrue(small.risk_label.isna().all())
        same_fines = sample_cell().assign(fines_count=0)
        result, audit = fit_cells(same_fines, "full")
        self.assertEqual(audit.status.iloc[0], "equal_mean_fines")
        self.assertTrue(result.risk_label.isna().all())
        identical = same_fines.assign(fuel_liters=0, horsepower=100, engine_volume=1.6)
        _, audit = fit_cells(identical, "full")
        self.assertEqual(audit.status.iloc[0], "identical_features")

    def test_calendar_month_boundary_and_zero_drivers(self):
        cohort = pd.DataFrame({"client_id": ["a", "b"], "region": ["77", "77"]})
        dates = pd.to_datetime(["2026-04-30 23:59:59", "2026-05-01 00:00:00"])
        fines = pd.DataFrame({"client_id": ["a", "a"], "bill_offence_date": dates})
        fuel = pd.DataFrame({"client_id": ["a", "a"], "order_datetime": dates,
                             "physical_fuel_volume_main": [20.0, 30.0]})
        result = aggregate_calendar_months(cohort, fines, fuel, pd.period_range("2026-04", "2026-05", freq="M"))
        self.assertEqual(len(result), 4)
        self.assertEqual(result.fines_count.sum(), 2)
        a = result[result.client_id.eq("a")].set_index("month")
        self.assertEqual(a.loc["2026-04", "fuel_liters"], 20)
        self.assertEqual(a.loc["2026-05", "fuel_liters"], 30)
        self.assertTrue(result.loc[result.client_id.eq("b"), "fines_count"].eq(0).all())

    def test_reconciliation_rejects_wrong_clients_even_when_totals_match(self):
        week = pd.Timestamp("2026-04-06")
        panel = pd.DataFrame({"client_id": ["a", "b"], "week": [week, week],
                              "fine_count": [1, 0], "fuel_volume_physical_teammate": [20, 0]})
        fines = pd.DataFrame({"client_id": ["b"], "bill_offence_date": [week]})
        fuel = pd.DataFrame({"client_id": ["a"], "order_datetime": [week], "physical_fuel_volume_main": [20]})
        with self.assertRaisesRegex(ValueError, "does not reproduce v2"):
            reconcile_weekly(panel, fines, fuel)

    def test_engine_parsing_and_missing_not_fabricated(self):
        engines = pd.DataFrame({"engine_type": ["1.5 (150.00 л.с.)", "2,0 (190,0 л.с.)", "Электрический", None]})
        result = parse_engines(engines)
        self.assertEqual(result.engine_volume.iloc[0], 1.5)
        self.assertEqual(result.horsepower.iloc[1], 190)
        self.assertTrue(result.engine_parse_missing.iloc[2:].all())

    def test_region_specific_baseline_and_missing_baseline(self):
        prices = pd.DataFrame({"region": ["77", "77", "78", "78", "02"],
                               "month": ["2026-04", "2026-05", "2026-04", "2026-05", "2026-05"],
                               "avg_fuel_price": [50.0, 55.0, 100.0, 120.0, 70.0]})
        result = add_price_shock(prices, "2026-04")
        np.testing.assert_allclose(result.price_shock.iloc[:4], [0, 0.1, 0, 0.2])
        self.assertTrue(pd.isna(result.price_shock.iloc[4]))

    def test_regional_change_summary_uses_first_and_last_month(self):
        sample = pd.DataFrame({
            "region": ["77", "77", "77"],
            "region_name": ["Москва"] * 3,
            "month": ["2026-05", "2026-04", "2026-06"],
            "share_high_risk": [0.22, 0.20, 0.25],
            "avg_fuel_price": [55.0, 50.0, 60.0],
        })
        result = summarize_no_liters_regions(sample).iloc[0]
        self.assertEqual(result.start_month, "2026-04")
        self.assertEqual(result.end_month, "2026-06")
        self.assertAlmostEqual(result.risk_change_pp, 5.0)
        self.assertAlmostEqual(result.price_change_pct, 20.0)


if __name__ == "__main__":
    unittest.main()
