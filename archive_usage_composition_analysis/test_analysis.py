"""Guard against future-dependent groups, changing denominators and false contrasts."""

import unittest

import numpy as np
import pandas as pd

from .prepare_data import assign_baseline_groups, GROUPS, ZERO, INSUFFICIENT
from .metrics import group_panels, composition
from .regressions import fit


class CompositionTests(unittest.TestCase):
    def baseline(self):
        return pd.DataFrame({"client_id": list("abcdefgh"), "baseline_fuel_liters": [0., 10, 10, 20, 20, 30, 30, 99999],
                             "baseline_observed_days": 30,
                             "subscription_creation_date": pd.to_datetime(["2025-01-01"] * 7 + ["2026-04-02"])})

    def test_zero_is_separate_and_late_registration_cannot_set_cutoffs(self):
        groups, cutoff = assign_baseline_groups(self.baseline())
        self.assertEqual(groups.iloc[0].usage_group, ZERO)
        self.assertEqual(groups.iloc[-1].usage_group, INSUFFICIENT)
        self.assertLess(cutoff["q2_liters"], 30)
        self.assertEqual(groups.usage_group.value_counts().loc[GROUPS].tolist(), [2, 2, 2])

    def test_ties_and_row_order_do_not_change_group_membership(self):
        first, _ = assign_baseline_groups(self.baseline())
        second, _ = assign_baseline_groups(self.baseline().sample(frac=1, random_state=4))
        pd.testing.assert_series_equal(first.set_index("client_id").usage_group.sort_index(),
                                       second.set_index("client_id").usage_group.sort_index())
        self.assertTrue(first.groupby("baseline_fuel_liters").usage_group.nunique().eq(1).all())

    def test_incomplete_baseline_is_not_a_tertile_member(self):
        baseline = self.baseline()
        baseline.loc[1, "baseline_observed_days"] = 29
        groups, _ = assign_baseline_groups(baseline)
        self.assertEqual(groups.iloc[1].usage_group, INSUFFICIENT)

    def drivers(self):
        return pd.DataFrame({"client_id": ["a", "b", "a", "b"], "region": "77", "region_name": "Region",
                             "month": ["2026-04", "2026-04", "2026-05", "2026-05"],
                             "usage_group": "LOW", "fuel_liters": [10., 90., 0., 100.], "fines_count": [0, 1, 1, 0]})

    def prices(self):
        return pd.DataFrame({"region": ["77", "77"], "month": ["2026-04", "2026-05"],
                             "price_shock": [0., .1], "avg_fuel_price": [50., 55.], "baseline_price_region": 50.})

    def test_zero_in_followup_remains_in_fixed_denominator(self):
        panel, _ = group_panels(self.drivers(), self.prices(), "2026-04")
        self.assertEqual(panel.n_drivers.tolist(), [2, 2])
        self.assertEqual(panel.liters_per_driver.tolist(), [50., 50.])
        self.assertEqual(panel.liters_index.tolist(), [100., 100.])
        self.assertEqual(panel.n_positive_buyers.tolist(), [2, 1])

    def test_normalization_is_group_mean_ratio_not_individual_ratio(self):
        source = self.drivers()
        source.loc[source.month.eq("2026-05"), "fuel_liters"] = [20., 80.]
        panel, _ = group_panels(source, self.prices(), "2026-04")
        self.assertAlmostEqual(panel.iloc[1].liters_index, 100.)
        self.assertNotAlmostEqual(panel.iloc[1].liters_index, 100 * ((20 / 10 + 80 / 90) / 2))

    def test_active_composition_explicit_zero_denominator_and_threshold(self):
        source = pd.DataFrame({"client_id": list("abcd"), "region": "77", "month": "2026-05",
                               "usage_group": ["LOW", "MEDIUM", "HIGH", ZERO], "fuel_liters": [10., 20., 30., 40.]})
        prices = self.prices().query("month == '2026-05'")
        main, _ = composition(source, prices)
        expanded, _ = composition(source, prices, include_zero=True)
        sensitivity, _ = composition(source, prices, threshold=20)
        self.assertEqual(main.n_active_total.iloc[0], 3)
        self.assertAlmostEqual(main.share_HIGH.iloc[0], 1 / 3)
        self.assertAlmostEqual(expanded.share_HIGH.iloc[0], 1 / 4)
        self.assertEqual(expanded.n_active_total.iloc[0], 4)
        self.assertEqual(sensitivity.active_LOW.iloc[0], 0)
        self.assertEqual(sensitivity.active_MEDIUM.iloc[0], 1)
        self.assertAlmostEqual(sensitivity.share_HIGH.iloc[0], .5)

    def test_interaction_is_difference_of_slopes_not_difference_of_significance(self):
        rng = np.random.default_rng(42)
        rows = []
        for region in range(6):
            for month in range(5):
                shock = month * .02 + region * .005
                for group, slope in zip(GROUPS, [-2., -5., 1.]):
                    rows.append(dict(region=str(region), month=month, usage_group=group,
                                     price_shock=shock, y=10 + slope * shock + rng.normal(scale=.02)))
        data = pd.DataFrame(rows)
        interaction, joint = fit(data, "y", "interaction", interactions=True)
        estimates = {}
        for group in GROUPS:
            single, _ = fit(data.loc[data.usage_group.eq(group)], "y", group)
            estimates[group] = single.set_index("term").loc["price_shock", "beta"]
        for group in ["MEDIUM", "HIGH"]:
            contrast = interaction.set_index("term").loc["PriceShock_x_" + group]
            self.assertAlmostEqual(contrast.beta, estimates[group] - estimates["LOW"])
            self.assertGreater(contrast.SE, 0)
        self.assertEqual(joint["df_denom"], 5)
        self.assertEqual(joint["df_num"], 2)

    def test_group_mapping_does_not_depend_on_followup_fuel_or_fines(self):
        baseline = self.baseline()
        original, _ = assign_baseline_groups(baseline)
        # The assignment API has no follow-up argument; unrelated outcome columns are ignored.
        changed, _ = assign_baseline_groups(baseline.assign(future_liters=0, future_fines=999999))
        pd.testing.assert_series_equal(original.usage_group, changed.usage_group)


if __name__ == "__main__":
    unittest.main()
