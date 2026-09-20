import numpy as np
import pandas as pd

from current_hypothesis.within_between_model import (
    fit_within_between,
    prepare_model_frame,
)


def make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    clients = np.repeat([f"c{i}" for i in range(80)], 14)
    week_number = np.tile(np.arange(14), 80)
    stable_fuel = np.repeat(rng.uniform(0.2, 1.6, 80), 14)
    weekly_change = rng.normal(0, 0.12, len(clients))
    fuel_100l = np.clip(stable_fuel + weekly_change, 0, None)
    fines = 0.8 * stable_fuel + 0.1 * weekly_change + rng.normal(0, 0.08, len(clients))
    return pd.DataFrame(
        {
            "client_id": clients,
            "week": pd.Timestamp("2026-04-06") + pd.to_timedelta(week_number * 7, unit="D"),
            "exposure_days": 7,
            "fine_count": np.clip(fines, 0, None),
            "fuel_volume_liters": fuel_100l * 100,
        }
    )


def test_within_between_model_detects_stronger_between_relationship():
    data = prepare_model_frame(make_panel(), min_weeks=12)
    result = fit_within_between(data, include_controls=False)
    test = result.comparison.loc[result.comparison["cluster"].eq("client")].iloc[0]
    assert test["beta_between"] > test["beta_within"]
    assert test["p_value_one_sided_between_greater"] < 0.05
    assert np.allclose(data.groupby("client_id")["fuel_within_100l"].mean(), 0)


def test_event_zeros_are_retained():
    panel = make_panel()
    panel.loc[0, ["fine_count", "fuel_volume_liters"]] = 0
    result = prepare_model_frame(panel, min_weeks=12)
    row = result.loc[(result["client_id"] == "c0") & (result["week"] == panel.loc[0, "week"])]
    assert row["fine_count"].iat[0] == 0
    assert row["fuel_volume_liters"].iat[0] == 0
