import numpy as np
import pandas as pd

from current_hypothesis.price_fuel_fines_model import (
    fit_decoupling_model,
    prepare_client_periods,
)


def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    clients = [f"c{i}" for i in range(200)]
    weeks = pd.date_range("2026-04-06", periods=21, freq="7D")
    rows = []
    for client in clients:
        base_fines = rng.uniform(0.05, 0.25)
        for week in weeks:
            post = week >= pd.Timestamp("2026-06-01")
            liters = max(0, rng.normal(28 if not post else 20, 3))
            price = rng.normal(68 if not post else 72, 0.5)
            rows.append(
                {
                    "client_id": client,
                    "week": week,
                    "exposure_days": 7,
                    "subscription_creation_date": "2025-01-01",
                    "fine_count": max(0, base_fines + rng.normal(0, 0.02)),
                    "fuel_volume_liters": liters,
                    "fuel_cost_rub": liters * price,
                }
            )
    return pd.DataFrame(rows)


def test_decoupling_hypothesis_is_detected_on_constructed_data():
    periods = prepare_client_periods(synthetic_panel())
    result = fit_decoupling_model(periods, equivalence_margin=0.05)
    assert result.diagnostics["all_prespecified_conditions_supported"]
    assert result.tests["supported_at_alpha"].all()


def test_late_subscriber_is_excluded():
    panel = synthetic_panel()
    panel.loc[panel["client_id"].eq("c0"), "subscription_creation_date"] = "2026-05-01"
    periods = prepare_client_periods(panel)
    assert "c0" not in set(periods["client_id"])
