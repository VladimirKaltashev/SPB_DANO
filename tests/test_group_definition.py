from __future__ import annotations

import pandas as pd

from current_hypothesis.define_groups import assign_behavior_clusters


def test_behavior_rules_and_priority() -> None:
    clients = [f"c{i}" for i in range(7)]
    features = pd.DataFrame(
        {
            "client_id": clients,
            "fines_last_12m_total": [0, 5, 1, 5, 100, 8, 3],
            "cars_count": [1] * 7,
            "first_subscription_date": pd.to_datetime(["2025-01-01"] * 7),
            "car_value_total": [1_000_000] * 7,
            "horsepower_max": [120] * 7,
        }
    )
    behavior = pd.DataFrame(
        {
            "client_id": clients[2:],
            "fines_pre": [1, 2, 1, 3, 1],
            "fine_amount_pre": [750, 1500, 750, 3000, 5000],
            "fine_active_days_pre": [1, 2, 1, 3, 1],
            "offence_types_pre": [1, 1, 1, 3, 1],
            "fine_regions_pre": [1, 1, 1, 2, 1],
            "fine_cars_pre": [1] * 5,
            "speeding_events_pre": [0, 2, 0, 0, 0],
            "dangerous_events_pre": [0, 0, 0, 0, 1],
            "high_speed_events_pre": [0, 0, 0, 0, 1],
            "red_light_events_pre": [0] * 5,
            "oncoming_events_pre": [0] * 5,
            "speeding_share_pre": [0, 1, 0, 0, 0],
        }
    )

    result, _ = assign_behavior_clusters(features, behavior)

    assert result.set_index("client_id")["cluster_id"].to_dict() == {
        "c0": 0,
        "c1": 1,
        "c2": 2,
        "c3": 3,
        "c4": 4,
        "c5": 5,
        "c6": 6,
    }


def test_dangerous_pattern_has_highest_priority() -> None:
    features = pd.DataFrame(
        {
            "client_id": ["danger"],
            "fines_last_12m_total": [100],
            "cars_count": [2],
            "first_subscription_date": pd.to_datetime(["2025-01-01"]),
        }
    )
    behavior = pd.DataFrame(
        {
            "client_id": ["danger"],
            "fines_pre": [12],
            "fine_amount_pre": [50_000],
            "fine_active_days_pre": [10],
            "offence_types_pre": [4],
            "fine_regions_pre": [3],
            "fine_cars_pre": [2],
            "speeding_events_pre": [10],
            "dangerous_events_pre": [1],
            "high_speed_events_pre": [1],
            "red_light_events_pre": [0],
            "oncoming_events_pre": [0],
            "speeding_share_pre": [10 / 12],
        }
    )

    result, _ = assign_behavior_clusters(features, behavior)

    assert result.loc[0, "cluster_id"] == 6
    assert result.loc[0, "tag_chronic"]
    assert result.loc[0, "tag_diverse"]
    assert result.loc[0, "tag_dangerous"]
