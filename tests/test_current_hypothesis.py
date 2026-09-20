from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from current_hypothesis.define_groups import assign_behavior_clusters
from current_hypothesis.check_hypothesis import METRICS, load_clusters, summarize_cluster_changes
from data_sources import read_csv_detected


def test_zero_history_is_not_chronic_and_missing_car_data_is_not_expensive():
    features = pd.DataFrame(
        {
            "client_id": ["a", "b"],
            "fines_last_12m_total": [0, 0],
            "cars_count": [1, 1],
            "car_value_total": pd.Series([pd.NA, 1000], dtype="Float64"),
            "horsepower_max": pd.Series([pd.NA, 90], dtype="Float64"),
            "first_subscription_date": pd.to_datetime(["2025-01-01", None]),
        }
    )
    result, _ = assign_behavior_clusters(features, pd.DataFrame({"client_id": []}))
    assert result["cluster_id"].tolist() == [0, 0]
    assert not result["tag_chronic"].any()
    assert not result.loc[0, "tag_expensive_fleet"]
    assert not result.loc[0, "tag_powerful_car"]
    assert result["tag_insufficient_observation"].tolist() == [False, True]


def test_analysis_requires_saved_groups():
    with pytest.raises(FileNotFoundError, match="Поведенческие группы"):
        load_clusters(pd.DataFrame({"client_id": ["a"]}), None)


def test_unobserved_group_is_kept_with_missing_effect():
    features = pd.DataFrame(
        {
            "client_id": ["a", "b"],
            "cluster": ["0", "6"],
            "cluster_label": ["Без недавних нарушений", "Опасные нарушения"],
            "eligible_comparison": [True, False],
        }
    )
    for metric in METRICS:
        features[f"{metric}_pre"] = [1.0, 0.0]
        features[f"{metric}_post"] = [2.0, 1.0]
    summary = summarize_cluster_changes(features, min_clients=40, alpha=0.05, min_effect=0.2)
    assert set(summary["cluster"]) == {"0", "6"}
    assert summary.loc[summary["cluster"].eq("6"), "clients_compared"].eq(0).all()
    assert summary["paired_effect_size"].isna().all()
    assert not summary["is_new_pattern"].any()


@pytest.mark.parametrize("separator,decimal", [(";", ","), (",", ".")])
def test_raw_and_cleaned_csv_numbers(tmp_path: Path, separator: str, decimal: str):
    path = tmp_path / "input.csv"
    pd.DataFrame({"client_id": ["a", "b"], "volume": [52.24, -4.5]}).to_csv(
        path, sep=separator, decimal=decimal, index=False
    )
    result = read_csv_detected(path)
    assert np.allclose(result["volume"], [52.24, -4.5])
