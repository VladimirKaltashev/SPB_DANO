from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_sources import export_clustered_weekly_panel, resolve_data_sources
from spb_clustering import build_fuel_features


def _touch_sources(directory: Path, processed: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    names = (
        (
            "clients_demographics_clean.csv",
            "fines_clean.csv",
            "fuel_clean.csv",
        )
        if processed
        else ("clients_demographics.csv", "fines_2026.csv", "fuel_transaction.csv")
    )
    for name in names:
        (directory / name).write_text("client_id\n", encoding="utf-8")


def test_processed_sources_have_priority(tmp_path: Path) -> None:
    _touch_sources(tmp_path, processed=False)
    _touch_sources(tmp_path / "data" / "processed", processed=True)

    sources = resolve_data_sources(None, tmp_path)

    assert sources.mode == "processed"
    assert sources.directory == (tmp_path / "data" / "processed").resolve()
    assert sources.fines == "fines_clean.csv"


def test_explicit_raw_directory_is_respected(tmp_path: Path) -> None:
    explicit = tmp_path / "external_raw"
    _touch_sources(explicit, processed=False)
    _touch_sources(tmp_path / "data" / "processed", processed=True)

    sources = resolve_data_sources(explicit, tmp_path)

    assert sources.mode == "raw"
    assert sources.directory == explicit.resolve()


def test_clean_fuel_uses_physical_volume_after_refund() -> None:
    fuel = pd.DataFrame(
        {
            "client_id": ["c1", "c1"],
            "order_id": ["positive", "refund"],
            "order_datetime": pd.to_datetime(
                ["2026-04-10 10:00:00", "2026-04-10 10:03:00"]
            ),
            "order_fuel_volume": [50.0, -35.0],
            "order_fuel_price_1liter": [100.0, 100.0],
            "physical_fuel_volume_main": [15.0, 0.0],
            "physical_fuel_transaction_main": [1, 0],
            "fuel_cost_rub": [1500.0, 0.0],
            "is_negative_volume": [False, True],
        }
    )

    result = build_fuel_features(fuel)

    assert result.loc["c1", "fuel_liters_pre"] == 15.0
    assert result.loc["c1", "fuel_spend_pre"] == 1500.0
    assert result.loc["c1", "fuel_refund_orders_pre"] == 1
    assert result.loc["c1", "fuel_refund_volume_pre"] == 35.0


def test_weekly_panel_receives_both_cluster_schemes(tmp_path: Path) -> None:
    panel_path = tmp_path / "client_week_panel.csv"
    clusters_path = tmp_path / "clusters.csv"
    behavior_path = tmp_path / "behavior.csv"
    output_path = tmp_path / "client_week_panel_clustered.csv"
    pd.DataFrame(
        {
            "client_id": ["c1", "c1", "c2"],
            "week": ["2026-04-06", "2026-04-13", "2026-04-06"],
            "fine_count": [0, 1, 2],
        }
    ).to_csv(panel_path, index=False)
    pd.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "cluster_id": [2, 4],
            "cluster_label": ["main 2", "main 4"],
        }
    ).to_csv(clusters_path, index=False)
    pd.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "cluster_id": [0, 6],
            "cluster_label": ["safe", "danger"],
        }
    ).to_csv(behavior_path, index=False)

    stats = export_clustered_weekly_panel(
        panel_path,
        clusters_path,
        output_path,
        behavior_clusters_path=behavior_path,
        chunksize=2,
    )
    result = pd.read_csv(output_path)

    assert stats == {
        "rows": 3,
        "unmatched_primary_rows": 0,
        "unmatched_behavior_rows": 0,
    }
    assert result["cluster_id"].tolist() == [2, 2, 4]
    assert result["behavior_cluster_id"].tolist() == [0, 0, 6]
