from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.data_sources import export_clustered_weekly_panel, resolve_data_sources
from pipeline.clean_data import build_weekly_panel
from current_hypothesis.prepare_clients import build_fuel_features


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


def test_weekly_panel_receives_behavior_groups(tmp_path: Path) -> None:
    panel_path = tmp_path / "client_week_panel.csv"
    clusters_path = tmp_path / "clusters.csv"
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
    stats = export_clustered_weekly_panel(
        panel_path,
        clusters_path,
        output_path,
        chunksize=2,
    )
    result = pd.read_csv(output_path)

    assert stats == {
        "rows": 3,
        "unmatched_rows": 0,
    }
    assert result["cluster_id"].tolist() == [2, 2, 4]


def test_weekly_panel_contains_reproducible_v2_fields() -> None:
    clients = pd.DataFrame(
        {
            "client_id": ["c1"],
            "subscription_creation_date": pd.to_datetime(["2026-04-03"]),
            "car_count": [2],
        }
    )
    fines = pd.DataFrame(
        {
            "client_id": ["c1"],
            "bill_id": ["f1"],
            "bill_offence_date": pd.to_datetime(["2026-04-04"]),
            "total_fine_amount": [100_00],
        }
    )
    fuel = pd.DataFrame(
        {
            "client_id": ["c1", "c1"],
            "order_id": ["positive", "refund"],
            "order_datetime": pd.to_datetime(
                ["2026-04-04 10:00:00", "2026-04-04 10:05:00"]
            ),
            "order_fuel_volume": [50.0, -10.0],
            "order_fuel_price_1liter": [70.0, 70.0],
            "physical_fuel_transaction_main": [1, 0],
            "physical_fuel_volume_main": [40.0, 0.0],
            "fuel_volume_positive_only": [50.0, 0.0],
            "fuel_volume_signed_net": [50.0, -10.0],
            "fuel_cost_rub": [2800.0, 0.0],
        }
    )

    panel = build_weekly_panel(
        clients,
        fines,
        fuel,
        start=pd.Timestamp("2026-04-01"),
        end_exclusive=pd.Timestamp("2026-04-08"),
        crisis_start=pd.Timestamp("2026-04-06"),
    )
    row = panel.iloc[0]

    assert row["fuel_all_transaction_count"] == 2
    assert row["fuel_positive_transaction_count"] == 1
    assert row["fuel_volume_physical_teammate"] == 40
    assert row["fuel_price_weighted"] == 70
    assert bool(row["client_has_multiple_autos"])
    assert bool(row["is_after_subscription"])
