"""Проверяемая очистка исходных таблиц проекта без изменения raw CSV."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


REFUND_MATCH_WINDOW_MINUTES = 60
PRICE_TOLERANCE = 0.01
VOLUME_TOLERANCE = 0.02

PAIR_KEYS = ["client_id", "auto_document_id"]
FINES_HISTORY_COLUMNS = [
    "april_2025_fines", "may_2025_fines", "jun_2025_fines",
    "jul_2025_fines", "aug_2025_fines", "fines_last_6_month",
    "fines_last_12_month", "fines_last_24_month", "fines_last_36_month",
]
VEHICLE_COLUMNS = ["auto_mark", "auto_year", "engine_type", "price", "color"]


def _read_source_csv(path, id_columns):
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8",
        dtype={column: "string" for column in id_columns},
        low_memory=False,
    )


def load_fuel_data(path="fuel_transaction.csv"):
    """Загрузить топливо и явно привести дату и числовые поля."""
    fuel = _read_source_csv(path, ["client_id", "order_id"])
    expected_columns = {
        "order_id", "order_datetime", "order_fuel_volume",
        "order_fuel_price_1liter", "client_id",
    }
    missing = expected_columns.difference(fuel.columns)
    if missing:
        raise ValueError(f"В fuel отсутствуют столбцы: {sorted(missing)}")

    before_missing = fuel[["order_datetime", "order_fuel_volume", "order_fuel_price_1liter"]].isna().sum()
    fuel["order_datetime"] = pd.to_datetime(fuel["order_datetime"], errors="coerce")
    for column in ["order_fuel_volume", "order_fuel_price_1liter"]:
        fuel[column] = pd.to_numeric(fuel[column], errors="coerce")
    after_missing = fuel[["order_datetime", "order_fuel_volume", "order_fuel_price_1liter"]].isna().sum()
    introduced = (after_missing - before_missing).clip(lower=0)
    if introduced.any():
        warnings.warn(f"При парсинге fuel появились NaN: {introduced.to_dict()}")
    fuel.attrs["parse_nan_introduced"] = introduced.to_dict()
    return fuel


def load_clients_data(path="clients_demographics.csv"):
    clients = _read_source_csv(path, PAIR_KEYS)
    before_missing = clients["subscription_creation_date"].isna().sum()
    clients["subscription_creation_date"] = pd.to_datetime(
        clients["subscription_creation_date"], errors="coerce"
    )
    introduced = int(clients["subscription_creation_date"].isna().sum() - before_missing)
    if introduced > 0:
        warnings.warn(f"При парсинге subscription_creation_date появились NaN: {introduced}")
    clients.attrs["parse_nan_introduced"] = {"subscription_creation_date": introduced}
    return clients


def resolve_field_conflict(series):
    """Вернуть единственное известное значение; настоящий конфликт оставить NaN."""
    known = pd.unique(series.dropna())
    if len(known) == 1:
        return known[0], False
    if len(known) == 0:
        return np.nan, False
    return np.nan, True


def resolve_duplicate_pair(group):
    """Свернуть одну пару клиент–автомобиль по правилам каждого поля."""
    row = {key: group.iloc[0][key] for key in PAIR_KEYS}
    conflicts = []
    resolved = []
    unresolved = []
    values_audit = {}

    for column in group.columns:
        if column in PAIR_KEYS or column == "subscription_creation_date":
            continue
        value, conflict = resolve_field_conflict(group[column])
        row[column] = value
        row[f"conflict_{column}"] = int(conflict)
        if conflict:
            conflicts.append(column)
            unresolved.append(column)
            values_audit[column] = [
                value.item() if isinstance(value, np.generic) else value
                for value in pd.unique(group[column].dropna())
            ]
        elif group[column].notna().sum() and group[column].isna().any():
            resolved.append(column)

    dates = group["subscription_creation_date"].dropna()
    row["subscription_creation_date_first"] = dates.min() if len(dates) else pd.NaT
    row["subscription_creation_date_last"] = dates.max() if len(dates) else pd.NaT
    row["subscription_date_conflict"] = int(dates.nunique() > 1)
    # Техническая совместимость: это минимум наблюдавшихся дат, а не доказанная истинная дата.
    row["subscription_creation_date"] = row["subscription_creation_date_first"]
    if row["subscription_date_conflict"]:
        conflicts.append("subscription_creation_date")
        values_audit["subscription_creation_date"] = [str(value) for value in pd.unique(dates)]

    row["source_row_count"] = len(group)
    row["has_duplicate_source_pair"] = int(len(group) > 1)
    row["fines_history_conflict"] = int(any(c in FINES_HISTORY_COLUMNS for c in unresolved))
    row["vehicle_data_conflict"] = int(any(c in VEHICLE_COLUMNS for c in unresolved))
    row["has_any_data_conflict"] = int(bool(unresolved))

    audit = {
        **{key: row[key] for key in PAIR_KEYS},
        "conflicting_columns": ", ".join(conflicts),
        "resolved_columns": ", ".join(resolved),
        "unresolved_columns": ", ".join(unresolved),
        "has_any_data_conflict": row["has_any_data_conflict"],
        "source_values": json.dumps(values_audit, ensure_ascii=False, default=str),
    }
    return row, audit


def resolve_client_auto_duplicates(clients):
    duplicate_mask = clients.duplicated(PAIR_KEYS, keep=False)
    unique = clients.loc[~duplicate_mask].copy()
    unique["subscription_creation_date_first"] = unique["subscription_creation_date"]
    unique["subscription_creation_date_last"] = unique["subscription_creation_date"]
    unique["subscription_date_conflict"] = 0
    unique["source_row_count"] = 1
    unique["has_duplicate_source_pair"] = 0
    unique["fines_history_conflict"] = 0
    unique["vehicle_data_conflict"] = 0
    unique["has_any_data_conflict"] = 0
    for column in clients.columns:
        if column not in PAIR_KEYS + ["subscription_creation_date"]:
            unique[f"conflict_{column}"] = 0

    rows = []
    audits = []
    duplicate_rows = clients.loc[duplicate_mask]
    for _, group in duplicate_rows.groupby(PAIR_KEYS, sort=False, dropna=False):
        row, audit = resolve_duplicate_pair(group)
        rows.append(row)
        audits.append(audit)
    resolved_duplicates = pd.DataFrame(rows)
    clean = pd.concat([unique, resolved_duplicates], ignore_index=True, sort=False)
    clean = clean.sort_values(PAIR_KEYS, kind="stable").reset_index(drop=True)
    audit = pd.DataFrame(audits)
    assert not clean.duplicated(PAIR_KEYS).any()
    return clean, audit


def find_refund_candidates(fuel, negative_index, used_positive_indices, client_positive=None):
    """Найти доступные положительные операции в консервативном окне."""
    negative = fuel.loc[negative_index]
    lower = negative["order_datetime"] - pd.Timedelta(minutes=REFUND_MATCH_WINDOW_MINUTES)
    if client_positive is None:
        client_positive = fuel[
            fuel["client_id"].eq(negative["client_id"]) & fuel["order_fuel_volume"].gt(0)
        ]
    candidates = client_positive[
        client_positive["order_datetime"].between(lower, negative["order_datetime"], inclusive="left")
        & client_positive["order_fuel_price_1liter"].sub(negative["order_fuel_price_1liter"]).abs().le(PRICE_TOLERANCE)
        & client_positive["order_fuel_volume"].add(VOLUME_TOLERANCE).ge(abs(negative["order_fuel_volume"]))
    ]
    if used_positive_indices:
        candidates = candidates[~candidates.index.isin(used_positive_indices)]
    return candidates


def match_refunds(fuel):
    """Сопоставить отрицание только при одном доступном кандидате."""
    fuel = fuel.copy()
    negative_indices = fuel.index[fuel["order_fuel_volume"].lt(0)].tolist()
    negative_indices.sort(key=lambda idx: fuel.at[idx, "order_datetime"])
    positive_by_client = {
        client_id: group.sort_values("order_datetime")
        for client_id, group in fuel[fuel["order_fuel_volume"].gt(0)].groupby("client_id", sort=False)
    }
    used_positive = set()
    matches = []
    unresolved = []

    for negative_index in negative_indices:
        negative = fuel.loc[negative_index]
        client_positive = positive_by_client.get(negative["client_id"], fuel.iloc[0:0])
        candidates = find_refund_candidates(
            fuel, negative_index, used_positive, client_positive=client_positive
        )
        if len(candidates) == 1:
            positive_index = candidates.index[0]
            positive = candidates.iloc[0]
            actual = positive["order_fuel_volume"] + negative["order_fuel_volume"]
            if actual >= -VOLUME_TOLERANCE:
                refund_type = "full_cancellation" if abs(actual) <= VOLUME_TOLERANCE else "partial_refund"
                matches.append({
                    "negative_index": negative_index,
                    "positive_index": positive_index,
                    "negative_order_id": negative["order_id"],
                    "positive_order_id": positive["order_id"],
                    "client_id": negative["client_id"],
                    "positive_datetime": positive["order_datetime"],
                    "negative_datetime": negative["order_datetime"],
                    "time_delta_minutes": (negative["order_datetime"] - positive["order_datetime"]).total_seconds() / 60,
                    "price": positive["order_fuel_price_1liter"],
                    "positive_volume": positive["order_fuel_volume"],
                    "negative_volume": negative["order_fuel_volume"],
                    "actual_volume": max(float(actual), 0.0),
                    "refund_type": refund_type,
                })
                used_positive.add(positive_index)
                continue

        nearby = client_positive[client_positive["order_datetime"].lt(negative["order_datetime"])].copy()
        if len(nearby):
            nearby["delta_minutes"] = (
                negative["order_datetime"] - nearby["order_datetime"]
            ).dt.total_seconds() / 60
            nearest = nearby.nsmallest(1, "delta_minutes").iloc[0]
            nearest_fields = {
                "nearest_positive_order_id": nearest["order_id"],
                "nearest_time_delta_minutes": nearest["delta_minutes"],
                "nearest_positive_volume": nearest["order_fuel_volume"],
                "nearest_positive_price": nearest["order_fuel_price_1liter"],
            }
        else:
            nearest_fields = {
                "nearest_positive_order_id": pd.NA,
                "nearest_time_delta_minutes": np.nan,
                "nearest_positive_volume": np.nan,
                "nearest_positive_price": np.nan,
            }
        unresolved.append({
            "negative_index": negative_index,
            "order_id": negative["order_id"],
            "client_id": negative["client_id"],
            "order_datetime": negative["order_datetime"],
            "order_fuel_volume": negative["order_fuel_volume"],
            "order_fuel_price_1liter": negative["order_fuel_price_1liter"],
            "candidate_count": len(candidates),
            **nearest_fields,
        })

    return pd.DataFrame(matches), pd.DataFrame(unresolved)


def build_fuel_metrics(fuel, matches):
    fuel = fuel.copy()
    fuel["is_negative_volume"] = fuel["order_fuel_volume"].lt(0)
    fuel["refund_match_status"] = np.where(fuel["is_negative_volume"], "unresolved", "not_applicable")
    fuel["matched_order_id"] = pd.Series(pd.NA, index=fuel.index, dtype="string")
    fuel["refund_type"] = np.where(fuel["is_negative_volume"], "unresolved_negative", "none")
    fuel["fuel_volume_positive_only"] = fuel["order_fuel_volume"].clip(lower=0)
    fuel["fuel_volume_signed_net"] = fuel["order_fuel_volume"]
    fuel["physical_fuel_volume_main"] = fuel["fuel_volume_positive_only"]
    fuel["physical_fuel_transaction_main"] = fuel["order_fuel_volume"].gt(0).astype(int)

    for match in matches.to_dict("records"):
        pi, ni = match["positive_index"], match["negative_index"]
        fuel.at[pi, "refund_match_status"] = "matched"
        fuel.at[ni, "refund_match_status"] = "matched"
        fuel.at[pi, "matched_order_id"] = str(match["negative_order_id"])
        fuel.at[ni, "matched_order_id"] = str(match["positive_order_id"])
        fuel.at[pi, "refund_type"] = match["refund_type"]
        fuel.at[ni, "refund_type"] = match["refund_type"]
        fuel.at[pi, "physical_fuel_volume_main"] = (
            0.0 if match["refund_type"] == "full_cancellation" else match["actual_volume"]
        )
        fuel.at[ni, "physical_fuel_volume_main"] = 0.0
        fuel.at[pi, "physical_fuel_transaction_main"] = int(match["refund_type"] == "partial_refund")
        fuel.at[ni, "physical_fuel_transaction_main"] = 0

    fuel["fuel_volume_matched_adjusted"] = fuel["physical_fuel_volume_main"]
    fuel["fuel_cost_rub"] = fuel["physical_fuel_volume_main"] * fuel["order_fuel_price_1liter"]
    return fuel


def clean_fuel_data(fuel):
    matches, unresolved = match_refunds(fuel)
    clean = build_fuel_metrics(fuel, matches)
    return clean, matches.drop(columns=["negative_index", "positive_index"], errors="ignore"), unresolved.drop(columns=["negative_index"], errors="ignore")


def validate_cleaned_data(raw_fuel, fuel_clean, matches, unresolved, clients_clean):
    assert len(fuel_clean) == len(raw_fuel)
    assert fuel_clean["order_id"].is_unique and fuel_clean["order_id"].notna().all()
    assert not matches["negative_order_id"].duplicated().any()
    assert not matches["positive_order_id"].duplicated().any()
    assert fuel_clean["physical_fuel_volume_main"].ge(0).all()
    full_ids = set(matches.loc[matches["refund_type"].eq("full_cancellation"), "positive_order_id"])
    assert fuel_clean.loc[fuel_clean["order_id"].isin(full_ids), "physical_fuel_volume_main"].eq(0).all()
    partial = matches[matches["refund_type"].eq("partial_refund")]
    if len(partial):
        actual = fuel_clean.set_index("order_id").loc[partial["positive_order_id"], "physical_fuel_volume_main"].to_numpy()
        assert np.allclose(actual, partial["positive_volume"] + partial["negative_volume"])
    unresolved_ids = set(unresolved["order_id"])
    unresolved_rows = fuel_clean[fuel_clean["order_id"].isin(unresolved_ids)]
    assert len(unresolved_rows) == len(unresolved)
    assert unresolved_rows["physical_fuel_volume_main"].eq(0).all()
    assert fuel_clean.loc[fuel_clean["is_negative_volume"], "physical_fuel_volume_main"].eq(0).all()
    assert not clients_clean.duplicated(PAIR_KEYS).any()

    known, conflict = resolve_field_conflict(pd.Series([np.nan, "СИНИЙ"]))
    assert known == "СИНИЙ" and not conflict
    unknown, conflict = resolve_field_conflict(pd.Series([2014, 2017]))
    assert pd.isna(unknown) and conflict


def build_cleaning_report(raw_fuel, scoped_fuel, matches, unresolved, raw_clients, clients_clean, resolved_audit):
    neg_raw = raw_fuel[raw_fuel["order_fuel_volume"].lt(0)]
    duplicate_mask = raw_clients.duplicated(PAIR_KEYS, keep=False)
    duplicate_rows = raw_clients[duplicate_mask]
    duplicate_pairs = duplicate_rows[PAIR_KEYS].drop_duplicates()
    conflict_flags = [c for c in clients_clean if c.startswith("conflict_")]
    values_set_nan = int(clients_clean[conflict_flags].sum().sum())

    positive_only = scoped_fuel["fuel_volume_positive_only"].sum()
    adjusted = scoped_fuel["fuel_volume_matched_adjusted"].sum()
    signed = scoped_fuel["fuel_volume_signed_net"].sum()
    pct = lambda value: 100 * (value / positive_only - 1) if positive_only else np.nan

    duplicate_clean = clients_clean[clients_clean["has_duplicate_source_pair"].eq(1)]
    only_subscription = int((
        duplicate_clean["subscription_date_conflict"].eq(1)
        & duplicate_clean["has_any_data_conflict"].eq(0)
    ).sum())
    missing_nonmissing = int(resolved_audit["resolved_columns"].fillna("").ne("").sum())
    fines_conflicts = int(duplicate_clean["fines_history_conflict"].sum())
    vehicle_conflicts = int(duplicate_clean["vehicle_data_conflict"].sum())

    sanity = []
    expected = {"negative operations": (len(neg_raw), 1500), "negative clients": (neg_raw["client_id"].nunique(), 1432),
                "duplicate pairs": (len(duplicate_pairs), 47), "affected clients": (duplicate_pairs["client_id"].nunique(), 46)}
    for label, (actual, wanted) in expected.items():
        sanity.append(f"- {'OK' if actual == wanted else 'WARNING'}: {label}: {actual} (ожидалось {wanted})")
    parse_nan = raw_fuel.attrs.get("parse_nan_introduced", {})

    return f"""# Cleaning report

Диапазон row-level таблицы топлива: {scoped_fuel['order_datetime'].min()} — {scoped_fuel['order_datetime'].max()}.
Аналитическая панель в ноутбуке дополнительно ограничена заданным периодом.
Сырые CSV не изменялись. Пропуски не заполнялись нулями.
Новых NaN при приведении типов: {json.dumps(parse_nan, ensure_ascii=False)}.

## Fuel

| Показатель | Значение |
|---|---:|
| Всего строк в raw | {len(raw_fuel)} |
| Строк в очищенной row-level таблице | {len(scoped_fuel)} |
| Положительных строк | {int(scoped_fuel['order_fuel_volume'].gt(0).sum())} |
| Отрицательных строк | {int(scoped_fuel['order_fuel_volume'].lt(0).sum())} |
| Клиентов с отрицательными строками | {scoped_fuel.loc[scoped_fuel['order_fuel_volume'].lt(0), 'client_id'].nunique()} |
| Сопоставленных отрицательных операций | {len(matches)} |
| Полных отмен | {int(matches['refund_type'].eq('full_cancellation').sum()) if len(matches) else 0} |
| Частичных возвратов | {int(matches['refund_type'].eq('partial_refund').sum()) if len(matches) else 0} |
| Неразрешённых отрицательных операций | {len(unresolved)} |
| Объём positive-only, л | {positive_only:.2f} |
| Объём matched-adjusted, л | {adjusted:.2f} ({pct(adjusted):+.4f}%) |
| Объём signed-net, л | {signed:.2f} ({pct(signed):+.4f}%) |

## Client-auto duplicates

| Показатель | Значение |
|---|---:|
| Строк, входящих в дубли | {len(duplicate_rows)} |
| Повторяющихся пар | {len(duplicate_pairs)} |
| Затронутых клиентов | {duplicate_pairs['client_id'].nunique()} |
| Пар только с разными датами подписки | {only_subscription} |
| Пар с дополнением пропуска известным значением | {missing_nonmissing} |
| Пар с реальным конфликтом истории штрафов | {fines_conflicts} |
| Пар с реальным конфликтом характеристик автомобиля | {vehicle_conflicts} |
| Значений, оставленных NaN из-за настоящего конфликта | {values_set_nan} |

## Sanity checks исходного набора

{chr(10).join(sanity)}

Все программные assertions выполнены при создании отчёта.
"""


def save_cleaning_outputs(output_dir, fuel_clean, matches, unresolved, raw_conflicts, resolved_audit, clients_clean, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fuel_clean.to_csv(output_dir / "fuel_clean.csv", index=False)
    matches.to_csv(output_dir / "fuel_refund_matches.csv", index=False)
    unresolved.to_csv(output_dir / "fuel_negative_unresolved.csv", index=False)
    raw_conflicts.to_csv(output_dir / "client_auto_conflicts.csv", index=False)
    resolved_audit.to_csv(output_dir / "client_auto_conflicts_resolved.csv", index=False)
    clients_clean.to_csv(output_dir / "clients_demographics_clean.csv", index=False)
    # Старое имя сохраняется для совместимости существующего проекта.
    clients_clean.to_csv(output_dir / "clients_clean.csv", index=False)
    (output_dir / "cleaning_report.md").write_text(report, encoding="utf-8")
