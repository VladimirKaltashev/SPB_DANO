"""Частичная проверка гипотезы о реакции групп на рост цены топлива.

Проверяются парные изменения одних и тех же клиентов внутри каждой заранее
определённой группы. Различия изменений между группами и причинный эффект цены
этот модуль пока не оценивает.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import resolve_data_sources
from plotting import plt
from report_gallery import write_gallery

try:
    from scipy.stats import ttest_1samp
except ImportError:  # используется асимптотическая нормальная оценка
    ttest_1samp = None

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # есть встроенные numpy-варианты
    IsolationForest = None


LOGGER = logging.getLogger("fuel_behavior_analysis")
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DEMOGRAPHICS = "clients_demographics.csv"
DEFAULT_FINES = "fines_2026.csv"
DEFAULT_FUEL = "fuel_transaction.csv"
DEFAULT_CRISIS_START = "2026-06-01"
DEFAULT_ANALYSIS_END = "2026-09-01"
OPTIONAL_2025_DETAIL_FILES = ("fines_2025_detail.csv", "fines_2025.csv")
BASELINE_FINE_COLUMNS = [
    "april_2025_fines",
    "may_2025_fines",
    "jun_2025_fines",
    "jul_2025_fines",
    "aug_2025_fines",
    "fines_last_6_month",
    "fines_last_12_month",
    "fines_last_24_month",
    "fines_last_36_month",
]

METRIC_LABELS = {
    "fuel_tx_count_30d": "число заправок за 30 дней",
    "fuel_liters_30d": "объём топлива за 30 дней",
    "fuel_spend_30d": "расходы на топливо за 30 дней",
    "fuel_avg_liters": "средний объём одной заправки",
    "fuel_avg_price": "средняя цена литра",
    "fuel_active_days_30d": "число дней с заправками за 30 дней",
    "fuel_weekend_share": "доля заправок в выходные",
    "fuel_night_share": "доля ночных заправок",
    "fuel_small_fill_share": "доля небольших заправок до 20 л",
    "fuel_large_fill_share": "доля крупных заправок от 50 л",
    "fine_count_30d": "число штрафов за 30 дней",
    "fine_amount_30d": "сумма штрафов за 30 дней, руб.",
    "fine_speeding_share": "доля штрафов за превышение скорости",
    "fine_regions_count": "число регионов со штрафами",
}

METRICS = list(METRIC_LABELS)
ZERO_IF_NO_EVENT = {
    "fuel_tx_count_30d",
    "fuel_liters_30d",
    "fuel_spend_30d",
    "fuel_active_days_30d",
    "fine_count_30d",
    "fine_amount_30d",
    "fine_regions_count",
}


@dataclass(frozen=True)
class AnalysisWindow:
    crisis_start: pd.Timestamp
    fuel_start: pd.Timestamp
    fuel_end: pd.Timestamp
    fines_start: pd.Timestamp
    fines_end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Сравнивает поведение клиентских кластеров до и после начала "
            "топливного кризиса и находит новые устойчивые паттерны."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--source", choices=["auto", "processed", "raw"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--demographics", default=None)
    parser.add_argument("--fines", default=None)
    parser.add_argument(
        "--fines-2025-detail",
        type=Path,
        default=None,
        help=(
            "Необязательный детальный CSV штрафов за 2025 год с client_id, "
            "bill_id, offence_short_statement и bill_offence_date."
        ),
    )
    parser.add_argument("--fuel", default=None)
    parser.add_argument(
        "--clusters-file",
        type=Path,
        default=None,
        help=(
            "CSV с client_id и cluster_id/cluster. По умолчанию используется "
            "outputs/behavior_clustering/tables/client_clusters.csv."
        ),
    )
    parser.add_argument(
        "--cluster-column",
        type=str,
        default=None,
        help="Имя столбца кластера. Без параметра определяется автоматически.",
    )
    parser.add_argument(
        "--crisis-start",
        type=str,
        default=DEFAULT_CRISIS_START,
        help=(
            "Дата YYYY-MM-DD. По умолчанию 2026-06-01. Передайте auto для "
            "оценки даты по динамике цен."
        ),
    )
    parser.add_argument(
        "--analysis-end",
        type=str,
        default=DEFAULT_ANALYSIS_END,
        help="Исключающая правая граница анализа. По умолчанию 2026-09-01.",
    )
    parser.add_argument("--min-clients", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.20)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as stream:
        header = stream.readline()
    return ";" if header.count(";") > header.count(",") else ","


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    frame = pd.read_csv(
        path,
        sep=_detect_separator(path),
        decimal="," if _detect_separator(path) == ";" else ".",
        low_memory=False,
        encoding="utf-8-sig",
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"В {path.name} отсутствуют колонки: {', '.join(missing)}")
    return frame


def load_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = args.data_dir.resolve()
    clients = read_csv(
        data_dir / args.demographics,
        ["client_id", "subscription_creation_date", *BASELINE_FINE_COLUMNS],
    )
    fines = read_csv(
        data_dir / args.fines,
        [
            "client_id",
            "bill_id",
            "offence_short_statement",
            "total_fine_amount",
            "region_name",
            "bill_offence_date",
        ],
    )
    fuel = read_csv(
        data_dir / args.fuel,
        [
            "client_id",
            "order_id",
            "order_datetime",
            "order_fuel_volume",
            "order_fuel_price_1liter",
        ],
    )

    clients["subscription_creation_date"] = pd.to_datetime(
        clients["subscription_creation_date"], errors="coerce"
    )
    fines["bill_offence_date"] = pd.to_datetime(fines["bill_offence_date"], errors="coerce")
    fuel["order_datetime"] = pd.to_datetime(fuel["order_datetime"], errors="coerce")

    for column in BASELINE_FINE_COLUMNS:
        clients[column] = pd.to_numeric(clients[column], errors="coerce")
    fines["total_fine_amount"] = pd.to_numeric(fines["total_fine_amount"], errors="coerce") / 100.0
    fuel["order_fuel_volume"] = pd.to_numeric(fuel["order_fuel_volume"], errors="coerce")
    # В очищенной таблице эта колонка уже учитывает полные и частичные
    # возвраты. Все дальнейшие расчёты используют фактически отпущенный объём.
    if "physical_fuel_volume_main" in fuel:
        fuel["order_fuel_volume"] = pd.to_numeric(
            fuel["physical_fuel_volume_main"], errors="coerce"
        )
    fuel["order_fuel_price_1liter"] = pd.to_numeric(
        fuel["order_fuel_price_1liter"], errors="coerce"
    )

    clients = clients.dropna(subset=["client_id"])
    fines = fines.dropna(subset=["client_id", "bill_offence_date", "total_fine_amount"])
    fuel = fuel.dropna(
        subset=[
            "client_id",
            "order_datetime",
            "order_fuel_volume",
            "order_fuel_price_1liter",
        ]
    )
    fuel = fuel.loc[(fuel["order_fuel_volume"] > 0) & (fuel["order_fuel_price_1liter"] > 0)].copy()
    fines = fines.loc[fines["total_fine_amount"] >= 0].copy()
    return clients, fines, fuel


def load_optional_fines_2025(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame | None, Path | None]:
    """Загружает детализацию 2025 года, если пользователь её предоставил."""
    data_dir = args.data_dir.resolve()
    requested = getattr(args, "fines_2025_detail", None)
    explicit = requested is not None
    candidates: list[Path]
    if explicit:
        requested_path = Path(requested)
        candidates = [requested_path if requested_path.is_absolute() else data_dir / requested_path]
    else:
        candidates = [data_dir / name for name in OPTIONAL_2025_DETAIL_FILES]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        if explicit:
            raise FileNotFoundError(f"Файл детальных штрафов 2025 не найден: {candidates[0]}")
        return None, None

    fines = read_csv(
        path,
        ["client_id", "bill_id", "offence_short_statement", "bill_offence_date"],
    )
    fines["bill_offence_date"] = pd.to_datetime(fines["bill_offence_date"], errors="coerce")
    fines = fines.dropna(
        subset=["client_id", "bill_id", "offence_short_statement", "bill_offence_date"]
    ).copy()
    return fines, path.resolve()


def detect_crisis_date(
    fuel: pd.DataFrame, min_side_days: int = 30
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Find the strongest sustained 14-day price level change.

    The result is a candidate change point, not a causal conclusion. A known date
    should be supplied with --crisis-start.
    """
    daily = (
        fuel.set_index("order_datetime")["order_fuel_price_1liter"]
        .resample("D")
        .median()
        .interpolate(limit_direction="both")
        .rename("median_price")
        .to_frame()
    )
    if len(daily) < 2 * min_side_days + 1:
        raise ValueError("Недостаточно дней, чтобы автоматически определить дату кризиса.")

    window = 14
    daily["price_before"] = (
        daily["median_price"].rolling(window, min_periods=window).median().shift(1)
    )
    daily["price_after"] = (
        daily["median_price"][::-1].rolling(window, min_periods=window).median()[::-1]
    )
    daily["absolute_shift"] = daily["price_after"] - daily["price_before"]
    rolling_mad = (
        daily["median_price"]
        .rolling(28, center=True, min_periods=14)
        .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    )
    daily["score"] = daily["absolute_shift"] / (1.4826 * rolling_mad.clip(lower=0.03))

    earliest = daily.index.min() + pd.Timedelta(days=min_side_days)
    latest = daily.index.max() - pd.Timedelta(days=min_side_days)
    candidates = daily.loc[(daily.index >= earliest) & (daily.index <= latest)]
    if candidates["score"].dropna().empty:
        raise ValueError("Не удалось автоматически определить дату кризиса.")
    rough_date = candidates["score"].idxmax().normalize()
    rough_row = daily.loc[rough_date]
    midpoint = (rough_row["price_before"] + rough_row["price_after"]) / 2.0
    local = daily.loc[
        rough_date - pd.Timedelta(days=window) : rough_date + pd.Timedelta(days=window)
    ]
    if rough_row["absolute_shift"] >= 0:
        crossings = local.index[local["median_price"] >= midpoint]
    else:
        crossings = local.index[local["median_price"] <= midpoint]
    crisis_date = crossings.min().normalize() if len(crossings) else rough_date.normalize()
    return crisis_date, daily.reset_index(names="date")


def validate_crisis_date(
    value: str | None, fuel: pd.DataFrame, fines: pd.DataFrame
) -> tuple[pd.Timestamp, str, pd.DataFrame]:
    if value and value.lower() != "auto":
        crisis = pd.Timestamp(value).normalize()
        source = "manual"
        _, price_daily = detect_crisis_date(fuel)
    else:
        crisis, price_daily = detect_crisis_date(fuel)
        source = "auto_price_change_candidate"

    starts = [
        fuel["order_datetime"].min().normalize(),
        fines["bill_offence_date"].min().normalize(),
    ]
    ends = [
        fuel["order_datetime"].max().normalize(),
        fines["bill_offence_date"].max().normalize(),
    ]
    common_start, common_end = max(starts), min(ends)
    if not (common_start + pd.Timedelta(days=14) <= crisis <= common_end - pd.Timedelta(days=14)):
        raise ValueError(
            f"Дата кризиса {crisis.date()} должна оставлять не менее 14 дней "
            f"до и после неё в общем интервале {common_start.date()}–{common_end.date()}."
        )
    return crisis, source, price_daily


def make_client_baseline(clients: pd.DataFrame) -> pd.DataFrame:
    clients = clients.copy()
    if "price" in clients:
        clients["price"] = pd.to_numeric(clients["price"], errors="coerce")
    if "auto_year" in clients:
        clients["auto_year"] = pd.to_numeric(clients["auto_year"], errors="coerce")

    # Сначала сворачиваем повторения одной пары клиент–автомобиль. Исторические
    # счётчики внутри такой пары нельзя складывать: это одна и та же машина.
    if "auto_document_id" in clients:
        pair_aggregations: dict[str, str] = {
            "subscription_creation_date": "min",
            **{column: "max" for column in BASELINE_FINE_COLUMNS},
        }
        if "price" in clients:
            pair_aggregations["price"] = "median"
        if "auto_year" in clients:
            pair_aggregations["auto_year"] = "median"
        cars = clients.groupby(["client_id", "auto_document_id"], as_index=False, sort=False).agg(
            pair_aggregations
        )

        client_aggregations: dict[str, str] = {
            "subscription_creation_date": "min",
            "auto_document_id": "nunique",
            **{column: "sum" for column in BASELINE_FINE_COLUMNS},
        }
        if "price" in cars:
            client_aggregations["price"] = "median"
        if "auto_year" in cars:
            client_aggregations["auto_year"] = "median"
        baseline = cars.groupby("client_id", as_index=False).agg(client_aggregations)
    else:
        aggregations: dict[str, str] = {
            "subscription_creation_date": "min",
            **{column: "max" for column in BASELINE_FINE_COLUMNS},
        }
        if "price" in clients:
            aggregations["price"] = "median"
        if "auto_year" in clients:
            aggregations["auto_year"] = "median"
        baseline = clients.groupby("client_id", as_index=False).agg(aggregations)

    baseline = baseline.rename(
        columns={
            "auto_document_id": "car_count",
            "price": "car_price_median",
            "auto_year": "car_year_median",
        }
    )
    baseline["fines_apr_aug_2025"] = baseline[
        [
            "april_2025_fines",
            "may_2025_fines",
            "jun_2025_fines",
            "jul_2025_fines",
            "aug_2025_fines",
        ]
    ].sum(axis=1, min_count=1)
    baseline["fines_recent_monthly_2025"] = baseline["fines_last_6_month"] / 6.0
    baseline["fines_long_monthly_2025"] = baseline["fines_last_36_month"] / 36.0
    baseline["fine_recent_vs_long_2025"] = (
        baseline["fines_recent_monthly_2025"] - baseline["fines_long_monthly_2025"]
    )
    return baseline


def load_clusters(
    baseline: pd.DataFrame,
    clusters_file: Path | None,
    cluster_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    if clusters_file is not None:
        path = clusters_file.resolve()
        supplied = read_csv(path, ["client_id"])
        candidates = [
            cluster_column,
            "cluster_id",
            "cluster",
            "group_id",
            "group",
            "segment_id",
            "segment",
        ]
        detected = next(
            (column for column in candidates if column and column in supplied.columns),
            None,
        )
        if detected is None:
            raise ValueError(
                f"В {path.name} не найден столбец кластера. Ожидался один из: "
                "cluster_id, cluster, group_id, group, segment_id, segment."
            )
        keep = ["client_id", detected]
        label_column = next(
            (
                column
                for column in ("cluster_label", "group_label", "segment_label")
                if column in supplied.columns
            ),
            None,
        )
        if label_column:
            keep.append(label_column)
        supplied = supplied[keep].copy().rename(columns={detected: "cluster"})
        if label_column:
            supplied = supplied.rename(columns={label_column: "cluster_label"})
        if supplied["client_id"].duplicated().any():
            raise ValueError("В файле кластеров client_id должен встречаться один раз.")
        result = baseline.merge(supplied, on="client_id", how="left", validate="one_to_one")
        if result["cluster"].isna().any():
            missing = int(result["cluster"].isna().sum())
            raise ValueError(f"В файле кластеров отсутствуют {missing} клиентов.")
        result["cluster"] = result["cluster"].astype(str)
        if "cluster_label" not in result:
            result["cluster_label"] = "Кластер " + result["cluster"]
        else:
            result["cluster_label"] = result["cluster_label"].fillna("Кластер " + result["cluster"])
        result["cluster_id"] = result["cluster"]
        return result, "supplied"

    if "cluster" in baseline.columns and baseline["cluster"].notna().all():
        baseline["cluster"] = baseline["cluster"].astype(str)
        baseline["cluster_id"] = baseline["cluster"]
        if "cluster_label" not in baseline:
            baseline["cluster_label"] = "Кластер " + baseline["cluster"]
        return baseline, "demographics_column"

    raise FileNotFoundError(
        "Поведенческие группы не найдены. Сначала запустите main.py "
        "или передайте --clusters-file. Анализ не создаёт другую схему автоматически."
    )


def period_mask(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (series >= start) & (series < end)


def _rename_period_columns(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{column}_{suffix}" for column in METRICS if column in frame}
    )


def fuel_features(
    fuel: pd.DataFrame,
    all_clients: pd.Index,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    days = max((end - start).total_seconds() / 86400.0, 1.0)
    subset = fuel.loc[period_mask(fuel["order_datetime"], start, end)].copy()
    subset["spend"] = subset["order_fuel_volume"] * subset["order_fuel_price_1liter"]
    subset["price_weight"] = subset["spend"]
    subset["day"] = subset["order_datetime"].dt.normalize()
    subset["is_weekend"] = subset["order_datetime"].dt.dayofweek >= 5
    hour = subset["order_datetime"].dt.hour
    subset["is_night"] = (hour >= 22) | (hour < 6)
    subset["is_small_fill"] = subset["order_fuel_volume"] <= 20
    subset["is_large_fill"] = subset["order_fuel_volume"] >= 50

    if subset.empty:
        features = pd.DataFrame(index=all_clients)
    else:
        features = subset.groupby("client_id").agg(
            fuel_tx_count_30d=("order_id", "nunique"),
            fuel_liters_30d=("order_fuel_volume", "sum"),
            fuel_spend_30d=("spend", "sum"),
            fuel_avg_liters=("order_fuel_volume", "mean"),
            fuel_price_weighted_sum=("price_weight", "sum"),
            fuel_active_days_30d=("day", "nunique"),
            fuel_weekend_share=("is_weekend", "mean"),
            fuel_night_share=("is_night", "mean"),
            fuel_small_fill_share=("is_small_fill", "mean"),
            fuel_large_fill_share=("is_large_fill", "mean"),
        )
        rate_columns = [
            "fuel_tx_count_30d",
            "fuel_liters_30d",
            "fuel_spend_30d",
            "fuel_active_days_30d",
        ]
        features[rate_columns] = features[rate_columns] * (30.0 / days)
        features["fuel_avg_price"] = features["fuel_price_weighted_sum"] / (
            features["fuel_liters_30d"] / (30.0 / days)
        )
        features = features.drop(columns="fuel_price_weighted_sum")

    features = features.reindex(all_clients)
    for column in ZERO_IF_NO_EVENT.intersection(features.columns):
        features[column] = features[column].fillna(0.0)
    return features


def fines_features(
    fines: pd.DataFrame,
    all_clients: pd.Index,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    days = max((end - start).total_seconds() / 86400.0, 1.0)
    subset = fines.loc[period_mask(fines["bill_offence_date"], start, end)].copy()
    subset["is_speeding"] = subset["offence_short_statement"].str.contains(
        r"превыш|скорост", case=False, regex=True, na=False
    )
    if subset.empty:
        features = pd.DataFrame(index=all_clients)
    else:
        features = subset.groupby("client_id").agg(
            fine_count_30d=("bill_id", "nunique"),
            fine_amount_30d=("total_fine_amount", "sum"),
            fine_speeding_share=("is_speeding", "mean"),
            fine_regions_count=("region_name", "nunique"),
        )
        rate_columns = ["fine_count_30d", "fine_amount_30d"]
        features[rate_columns] = features[rate_columns] * (30.0 / days)

    features = features.reindex(all_clients)
    for column in ZERO_IF_NO_EVENT.intersection(features.columns):
        features[column] = features[column].fillna(0.0)
    return features


def build_behavior_features(
    baseline: pd.DataFrame,
    fines: pd.DataFrame,
    fuel: pd.DataFrame,
    crisis: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> tuple[pd.DataFrame, AnalysisWindow]:
    fuel_start = fuel["order_datetime"].min().normalize()
    fuel_end = min(
        fuel["order_datetime"].max().normalize() + pd.Timedelta(days=1),
        analysis_end,
    )
    fines_start = fines["bill_offence_date"].min().normalize()
    fines_end = min(
        fines["bill_offence_date"].max().normalize() + pd.Timedelta(days=1),
        analysis_end,
    )
    if fuel_end <= crisis or fines_end <= crisis:
        raise ValueError("Правая граница анализа должна быть позже даты кризиса.")
    all_clients = pd.Index(baseline["client_id"].unique(), name="client_id")

    pre = fuel_features(fuel, all_clients, fuel_start, crisis).join(
        fines_features(fines, all_clients, fines_start, crisis), how="outer"
    )
    post = fuel_features(fuel, all_clients, crisis, fuel_end).join(
        fines_features(fines, all_clients, crisis, fines_end), how="outer"
    )
    pre = _rename_period_columns(pre, "pre")
    post = _rename_period_columns(post, "post")

    result = baseline.set_index("client_id").join(pre).join(post).reset_index()
    result["eligible_comparison"] = result["subscription_creation_date"].le(
        max(fuel_start, fines_start)
    )
    for metric in METRICS:
        pre_col, post_col = f"{metric}_pre", f"{metric}_post"
        if pre_col not in result:
            result[pre_col] = np.nan
        if post_col not in result:
            result[post_col] = np.nan
        result[f"{metric}_delta"] = result[post_col] - result[pre_col]

    window = AnalysisWindow(
        crisis_start=crisis,
        fuel_start=fuel_start,
        fuel_end=fuel_end,
        fines_start=fines_start,
        fines_end=fines_end,
    )
    return result, window


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().clip(0, 1)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = ordered.to_numpy() * len(ordered) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.clip(adjusted, 0, 1)
    return result


def summarize_cluster_changes(
    features: pd.DataFrame,
    min_clients: int,
    alpha: float,
    min_effect: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = features.loc[features["eligible_comparison"]].copy()
    for cluster, members in features.groupby("cluster", dropna=False):
        group = eligible.loc[eligible["cluster"].eq(cluster)]
        cluster_label = members["cluster_label"].iloc[0]
        for metric in METRICS:
            pre_col, post_col = f"{metric}_pre", f"{metric}_post"
            paired = group[[pre_col, post_col]].dropna()
            difference = paired[post_col] - paired[pre_col]
            n = len(difference)
            if n >= min_clients and difference.std(ddof=1) > 0:
                effect_size = float(difference.mean() / difference.std(ddof=1))
                if ttest_1samp is not None:
                    test = ttest_1samp(difference, popmean=0.0, nan_policy="omit")
                    p_value = float(test.pvalue)
                else:
                    z_value = abs(effect_size) * math.sqrt(n)
                    p_value = float(math.erfc(z_value / math.sqrt(2.0)))
            else:
                p_value = np.nan
                effect_size = np.nan

            pre_mean = float(paired[pre_col].mean()) if n else np.nan
            post_mean = float(paired[post_col].mean()) if n else np.nan
            delta_mean = float(difference.mean()) if n else np.nan
            relative_change = delta_mean / abs(pre_mean) if n and abs(pre_mean) > 1e-12 else np.nan
            rows.append(
                {
                    "cluster": str(cluster),
                    "cluster_id": str(cluster),
                    "cluster_label": cluster_label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "clients_compared": n,
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "mean_change": delta_mean,
                    "relative_change": relative_change,
                    "paired_effect_size": effect_size,
                    "p_value": p_value,
                }
            )

    summary = pd.DataFrame(rows)
    summary["q_value"] = benjamini_hochberg(summary["p_value"])
    summary["is_new_pattern"] = (
        summary["clients_compared"].ge(min_clients)
        & summary["q_value"].le(alpha)
        & summary["paired_effect_size"].abs().ge(min_effect)
    )
    summary["direction"] = np.select(
        [summary["mean_change"] > 0, summary["mean_change"] < 0],
        ["рост", "снижение"],
        default="без изменения",
    )
    return summary.sort_values(
        ["is_new_pattern", "q_value", "paired_effect_size"],
        ascending=[False, True, False],
        na_position="last",
    )


def robust_z_scores(frame: pd.DataFrame) -> pd.DataFrame:
    median = frame.median(axis=0)
    mad = (frame - median).abs().median(axis=0)
    fallback = frame.std(axis=0, ddof=0).replace(0, 1.0)
    scale = (1.4826 * mad).where(mad > 1e-9, fallback).replace(0, 1.0)
    return (frame - median) / scale


def describe_top_changes(row: pd.Series, z_row: pd.Series, top_n: int = 3) -> str:
    candidates = z_row.abs().dropna().sort_values(ascending=False).head(top_n).index
    descriptions: list[str] = []
    for metric in candidates:
        delta = row.get(f"{metric}_delta", np.nan)
        if pd.isna(delta):
            continue
        direction = "выше" if delta > 0 else "ниже"
        descriptions.append(f"{METRIC_LABELS[metric]}: {direction} на {abs(float(delta)):.2f}")
    return "; ".join(descriptions)


def find_client_novelty(features: pd.DataFrame, anomaly_rate: float) -> pd.DataFrame:
    if not 0 < anomaly_rate <= 0.20:
        raise ValueError("--anomaly-rate должен быть больше 0 и не выше 0.20.")
    eligible = features.loc[features["eligible_comparison"]].copy()
    delta_columns = [f"{metric}_delta" for metric in METRICS]
    output_parts: list[pd.DataFrame] = []

    for _, group in eligible.groupby("cluster", dropna=False):
        matrix = group[delta_columns].replace([np.inf, -np.inf], np.nan)
        usable = [column for column in matrix if matrix[column].notna().sum() >= 10]
        if not usable:
            result = group[["client_id", "cluster", "cluster_id", "cluster_label"]].copy()
            result["novelty_score_percentile"] = np.nan
            result["is_novel_client_pattern"] = False
            result["top_changes"] = ""
            output_parts.append(result)
            continue
        matrix = matrix[usable]
        filled = matrix.fillna(matrix.median()).fillna(0.0)
        z = robust_z_scores(filled)

        if len(group) >= 50 and IsolationForest is not None:
            model = IsolationForest(
                n_estimators=250,
                contamination=anomaly_rate,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            raw_score = -model.fit(z).decision_function(z)
        else:
            raw_score = z.abs().max(axis=1).to_numpy()

        percentile = pd.Series(raw_score, index=group.index).rank(pct=True)
        result = group[["client_id", "cluster", "cluster_id", "cluster_label"]].copy()
        result["novelty_score_percentile"] = percentile
        result["is_novel_client_pattern"] = percentile.ge(1.0 - anomaly_rate)

        metric_names = [column.removesuffix("_delta") for column in usable]
        z.columns = metric_names
        result["top_changes"] = [
            describe_top_changes(group.loc[index], z.loc[index]) for index in group.index
        ]
        output_parts.append(result)

    if not output_parts:
        return pd.DataFrame(
            columns=[
                "client_id",
                "cluster",
                "cluster_id",
                "cluster_label",
                "novelty_score_percentile",
                "is_novel_client_pattern",
                "top_changes",
            ]
        )
    return pd.concat(output_parts, ignore_index=True).sort_values(
        "novelty_score_percentile", ascending=False
    )


def make_cluster_profiles(baseline: pd.DataFrame) -> pd.DataFrame:
    profile_columns = [
        "fines_apr_aug_2025",
        "fines_last_6_month",
        "fines_last_12_month",
        "fines_last_36_month",
        "fine_recent_vs_long_2025",
    ]
    available = [column for column in profile_columns if column in baseline]
    profiles = (
        baseline.groupby(["cluster", "cluster_id", "cluster_label"], dropna=False)[available]
        .median()
        .reset_index()
    )
    sizes = baseline.groupby("cluster", dropna=False)["client_id"].nunique().rename("clients")
    return profiles.merge(sizes.reset_index(), on="cluster", how="left")


def summarize_year_over_year_fines(
    baseline: pd.DataFrame,
    fines: pd.DataFrame,
    min_clients: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сравнивает одинаковое окно апрель–август 2025 и 2026 по клиентам."""
    start_2026 = pd.Timestamp("2026-04-01")
    end_2026 = pd.Timestamp("2026-09-01")
    days = (end_2026 - start_2026).days
    counts_2026 = (
        fines.loc[period_mask(fines["bill_offence_date"], start_2026, end_2026)]
        .groupby("client_id")["bill_id"]
        .nunique()
        .rename("fines_apr_aug_2026")
    )
    client_comparison = baseline[
        [
            "client_id",
            "cluster",
            "cluster_id",
            "cluster_label",
            "fines_apr_aug_2025",
        ]
    ].copy()
    client_comparison = client_comparison.join(counts_2026, on="client_id")
    client_comparison["fines_apr_aug_2026"] = client_comparison["fines_apr_aug_2026"].fillna(0)
    client_comparison["fine_count_30d_2025"] = client_comparison["fines_apr_aug_2025"] * 30.0 / days
    client_comparison["fine_count_30d_2026"] = client_comparison["fines_apr_aug_2026"] * 30.0 / days
    client_comparison["fine_count_30d_delta"] = (
        client_comparison["fine_count_30d_2026"] - client_comparison["fine_count_30d_2025"]
    )

    rows = []
    for cluster, group in client_comparison.groupby("cluster", dropna=False):
        difference = group["fine_count_30d_delta"].dropna()
        n = len(difference)
        effect = (
            float(difference.mean() / difference.std(ddof=1))
            if n >= min_clients and difference.std(ddof=1) > 0
            else 0.0
        )
        if n >= min_clients and difference.std(ddof=1) > 0:
            if ttest_1samp is not None:
                p_value = float(ttest_1samp(difference, 0.0).pvalue)
            else:
                p_value = float(math.erfc(abs(effect) * math.sqrt(n) / math.sqrt(2.0)))
        else:
            p_value = np.nan
        rows.append(
            {
                "cluster": str(cluster),
                "cluster_id": str(cluster),
                "cluster_label": group["cluster_label"].iloc[0],
                "clients_compared": n,
                "fine_count_30d_2025": float(group["fine_count_30d_2025"].mean()),
                "fine_count_30d_2026": float(group["fine_count_30d_2026"].mean()),
                "mean_change": float(difference.mean()),
                "paired_effect_size": effect,
                "p_value": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    summary["q_value"] = benjamini_hochberg(summary["p_value"])
    return client_comparison, summary


def _normalize_offence_names(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .fillna("Не указан тип нарушения")
    )


def _aggregate_offence_types(
    fines: pd.DataFrame,
    mapping: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    suffix: str,
) -> pd.DataFrame:
    subset = fines.loc[
        period_mask(fines["bill_offence_date"], start, end),
        ["client_id", "bill_id", "offence_short_statement"],
    ].copy()
    subset["offence_short_statement"] = _normalize_offence_names(subset["offence_short_statement"])
    subset = subset.merge(mapping, on="client_id", how="inner")
    return (
        subset.groupby(
            ["cluster", "cluster_id", "cluster_label", "offence_short_statement"],
            dropna=False,
        )
        .agg(
            **{
                f"fine_count_{suffix}": ("bill_id", "nunique"),
                f"clients_with_offence_{suffix}": ("client_id", "nunique"),
            }
        )
        .reset_index()
    )


def build_offence_type_analysis(
    baseline: pd.DataFrame,
    fines_2026: pd.DataFrame,
    fines_2025: pd.DataFrame | None,
    crisis: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Детализация типов штрафов без предположений о числе кластеров."""
    mapping = baseline[["client_id", "cluster", "cluster_id", "cluster_label"]].drop_duplicates(
        "client_id"
    )
    cluster_sizes = (
        mapping.groupby(["cluster", "cluster_id", "cluster_label"], dropna=False)["client_id"]
        .nunique()
        .rename("cluster_clients")
        .reset_index()
    )

    start_2026 = pd.Timestamp("2026-04-01")
    end_2026 = min(pd.Timestamp("2026-09-01"), analysis_end)
    detail_2026 = _aggregate_offence_types(fines_2026, mapping, start_2026, end_2026, "2026")
    offence_names = set(detail_2026["offence_short_statement"].astype(str))

    detail_2025: pd.DataFrame | None = None
    if fines_2025 is not None:
        detail_2025 = _aggregate_offence_types(
            fines_2025,
            mapping,
            pd.Timestamp("2025-04-01"),
            pd.Timestamp("2025-09-01"),
            "2025",
        )
        offence_names.update(detail_2025["offence_short_statement"].astype(str))

    offences = pd.DataFrame({"offence_short_statement": sorted(offence_names, key=str.casefold)})
    year_table = cluster_sizes.merge(offences, how="cross")
    keys = ["cluster", "cluster_id", "cluster_label", "offence_short_statement"]
    year_table = year_table.merge(detail_2026, on=keys, how="left")
    for column in ("fine_count_2026", "clients_with_offence_2026"):
        year_table[column] = year_table[column].fillna(0).astype(int)

    if detail_2025 is None:
        year_table["fine_count_2025"] = pd.Series(pd.NA, index=year_table.index, dtype="Int64")
        year_table["clients_with_offence_2025"] = pd.Series(
            pd.NA, index=year_table.index, dtype="Int64"
        )
        year_table["detail_2025_available"] = False
    else:
        year_table = year_table.merge(detail_2025, on=keys, how="left")
        for column in ("fine_count_2025", "clients_with_offence_2025"):
            year_table[column] = year_table[column].fillna(0).astype(int)
        year_table["detail_2025_available"] = True

    for year in ("2025", "2026"):
        count = year_table[f"fine_count_{year}"].astype("Float64")
        year_table[f"fines_per_1000_clients_{year}"] = (
            count / year_table["cluster_clients"] * 1000.0
        )
        totals = count.groupby(year_table["cluster"]).transform("sum")
        year_table[f"fine_share_{year}"] = count.div(totals.where(totals > 0))

    overall_2026 = year_table.groupby("offence_short_statement")["fine_count_2026"].transform("sum")
    overall_total_2026 = float(year_table["fine_count_2026"].sum())
    overall_share_2026 = overall_2026 / overall_total_2026 if overall_total_2026 else np.nan
    year_table["overrepresentation_index_2026"] = year_table["fine_share_2026"].div(
        overall_share_2026.replace(0, np.nan)
    )
    year_table["is_overrepresented_2026"] = year_table["fine_count_2026"].ge(10) & year_table[
        "overrepresentation_index_2026"
    ].ge(1.5)
    year_table["rank_in_cluster_2026"] = (
        year_table.groupby("cluster")["fine_count_2026"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    pre_start = fines_2026["bill_offence_date"].min().normalize()
    post_end = min(
        analysis_end,
        fines_2026["bill_offence_date"].max().normalize() + pd.Timedelta(days=1),
    )
    pre_detail = _aggregate_offence_types(fines_2026, mapping, pre_start, crisis, "pre_crisis_2026")
    post_detail = _aggregate_offence_types(
        fines_2026, mapping, crisis, post_end, "post_crisis_2026"
    )
    prepost = year_table[keys + ["cluster_clients"]].merge(pre_detail, on=keys, how="left")
    prepost = prepost.merge(post_detail, on=keys, how="left")
    count_columns = [
        "fine_count_pre_crisis_2026",
        "clients_with_offence_pre_crisis_2026",
        "fine_count_post_crisis_2026",
        "clients_with_offence_post_crisis_2026",
    ]
    prepost[count_columns] = prepost[count_columns].fillna(0).astype(int)
    pre_days = max(1, (crisis - pre_start).days)
    post_days = max(1, (post_end - crisis).days)
    prepost["fines_per_1000_clients_30d_pre"] = (
        prepost["fine_count_pre_crisis_2026"]
        / prepost["cluster_clients"]
        * 1000.0
        * 30.0
        / pre_days
    )
    prepost["fines_per_1000_clients_30d_post"] = (
        prepost["fine_count_post_crisis_2026"]
        / prepost["cluster_clients"]
        * 1000.0
        * 30.0
        / post_days
    )
    prepost["rate_change_per_1000_clients_30d"] = (
        prepost["fines_per_1000_clients_30d_post"] - prepost["fines_per_1000_clients_30d_pre"]
    )
    prepost["rate_ratio_post_vs_pre"] = prepost["fines_per_1000_clients_30d_post"].div(
        prepost["fines_per_1000_clients_30d_pre"].replace(0, np.nan)
    )
    enough_events = (
        prepost["fine_count_pre_crisis_2026"] + prepost["fine_count_post_crisis_2026"]
    ).ge(10)
    material_ratio = prepost["rate_ratio_post_vs_pre"].ge(1.5) | prepost[
        "rate_ratio_post_vs_pre"
    ].le(2.0 / 3.0)
    new_after_crisis = prepost["fine_count_pre_crisis_2026"].eq(0) & prepost[
        "fine_count_post_crisis_2026"
    ].ge(10)
    prepost["is_strong_pre_post_change"] = enough_events & (material_ratio | new_after_crisis)

    anomalies = year_table[
        keys
        + [
            "cluster_clients",
            "fine_count_2026",
            "fines_per_1000_clients_2026",
            "fine_share_2026",
            "overrepresentation_index_2026",
            "is_overrepresented_2026",
        ]
    ].merge(
        prepost[
            keys
            + [
                "fines_per_1000_clients_30d_pre",
                "fines_per_1000_clients_30d_post",
                "rate_change_per_1000_clients_30d",
                "rate_ratio_post_vs_pre",
                "is_strong_pre_post_change",
            ]
        ],
        on=keys,
        how="left",
    )
    anomalies = anomalies.loc[
        anomalies["is_overrepresented_2026"] | anomalies["is_strong_pre_post_change"]
    ].sort_values(
        ["cluster", "is_overrepresented_2026", "fine_count_2026"],
        ascending=[True, False, False],
    )
    return year_table, prepost, anomalies


def build_monthly_cluster_trends(
    baseline: pd.DataFrame,
    fines: pd.DataFrame,
    fuel: pd.DataFrame,
    analysis_end: pd.Timestamp,
) -> pd.DataFrame:
    mapping = baseline[["client_id", "cluster", "cluster_label"]].drop_duplicates()
    sizes = mapping.groupby(["cluster", "cluster_label"])["client_id"].nunique()
    month_start = max(
        fuel["order_datetime"].min().to_period("M"),
        fines["bill_offence_date"].min().to_period("M"),
    )
    month_end = (analysis_end - pd.Timedelta(days=1)).to_period("M")
    months = pd.period_range(month_start, month_end, freq="M")
    cluster_grid = sizes.rename("cluster_clients").reset_index()
    result = cluster_grid.merge(pd.DataFrame({"month": months}), how="cross")

    fuel_monthly = fuel.loc[fuel["order_datetime"].lt(analysis_end)].copy()
    fuel_monthly["month"] = fuel_monthly["order_datetime"].dt.to_period("M")
    fuel_monthly = fuel_monthly.merge(mapping, on="client_id", how="inner")
    fuel_totals = (
        fuel_monthly.groupby(["cluster", "cluster_label", "month"])["order_fuel_volume"]
        .sum()
        .rename("fuel_liters_total")
        .reset_index()
    )

    fines_monthly = fines.loc[fines["bill_offence_date"].lt(analysis_end)].copy()
    fines_monthly["month"] = fines_monthly["bill_offence_date"].dt.to_period("M")
    fines_monthly = fines_monthly.merge(mapping, on="client_id", how="inner")
    fine_totals = (
        fines_monthly.groupby(["cluster", "cluster_label", "month"])["bill_id"]
        .nunique()
        .rename("fine_count_total")
        .reset_index()
    )

    result = result.merge(fuel_totals, on=["cluster", "cluster_label", "month"], how="left").merge(
        fine_totals, on=["cluster", "cluster_label", "month"], how="left"
    )
    result[["fuel_liters_total", "fine_count_total"]] = result[
        ["fuel_liters_total", "fine_count_total"]
    ].fillna(0)
    result["days_in_month"] = result["month"].dt.days_in_month
    result["month_start_ts"] = result["month"].dt.to_timestamp()
    result["month_end_ts"] = result["month_start_ts"] + pd.offsets.MonthBegin(1)
    fuel_observation_start = fuel["order_datetime"].min().normalize()
    fines_observation_start = fines["bill_offence_date"].min().normalize()
    result["fuel_observed_days"] = result.apply(
        lambda row: max(
            1,
            (
                min(row["month_end_ts"], analysis_end)
                - max(row["month_start_ts"], fuel_observation_start)
            ).days,
        ),
        axis=1,
    )
    result["fine_observed_days"] = result.apply(
        lambda row: max(
            1,
            (
                min(row["month_end_ts"], analysis_end)
                - max(row["month_start_ts"], fines_observation_start)
            ).days,
        ),
        axis=1,
    )
    result["fuel_liters_30d_per_client"] = (
        result["fuel_liters_total"]
        / result["cluster_clients"]
        * 30.0
        / result["fuel_observed_days"]
    )
    result["fine_count_30d_per_client"] = (
        result["fine_count_total"] / result["cluster_clients"] * 30.0 / result["fine_observed_days"]
    )
    result["month_start"] = result["month_start_ts"]
    result = result.drop(columns=["month_start_ts", "month_end_ts"])
    return result


def _ordered_cluster_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    def sort_key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    pairs = frame[["cluster", "cluster_label"]].drop_duplicates()
    return pairs.assign(_sort=pairs["cluster"].map(sort_key)).sort_values("_sort")


def _ordered_cluster_labels(frame: pd.DataFrame) -> list[str]:
    return _ordered_cluster_pairs(frame)["cluster_label"].tolist()


def _short_cluster_labels(frame: pd.DataFrame) -> list[str]:
    return [
        f"Кластер {cluster}" for cluster in _ordered_cluster_pairs(frame)["cluster"].astype(str)
    ]


def save_price_plot(price_daily: pd.DataFrame, crisis: pd.Timestamp, output_dir: Path) -> None:
    if plt is None:
        LOGGER.warning("Matplotlib недоступен: график цены пропущен.")
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        price_daily["date"],
        price_daily["median_price"],
        color="#28536B",
        linewidth=1.8,
    )
    ax.axvline(crisis, color="#C44536", linestyle="--", linewidth=1.6)
    ax.set_title("Медианная цена топлива и дата разделения периодов")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена за литр")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "fuel_price_timeline.png", dpi=160)
    plt.close(fig)


def save_cluster_comparison_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return
    metrics = [
        "fuel_liters_30d",
        "fuel_spend_30d",
        "fine_count_30d",
        "fine_amount_30d",
    ]
    available = summary.loc[summary["metric"].isin(metrics)].copy()
    labels = _ordered_cluster_labels(available)
    short_labels = _short_cluster_labels(available)
    figure_width = max(15, 1.35 * len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(figure_width, 10), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics, strict=True):
        part = available.loc[available["metric"] == metric].set_index("cluster_label")
        part = part.reindex(labels)
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, part["pre_mean"], width, label="до кризиса", color="#4C78A8")
        ax.bar(
            x + width / 2,
            part["post_mean"],
            width,
            label="после начала кризиса",
            color="#F2B705",
        )
        ax.set_title(METRIC_LABELS[metric])
        rotation = 35 if len(labels) > 6 else 0
        ax.set_xticks(x, short_labels, rotation=rotation, ha="right" if rotation else "center")
        ax.tick_params(axis="x", labelsize=8 if len(labels) > 8 else 10)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Средние показатели кластеров до и после начала кризиса", fontsize=15)
    fig.savefig(output_dir / "cluster_pre_post_comparison.png", dpi=180)
    plt.close(fig)


def save_effect_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return
    matrix = summary.pivot(
        index="cluster_label", columns="metric_label", values="paired_effect_size"
    )
    matrix = matrix.reindex(_ordered_cluster_labels(summary)).astype(float)
    short_labels = _short_cluster_labels(summary)
    width = max(12, 0.9 * len(matrix.columns))
    height = max(4.5, 0.75 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=-0.8, vmax=0.8)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), short_labels)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            text = "н/д" if pd.isna(value) else f"{value:+.2f}"
            ax.text(column, row, text, ha="center", va="center", fontsize=8)
    ax.set_title("Размер изменения после начала кризиса по кластерам")
    fig.colorbar(image, ax=ax, label="Парный размер эффекта")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_change_heatmap.png", dpi=180)
    plt.close(fig)


def save_monthly_trends_plot(
    trends: pd.DataFrame,
    crisis: pd.Timestamp,
    output_dir: Path,
) -> None:
    if plt is None:
        return
    labels = _ordered_cluster_labels(trends)
    short_labels = _short_cluster_labels(trends)
    colors = plt.get_cmap("tab20", max(len(labels), 1))
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for index, (label, short_label) in enumerate(zip(labels, short_labels, strict=True)):
        part = trends.loc[trends["cluster_label"] == label].sort_values("month_start")
        axes[0].plot(
            part["month_start"],
            part["fuel_liters_30d_per_client"],
            marker="o",
            label=short_label,
            color=colors(index),
        )
        axes[1].plot(
            part["month_start"],
            part["fine_count_30d_per_client"],
            marker="o",
            label=short_label,
            color=colors(index),
        )
    axes[0].set_title("Объём топлива на клиента по месяцам")
    axes[0].set_ylabel("Литров за 30 дней")
    axes[1].set_title("Количество штрафов на клиента по месяцам")
    axes[1].set_ylabel("Штрафов за 30 дней")
    axes[1].set_xlabel("Месяц")
    for ax in axes:
        ax.axvline(crisis, color="#C44536", linestyle="--", linewidth=1.5)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=max(1, min(3, len(labels))), fontsize=8)
    fig.savefig(output_dir / "cluster_monthly_trends.png", dpi=180)
    plt.close(fig)


def save_year_over_year_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return
    ordered = summary.copy()
    labels = _ordered_cluster_labels(ordered)
    short_labels = _short_cluster_labels(ordered)
    ordered = ordered.set_index("cluster_label").reindex(labels)
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        ordered["fine_count_30d_2025"],
        width,
        label="апрель–август 2025",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        ordered["fine_count_30d_2026"],
        width,
        label="апрель–август 2026",
        color="#F2B705",
    )
    ax.set_xticks(x, short_labels)
    ax.set_ylabel("Штрафов на клиента за 30 дней")
    ax.set_title("Сравнение частоты штрафов за одинаковые месяцы 2025 и 2026")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_fines_2025_vs_2026.png", dpi=180)
    plt.close(fig)


def save_offence_type_outputs(
    year_table: pd.DataFrame,
    prepost_table: pd.DataFrame,
    change_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Сохраняет таблицу и единый аналитический график каждого кластера."""
    detail_dir = output_dir / "offence_types"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "cluster_*_offence_types.csv",
        "cluster_*_offence_types.png",
        "cluster_*_analysis.png",
    ):
        for stale in detail_dir.glob(pattern):
            stale.unlink()

    keys = ["cluster", "cluster_id", "cluster_label", "offence_short_statement"]
    combined = year_table.merge(
        prepost_table.drop(columns=["cluster_clients"]), on=keys, how="left"
    )
    manifest_rows: list[dict[str, object]] = []
    for pair in _ordered_cluster_pairs(combined).itertuples(index=False):
        cluster = str(pair.cluster)
        label = str(pair.cluster_label)
        part = combined.loc[combined["cluster"].astype(str).eq(cluster)].copy()
        part["_sort_total"] = part["fine_count_2026"] + part["fine_count_2025"].astype(
            "Float64"
        ).fillna(0)
        part = part.sort_values(
            ["_sort_total", "offence_short_statement"], ascending=[True, False]
        ).drop(columns="_sort_total")
        safe_cluster = re.sub(r"[^0-9A-Za-z_-]+", "_", cluster).strip("_") or "unknown"
        table_name = f"cluster_{safe_cluster}_offence_types.csv"
        chart_name = f"cluster_{safe_cluster}_analysis.png"
        part.to_csv(detail_dir / table_name, index=False, encoding="utf-8-sig")

        if plt is not None:
            count = len(part)
            y = np.arange(count)
            figure_height = max(16.0, 0.43 * count + 7.0)
            fig = plt.figure(figsize=(22, figure_height), constrained_layout=True)
            grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 3.2])
            change_ax = fig.add_subplot(grid[0, 0])
            year_ax = fig.add_subplot(grid[0, 1])
            types_ax = fig.add_subplot(grid[1, 0])
            prepost_ax = fig.add_subplot(grid[1, 1], sharey=types_ax)

            cluster_changes = change_summary.loc[
                change_summary["cluster"].astype(str).eq(cluster)
            ].copy()
            cluster_changes = cluster_changes.reindex(
                cluster_changes["paired_effect_size"].abs().sort_values(ascending=True).index
            )
            change_colors = np.where(
                cluster_changes["paired_effect_size"].ge(0), "#2A9D8F", "#E76F51"
            )
            change_ax.barh(
                cluster_changes["metric_label"],
                cluster_changes["paired_effect_size"],
                color=change_colors,
            )
            change_ax.axvline(0, color="#333333", linewidth=0.8)
            change_ax.set_title("Изменения поведения после начала кризиса")
            change_ax.set_xlabel("Размер изменения: влево — снижение, вправо — рост")
            change_ax.grid(axis="x", alpha=0.2)

            cluster_year = year_summary.loc[year_summary["cluster"].astype(str).eq(cluster)].iloc[0]
            year_bars = year_ax.bar(
                ["апрель–август 2025", "апрель–август 2026"],
                [
                    cluster_year["fine_count_30d_2025"],
                    cluster_year["fine_count_30d_2026"],
                ],
                color=["#4C78A8", "#F2B705"],
                width=0.6,
            )
            year_ax.bar_label(year_bars, fmt="%.2f", padding=3)
            year_ax.set_title("Общая частота штрафов за одинаковые месяцы")
            year_ax.set_ylabel("Штрафов на клиента за 30 дней")
            year_ax.grid(axis="y", alpha=0.2)
            has_2025 = bool(part["detail_2025_available"].iloc[0])
            if has_2025:
                height = 0.38
                types_ax.barh(
                    y - height / 2,
                    part["fine_count_2025"].astype(float),
                    height,
                    label="апрель–август 2025",
                    color="#4C78A8",
                )
                types_ax.barh(
                    y + height / 2,
                    part["fine_count_2026"],
                    height,
                    label="апрель–август 2026",
                    color="#F2B705",
                )
            else:
                colors = np.where(part["is_overrepresented_2026"], "#E76F51", "#F2B705")
                types_ax.barh(y, part["fine_count_2026"], color=colors)
                types_ax.text(
                    0.99,
                    0.01,
                    "За 2025 год типы нарушений отсутствуют в исходных данных",
                    transform=types_ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
            types_ax.set_title("Количество штрафов по типам")
            types_ax.set_xlabel("Штрафов (логарифмическая шкала)")
            types_ax.set_xscale("symlog", linthresh=5)
            if has_2025:
                types_ax.legend()
            else:
                types_ax.text(
                    0.99,
                    0.05,
                    "Красный: доля типа в 1.5+ раза выше, чем по всей выборке",
                    transform=types_ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )

            height = 0.38
            prepost_ax.barh(
                y - height / 2,
                part["fines_per_1000_clients_30d_pre"],
                height,
                label="до кризиса",
                color="#4C78A8",
            )
            prepost_ax.barh(
                y + height / 2,
                part["fines_per_1000_clients_30d_post"],
                height,
                label="после начала кризиса",
                color="#F2B705",
            )
            prepost_ax.set_title("Частота типов штрафов до и после начала кризиса")
            prepost_ax.set_xlabel("Штрафов на 1000 клиентов за 30 дней (логарифмическая шкала)")
            prepost_ax.set_xscale("symlog", linthresh=0.5)
            prepost_ax.legend()
            types_ax.set_yticks(y, part["offence_short_statement"], fontsize=8)
            prepost_ax.tick_params(axis="y", labelleft=False)
            for axis in (types_ax, prepost_ax):
                axis.grid(axis="x", alpha=0.2)
            fig.suptitle(
                f"{label} · {int(part['cluster_clients'].iloc[0]):,} клиентов",
                fontsize=16,
            )
            fig.savefig(detail_dir / chart_name, dpi=170, bbox_inches="tight")
            plt.close(fig)

        manifest_rows.append(
            {
                "cluster": cluster,
                "cluster_id": cluster,
                "cluster_label": label,
                "cluster_clients": int(part["cluster_clients"].iloc[0]),
                "offence_types": int(part["offence_short_statement"].nunique()),
                "table_file": f"offence_types/{table_name}",
                "chart_file": f"offence_types/{chart_name}",
            }
        )
    return pd.DataFrame(manifest_rows)


def save_offence_type_heatmap(year_table: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or year_table.empty:
        return
    matrix = year_table.pivot(
        index="cluster_label",
        columns="offence_short_statement",
        values="overrepresentation_index_2026",
    )
    matrix = matrix.reindex(_ordered_cluster_labels(year_table)).astype(float)
    short_labels = _short_cluster_labels(year_table)
    width = max(18, 0.8 * len(matrix.columns))
    height = max(8, 0.9 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=3)
    short_offence_labels = [
        textwrap.shorten(str(label), width=34, placeholder="…") for label in matrix.columns
    ]
    ax.set_xticks(
        np.arange(len(matrix.columns)),
        short_offence_labels,
        rotation=55,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(matrix.index)), short_labels)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            if pd.notna(value) and value >= 1.5:
                ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title(
        "Относительная представленность типов штрафов в кластерах, 2026\n"
        "Значение выше 1.5 означает заметно большую долю, чем по всей выборке"
    )
    fig.colorbar(image, ax=ax, label="Индекс представленности")
    fig.savefig(output_dir / "cluster_offence_type_heatmap.png", dpi=180)
    plt.close(fig)


def save_change_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        LOGGER.warning("Matplotlib недоступен: график изменений пропущен.")
        return
    significant = summary.loc[summary["is_new_pattern"]].copy()
    if significant.empty:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "При заданных порогах статистически устойчивых изменений не найдено",
            ha="center",
            va="center",
            fontsize=13,
        )
        fig.savefig(output_dir / "cluster_new_patterns.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    significant = significant.reindex(
        significant["paired_effect_size"].abs().sort_values(ascending=False).head(20).index
    )
    labels = [f"{row.cluster_label}: {row.metric_label}" for row in significant.itertuples()]
    colors = np.where(significant["paired_effect_size"] >= 0, "#2A9D8F", "#E76F51")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(significant))))
    ax.barh(
        labels[::-1],
        significant["paired_effect_size"].to_numpy()[::-1],
        color=colors[::-1],
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Размер изменения внутри клиентов")
    ax.set_title("Самые заметные новые паттерны после начала кризиса")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_new_patterns.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def data_quality_report(
    clients: pd.DataFrame,
    fines: pd.DataFrame,
    fuel: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, object]:
    return {
        "rows": {
            "demographics": len(clients),
            "fines": len(fines),
            "fuel_transactions": len(fuel),
        },
        "unique_clients": {
            "demographics": int(clients["client_id"].nunique()),
            "fines": int(fines["client_id"].nunique()),
            "fuel_transactions": int(fuel["client_id"].nunique()),
        },
        "comparison_eligible_clients": int(features["eligible_comparison"].sum()),
        "future_or_late_subscriptions_excluded": int((~features["eligible_comparison"]).sum()),
        "duplicate_full_rows": {
            "demographics": int(clients.duplicated().sum()),
            "fines": int(fines.duplicated().sum()),
            "fuel_transactions": int(fuel.duplicated().sum()),
        },
    }


def write_summary_markdown(
    path: Path,
    summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    offence_year: pd.DataFrame,
    offence_anomalies: pd.DataFrame,
    novelty: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    patterns = summary.loc[summary["is_new_pattern"]].head(15)
    lines = [
        "# Анализ поведения после начала топливного кризиса",
        "",
        f"**Дата разделения:** {metadata['crisis_start']}",
        f"**Источник даты:** {metadata['crisis_date_source']}",
        f"**Метод кластеров:** {metadata['cluster_source']}",
        f"**Число кластеров:** {metadata['cluster_count']}",
        "",
        "[Все графики и расшифровка групп](index.html)",
        "",
        "## Найденные изменения кластеров",
        "",
    ]
    if patterns.empty:
        lines.append("При заданных порогах устойчивых новых паттернов не найдено.")
    else:
        for row in patterns.itertuples():
            near_zero_baseline = abs(row.pre_mean) < 0.05 * max(
                abs(row.post_mean), abs(row.mean_change), 1e-12
            )
            if near_zero_baseline:
                rel = "с почти нулевой базы"
            elif pd.isna(row.relative_change):
                rel = "относительное изменение н/д"
            else:
                rel = f"{row.relative_change:+.1%}"
            lines.append(
                f"- {row.cluster_label}: {row.metric_label} — {row.direction}, "
                f"среднее изменение {row.mean_change:+.2f} ({rel}), "
                f"эффект {row.paired_effect_size:+.2f}, q={row.q_value:.3g}."
            )
    lines.extend(["", "## Сравнение штрафов за 2025 и 2026 годы", ""])
    for row in year_summary.itertuples():
        lines.append(
            f"- {row.cluster_label}: {row.fine_count_30d_2025:.2f} → "
            f"{row.fine_count_30d_2026:.2f} штрафа на клиента за 30 дней; "
            f"изменение {row.mean_change:+.2f}, q={row.q_value:.3g}."
        )
    lines.extend(["", "## Типы штрафов по кластерам", ""])
    if not metadata["offence_type_detail_2025_available"]:
        lines.append(
            "В исходных данных 2025 года нет типа нарушения. "
            "Поэтому по типам показан 2026 год, а для 2025 оставлено "
            "явное отсутствие данных."
        )
        lines.append("")
    for pair in _ordered_cluster_pairs(offence_year).itertuples(index=False):
        cluster_rows = offence_year.loc[offence_year["cluster"].astype(str).eq(str(pair.cluster))]
        top = cluster_rows.nlargest(5, "fine_count_2026")
        total = int(cluster_rows["fine_count_2026"].sum())
        lines.append(f"### {pair.cluster_label}")
        lines.append("")
        lines.append(f"Всего в апреле–августе 2026: **{total}** штрафов.")
        if total:
            top_text = "; ".join(
                f"{row.offence_short_statement} — {int(row.fine_count_2026)} "
                f"({row.fine_share_2026:.1%})"
                for row in top.itertuples()
            )
            lines.append(f"Самые частые типы: {top_text}.")
        cluster_anomalies = offence_anomalies.loc[
            offence_anomalies["cluster"].astype(str).eq(str(pair.cluster))
        ]
        overrepresented = cluster_anomalies.loc[
            cluster_anomalies["is_overrepresented_2026"]
        ].nlargest(5, "overrepresentation_index_2026")
        if not overrepresented.empty:
            anomaly_text = "; ".join(
                f"{row.offence_short_statement} ×{row.overrepresentation_index_2026:.1f}"
                for row in overrepresented.itertuples()
            )
            lines.append("Повышенная доля относительно всей выборки: " + anomaly_text + ".")
        changed = cluster_anomalies.loc[cluster_anomalies["is_strong_pre_post_change"]].copy()
        changed["_abs_change"] = changed["rate_change_per_1000_clients_30d"].abs()
        changed = changed.nlargest(5, "_abs_change")
        if not changed.empty:
            change_text = "; ".join(
                f"{row.offence_short_statement} "
                f"{row.fines_per_1000_clients_30d_pre:.1f}→"
                f"{row.fines_per_1000_clients_30d_post:.1f}"
                for row in changed.itertuples()
            )
            lines.append(
                "Заметные сдвиги до/после кризиса "
                "(штрафов на 1000 клиентов за 30 дней): " + change_text + "."
            )
        lines.append("")
    novel_count = int(novelty["is_novel_client_pattern"].sum()) if not novelty.empty else 0
    lines.extend(
        [
            "",
            "## Нетипичные индивидуальные изменения",
            "",
            f"Отмечено клиентов: **{novel_count}**.",
            "",
            "## Ограничения",
            "",
            "- Автоматическая дата — кандидат по сдвигу цены, а не доказанная причина изменений.",
            "- Статистическая связь не доказывает, что кризис вызвал изменение.",
            "- Кластеры построены по докризисному поведению. Изменения тех же показателей "
            "могут частично отражать отбор и возврат к среднему.",
            "- Детальное сравнение типов штрафов 2025 и 2026 возможно "
            "только при наличии детального файла за 2025 год.",
            "- Флаги клиентов нужны для анализа, а не для автоматических санкций или отказов в обслуживании.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    project_root = PROJECT_ROOT
    sources = resolve_data_sources(
        args.data_dir, project_root, source=getattr(args, "source", "auto")
    )
    args.data_dir = sources.directory
    args.demographics = args.demographics or sources.demographics
    args.fines = args.fines or sources.fines
    args.fuel = args.fuel or sources.fuel
    clusters_file = args.clusters_file or (
        project_root / "outputs" / "behavior_clustering" / "tables" / "client_clusters.csv"
    )
    if not clusters_file.is_file():
        raise FileNotFoundError(
            f"Нет поведенческих групп: {clusters_file}. Сначала запустите main.py."
        )
    LOGGER.info("Загрузка данных для анализа")
    clients, fines, fuel = load_data(args)
    fines_2025_detail, fines_2025_path = load_optional_fines_2025(args)
    crisis, crisis_source, price_daily = validate_crisis_date(args.crisis_start, fuel, fines)
    analysis_end = pd.Timestamp(args.analysis_end).normalize()
    price_daily = price_daily.loc[price_daily["date"].lt(analysis_end)].copy()
    LOGGER.info("Дата разделения периодов: %s (%s)", crisis.date(), crisis_source)

    baseline = make_client_baseline(clients)
    baseline, cluster_source = load_clusters(
        baseline,
        clusters_file,
        cluster_column=args.cluster_column,
    )
    LOGGER.info("Сравнение поведения %s клиентов", len(baseline))
    features, window = build_behavior_features(baseline, fines, fuel, crisis, analysis_end)
    summary = summarize_cluster_changes(features, args.min_clients, args.alpha, args.min_effect)
    novelty = find_client_novelty(features, args.anomaly_rate)
    profiles = make_cluster_profiles(baseline)
    year_clients, year_summary = summarize_year_over_year_fines(baseline, fines, args.min_clients)
    monthly_trends = build_monthly_cluster_trends(baseline, fines, fuel, analysis_end)
    offence_year, offence_prepost, offence_anomalies = build_offence_type_analysis(
        baseline,
        fines,
        fines_2025_detail,
        crisis,
        analysis_end,
    )

    output_dir = (
        args.output_dir or (project_root / "outputs" / "behavior_cluster_analysis")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline.to_csv(output_dir / "client_clusters.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(output_dir / "cluster_profiles_2025.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / "client_behavior_features.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "cluster_new_patterns.csv", index=False, encoding="utf-8-sig")
    year_clients.to_csv(
        output_dir / "client_fines_2025_vs_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    year_summary.to_csv(
        output_dir / "cluster_fines_2025_vs_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly_trends.to_csv(
        output_dir / "cluster_monthly_trends.csv", index=False, encoding="utf-8-sig"
    )
    offence_year.to_csv(
        output_dir / "cluster_offence_types_2025_vs_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    offence_prepost.to_csv(
        output_dir / "cluster_offence_types_pre_post_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    offence_anomalies.to_csv(
        output_dir / "cluster_offence_type_anomalies_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    novelty.to_csv(output_dir / "client_new_patterns.csv", index=False, encoding="utf-8-sig")
    price_daily.to_csv(output_dir / "daily_fuel_price.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("Создание графиков по каждой группе")
    offence_manifest = save_offence_type_outputs(
        offence_year,
        offence_prepost,
        summary,
        year_summary,
        output_dir,
    )
    offence_manifest.to_csv(
        output_dir / "cluster_offence_type_files.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata: dict[str, object] = {
        "crisis_start": str(crisis.date()),
        "crisis_date_source": crisis_source,
        "cluster_source": cluster_source,
        "cluster_count": int(baseline["cluster"].nunique()),
        "cluster_ids": sorted(baseline["cluster"].astype(str).unique().tolist()),
        "source_files": {
            "demographics": str(args.data_dir / args.demographics),
            "fines": str(args.data_dir / args.fines),
            "fuel": str(args.data_dir / args.fuel),
        },
        "offence_type_detail_2025_available": fines_2025_detail is not None,
        "offence_type_detail_2025_source": (
            str(fines_2025_path) if fines_2025_path is not None else None
        ),
        "offence_type_count_2026": int(
            offence_year.loc[
                offence_year["fine_count_2026"].gt(0), "offence_short_statement"
            ].nunique()
        ),
        "fuel_period": [
            str(window.fuel_start.date()),
            str((window.fuel_end - pd.Timedelta(days=1)).date()),
        ],
        "fines_period": [
            str(window.fines_start.date()),
            str((window.fines_end - pd.Timedelta(days=1)).date()),
        ],
        "thresholds": {
            "alpha": args.alpha,
            "min_effect": args.min_effect,
            "min_clients": args.min_clients,
            "anomaly_rate": args.anomaly_rate,
        },
        "quality": data_quality_report(clients, fines, fuel, features),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_markdown(
        output_dir / "analysis_summary.md",
        summary,
        year_summary,
        offence_year,
        offence_anomalies,
        novelty,
        metadata,
    )
    LOGGER.info("Создание общих графиков")
    save_price_plot(price_daily, crisis, output_dir)
    save_offence_type_heatmap(offence_year, output_dir)
    save_cluster_comparison_plot(summary, output_dir)
    save_effect_heatmap(summary, output_dir)
    save_monthly_trends_plot(monthly_trends, crisis, output_dir)
    save_year_over_year_plot(year_summary, output_dir)
    save_change_plot(summary, output_dir)
    chart_files = write_gallery(output_dir, baseline, offence_manifest, summary)
    metadata["chart_files"] = chart_files
    metadata["chart_count"] = len(chart_files)
    metadata["clusters_file"] = str(clusters_file.resolve())
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Проверено графиков: %s. Откройте %s",
        len(chart_files),
        output_dir / "index.html",
    )

    LOGGER.info("Готово. Отчёты: %s", output_dir)
    return output_dir


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
