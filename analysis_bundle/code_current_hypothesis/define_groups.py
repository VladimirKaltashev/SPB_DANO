"""Правиловая сегментация клиентов по нарушениям ПДД до топливного кризиса.

Сегментация намеренно не использует характеристики автомобиля для назначения
основного кластера. Автомобили и заправки добавляются в
профиль результата и в независимые теги.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data_sources import DataSources, read_csv_detected, resolve_data_sources
from pipeline.plotting import plt
from current_hypothesis.prepare_clients import (
    CRISIS_START,
    PRE_START,
    build_client_features,
    load_sources,
    validate_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLUSTER_LABELS = {
    0: "Без недавних нарушений",
    1: "Без новых штрафов, с историей",
    2: "Эпизодические нарушители",
    3: "Регулярные любители скорости",
    4: "Хронические нарушители",
    5: "Разноплановые нарушители",
    6: "Опасные нарушения",
}

DANGEROUS_PATTERN = r"40-60|60-80|более чем на 80|красный сигнал|встречного движения"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=("auto", "processed", "raw"),
        default="auto",
        help="auto предпочитает очищенные таблицы из data/processed.",
    )
    parser.add_argument(
        "--fines",
        default=None,
        help="Имя файла штрафов; обычно определяется автоматически.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=None,
        help="Необязательная готовая витрина. По умолчанию строится из исходных CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_inputs(
    data_dir: Path,
    features_file: Path,
    fines_file: str = "fines_2026.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fines_path = data_dir / fines_file
    if not fines_path.exists():
        raise FileNotFoundError(f"Не найден файл штрафов: {fines_path}")
    if not features_file.exists():
        raise FileNotFoundError(
            f"Не найдена докризисная витрина: {features_file}. "
            "Уберите --features-file, чтобы построить витрину из исходных CSV."
        )

    fines = read_csv_detected(fines_path)
    features = read_csv_detected(features_file)
    required_fines = {
        "client_id",
        "bill_id",
        "offence_short_statement",
        "total_fine_amount",
        "region_name",
        "bill_offence_date",
        "auto_document_id",
    }
    required_features = {
        "client_id",
        "fines_last_12m_total",
        "cars_count",
        "first_subscription_date",
    }
    for name, frame, required in [
        ("fines_2026", fines, required_fines),
        ("client_features_precrisis", features, required_features),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"В {name} отсутствуют столбцы: {missing}")

    fines["bill_offence_date"] = pd.to_datetime(fines["bill_offence_date"], errors="coerce")
    features["first_subscription_date"] = pd.to_datetime(
        features["first_subscription_date"], errors="coerce"
    )
    return fines, features


def build_fine_behavior(fines: pd.DataFrame) -> pd.DataFrame:
    """Агрегировать нарушения в доступном докризисном окне."""
    pre = fines.loc[
        fines["bill_offence_date"].ge(PRE_START) & fines["bill_offence_date"].lt(CRISIS_START)
    ].copy()
    statement = pre["offence_short_statement"].fillna("").astype(str)
    pre["is_speeding"] = statement.str.contains("Превышение скорости", case=False, regex=False)
    pre["is_dangerous"] = statement.str.contains(DANGEROUS_PATTERN, case=False, regex=True)
    pre["is_high_speed"] = statement.str.contains(
        r"40-60|60-80|более чем на 80", case=False, regex=True
    )
    pre["is_red_light"] = statement.str.contains("красный сигнал", case=False, regex=False)
    pre["is_oncoming"] = statement.str.contains("встречного движения", case=False, regex=False)
    pre["fine_amount_rub"] = (
        pd.to_numeric(pre["total_fine_amount"], errors="coerce").fillna(0) / 100
    )

    result = pre.groupby("client_id", sort=False).agg(
        fines_pre=("bill_id", "nunique"),
        fine_amount_pre=("fine_amount_rub", "sum"),
        fine_active_days_pre=("bill_offence_date", lambda x: x.dt.date.nunique()),
        offence_types_pre=("offence_short_statement", "nunique"),
        fine_regions_pre=("region_name", "nunique"),
        fine_cars_pre=("auto_document_id", "nunique"),
        speeding_events_pre=("is_speeding", "sum"),
        dangerous_events_pre=("is_dangerous", "sum"),
        high_speed_events_pre=("is_high_speed", "sum"),
        red_light_events_pre=("is_red_light", "sum"),
        oncoming_events_pre=("is_oncoming", "sum"),
    )
    result["speeding_share_pre"] = (result["speeding_events_pre"] / result["fines_pre"]).fillna(0)
    return result.reset_index()


def assign_behavior_clusters(
    features: pd.DataFrame, fine_behavior: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    data = features.merge(
        fine_behavior,
        on="client_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_events"),
    )
    behavior_columns = [
        "fines_pre",
        "fine_amount_pre",
        "fine_active_days_pre",
        "offence_types_pre",
        "fine_regions_pre",
        "fine_cars_pre",
        "speeding_events_pre",
        "dangerous_events_pre",
        "high_speed_events_pre",
        "red_light_events_pre",
        "oncoming_events_pre",
        "speeding_share_pre",
    ]
    for column in behavior_columns:
        event_column = f"{column}_events"
        if event_column in data:
            data[column] = data[event_column]
            data = data.drop(columns=event_column)
        if column not in data:
            data[column] = 0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

    history = pd.to_numeric(data["fines_last_12m_total"], errors="coerce").fillna(0)
    chronic_history_threshold = float(history.quantile(0.90, interpolation="higher"))
    positive_amounts = data.loc[data["fine_amount_pre"] > 0, "fine_amount_pre"]
    high_amount_threshold = (
        float(positive_amounts.quantile(0.95, interpolation="higher"))
        if not positive_amounts.empty
        else 0.0
    )

    no_recent = data["fines_pre"].eq(0)
    cluster = np.full(len(data), 2, dtype="int8")
    cluster[no_recent & history.gt(2)] = 1
    cluster[no_recent & history.le(2)] = 0

    speed_regular = data["fines_pre"].ge(2) & data["speeding_share_pre"].ge(0.75)
    chronic = (history.gt(0) & history.ge(chronic_history_threshold)) | data["fines_pre"].ge(10)
    diverse = data["offence_types_pre"].ge(3)
    dangerous = data["dangerous_events_pre"].ge(1)

    # Приоритет отражает наиболее содержательный рисунок поведения. Все
    # пересекающиеся признаки дополнительно сохраняются как независимые теги.
    cluster[speed_regular] = 3
    cluster[chronic] = 4
    cluster[diverse] = 5
    cluster[dangerous] = 6
    data["cluster_id"] = cluster
    data["cluster_label"] = data["cluster_id"].map(CLUSTER_LABELS)
    data["behavior_cluster_id"] = data["cluster_id"]
    data["behavior_cluster_label"] = data["cluster_label"]

    car_value_q75 = (
        float(data["car_value_total"].quantile(0.75)) if "car_value_total" in data else 0
    )
    horsepower_q75 = float(data["horsepower_max"].quantile(0.75)) if "horsepower_max" in data else 0
    data["tag_speed_dominant"] = data["speeding_share_pre"].ge(0.75) & data["fines_pre"].gt(0)
    data["tag_chronic"] = chronic
    data["tag_diverse"] = diverse
    data["tag_dangerous"] = dangerous
    data["tag_multi_region"] = data["fine_regions_pre"].ge(3)
    data["tag_high_fine_amount"] = data["fine_amount_pre"].ge(high_amount_threshold) & data[
        "fine_amount_pre"
    ].gt(0)
    data["tag_multi_car"] = data["cars_count"].gt(1)
    data["tag_expensive_fleet"] = data.get(
        "car_value_total", pd.Series(np.nan, index=data.index)
    ).ge(car_value_q75)
    data["tag_powerful_car"] = data.get("horsepower_max", pd.Series(np.nan, index=data.index)).ge(
        horsepower_q75
    )
    data["tag_insufficient_observation"] = data["first_subscription_date"].isna() | data[
        "first_subscription_date"
    ].ge(PRE_START)

    tag_columns = [column for column in data.columns if column.startswith("tag_")]
    data[tag_columns] = data[tag_columns].fillna(False).astype(bool)
    tag_names = {column: column.removeprefix("tag_") for column in tag_columns}
    data["behavior_tags"] = data[tag_columns].apply(
        lambda row: ";".join(tag_names[col] for col in tag_columns if bool(row[col])),
        axis=1,
    )
    thresholds = {
        "chronic_history_12m": chronic_history_threshold,
        "chronic_pre_window": 10.0,
        "high_fine_amount_pre": high_amount_threshold,
        "car_value_q75": car_value_q75,
        "horsepower_q75": horsepower_q75,
    }
    return data, thresholds


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["cluster_id", "cluster_label"], observed=True)
        .agg(
            clients=("client_id", "nunique"),
            fines_pre_mean=("fines_pre", "mean"),
            fines_pre_median=("fines_pre", "median"),
            historical_fines_12m_median=("fines_last_12m_total", "median"),
            fine_amount_pre_median=("fine_amount_pre", "median"),
            speeding_share_pre_mean=("speeding_share_pre", "mean"),
            dangerous_client_share=("tag_dangerous", "mean"),
            offence_types_pre_median=("offence_types_pre", "median"),
            fine_regions_pre_median=("fine_regions_pre", "median"),
            cars_count_mean=("cars_count", "mean"),
            insufficient_observation_share=("tag_insufficient_observation", "mean"),
        )
        .reset_index()
    )
    optional = {
        "car_value_total": "car_value_total_median",
        "horsepower_max": "horsepower_max_median",
        "fuel_liters_per_week_pre": "fuel_liters_per_week_pre_median",
    }
    for source, target in optional.items():
        if source in data:
            values = data.groupby("cluster_id", observed=True)[source].median()
            summary[target] = summary["cluster_id"].map(values)
    summary["share_pct"] = summary["clients"] / summary["clients"].sum() * 100
    return summary.sort_values("cluster_id").reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = {
        "cluster_id": "ID",
        "cluster_label": "Кластер",
        "clients": "Клиентов",
        "share_pct": "Доля",
        "fines_pre_median": "Медиана штрафов до кризиса",
        "historical_fines_12m_median": "Медиана за 12 мес.",
        "dangerous_client_share": "С опасным нарушением",
        "car_value_total_median": "Медиана стоимости автопарка",
    }
    lines = [
        "| " + " | ".join(labels.get(c, c) for c in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if column == "share_pct":
                values.append(f"{value:.1f}%")
            elif column == "dangerous_client_share":
                values.append(f"{value * 100:.1f}%")
            elif column == "car_value_total_median":
                values.append(f"{value:,.0f} ₽".replace(",", " "))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.1f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    thresholds: dict[str, float],
    output_dir: Path,
) -> None:
    table_columns = [
        "cluster_id",
        "cluster_label",
        "clients",
        "share_pct",
        "fines_pre_median",
        "historical_fines_12m_median",
        "dangerous_client_share",
    ]
    if "car_value_total_median" in summary:
        table_columns.append("car_value_total_median")
    partial = int(data["tag_insufficient_observation"].sum())
    after_cutoff = int(data["first_subscription_date"].ge(CRISIS_START).sum())
    lines = [
        "# Новые поведенческие кластеры водителей",
        "",
        "Кластеры рассчитаны только по информации, доступной до 1 июня 2026 года. "
        "Характеристики автомобилей не влияют на назначение основной группы: они "
        "используются только для профиля и дополнительных тегов.",
        "",
        *markdown_table(summary, table_columns),
        "",
        "## Правила назначения",
        "",
        "Правила применяются снизу вверх; более рискованный или содержательный рисунок "
        "поведения имеет приоритет. Пересечения не теряются и записываются в теги.",
        "",
        "1. **Без недавних нарушений** — нет штрафов с 20 марта по 31 мая, а за 12 месяцев было не более двух.",
        "2. **Без новых штрафов, с историей** — в контрольном окне штрафов нет, но за 12 месяцев их больше двух.",
        "3. **Эпизодические нарушители** — есть штрафы, но не выполнены более сильные правила ниже.",
        "4. **Регулярные любители скорости** — не менее двух штрафов, из которых минимум 75% связаны со скоростью.",
        f"5. **Хронические нарушители** — не менее {thresholds['chronic_history_12m']:.0f} штрафов за 12 месяцев (верхние 10% базы) либо 10 и более в контрольном окне.",
        "6. **Разноплановые нарушители** — не менее трёх разных типов нарушений.",
        "7. **Опасные нарушения** — хотя бы один эпизод: превышение на 40+ км/ч, красный сигнал или встречная полоса.",
        "",
        "## Дополнительные теги",
        "",
        "Для каждого клиента сохранены независимые признаки: доминирование скорости, "
        "хроническая активность, разные типы и регионы нарушений, высокая сумма штрафов, "
        "несколько автомобилей, дорогой автопарк, мощный автомобиль и недостаточное "
        "докризисное наблюдение.",
        "",
        "## Ограничения",
        "",
        "- В таблицах нет статуса оплаты, поэтому выделять «злостных неплательщиков» нельзя. Вместо этого используется высокая штрафная нагрузка.",
        f"- У {partial:,} клиентов подписка началась не раньше 20 марта 2026 года; у {after_cutoff:,} — уже после отсечки. Их кластер нужно трактовать осторожно, для этого есть тег `insufficient_observation`.".replace(
            ",", " "
        ),
        "- Название «без недавних нарушений» не означает абсолютную законопослушность: допускается до двух исторических штрафов.",
        "- Автомобильные признаки полезны для описания групп, но намеренно не определяют основную поведенческую категорию.",
        "",
        "## Выходные файлы",
        "",
        "- `tables/client_behavior_clusters.csv` — клиент, новая группа, признаки и теги;",
        "- `tables/client_clusters.csv` — компактный файл для повторного анализа после кризиса;",
        "- `tables/behavior_cluster_summary.csv` — профиль новых групп;",
        "- `figures/behavior_cluster_sizes.png` — размеры групп;",
        "- `figures/behavior_cluster_profiles.png` — профили поведения.",
    ]
    (output_dir / "behavior_cluster_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_behavior_plots(clients: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [f"{row.cluster_id}. {row.cluster_label}" for row in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(12, max(4, len(summary) * 0.7)), layout="constrained")
    bars = ax.barh(labels, summary["clients"], color="#4C78A8")
    ax.bar_label(
        bars,
        labels=[f"{row.clients:,} ({row.share_pct:.1f}%)" for row in summary.itertuples()],
        padding=5,
    )
    ax.invert_yaxis()
    ax.margins(x=0.22)
    ax.set(title="Размеры поведенческих групп", xlabel="Клиентов")
    fig.savefig(figures / "behavior_cluster_sizes.png", dpi=160)
    plt.close(fig)

    columns = {
        "fines_pre": "Штрафы до кризиса",
        "fines_last_12m_total": "Исторические штрафы",
        "speeding_share_pre": "Доля скорости",
        "dangerous_events_pre": "Опасные нарушения",
        "offence_types_pre": "Типы нарушений",
    }
    values = clients[list(columns)]
    matrix = (
        (clients.groupby("cluster_id")[list(columns)].mean() - values.mean())
        .div(values.std().replace(0, np.nan))
        .reindex(summary["cluster_id"])
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, max(4, len(summary) * 0.7)), layout="constrained")
    heatmap = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=-3, vmax=3)
    ax.set_xticks(range(len(columns)), columns.values(), rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Поведенческие профили: отклонение от среднего")
    for row in range(len(matrix)):
        for col in range(len(columns)):
            ax.text(col, row, f"{matrix.iloc[row, col]:+.1f}", ha="center", va="center")
    fig.colorbar(heatmap, ax=ax, label="Стандартные отклонения")
    fig.savefig(figures / "behavior_cluster_profiles.png", dpi=160)
    plt.close(fig)


def run_behavior_clustering(
    data_dir: Path,
    features_file: Path | None = None,
    output_dir: Path | None = None,
    fines_file: str = "fines_2026.csv",
    sources: DataSources | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or PROJECT_ROOT / "outputs" / "behavior_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    if features_file is None:
        logging.info("Подготовка докризисной витрины из %s", data_dir)
        sources = sources or resolve_data_sources(data_dir, PROJECT_ROOT)
        demographics, fines, fuel = load_sources(data_dir, sources=sources)
        validation = validate_sources(demographics, fines, fuel)
        validation["source_mode"] = sources.mode
        features = build_client_features(demographics, fines, fuel)
        (output_dir / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        fines, features = load_inputs(
            data_dir,
            features_file,
            fines_file=fines_file,
        )
    features = features.drop(
        columns=[
            "cluster_id",
            "cluster_label",
            "old_cluster_id",
            "old_cluster_label",
            "behavior_cluster_id",
            "behavior_cluster_label",
        ],
        errors="ignore",
    )
    fine_behavior = build_fine_behavior(fines)
    clients, thresholds = assign_behavior_clusters(features, fine_behavior)
    summary = build_summary(clients)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    client_columns = [
        "client_id",
        "cluster_id",
        "cluster_label",
        "behavior_cluster_id",
        "behavior_cluster_label",
        "behavior_tags",
        "fines_pre",
        "fine_amount_pre",
        "offence_types_pre",
        "fine_regions_pre",
        "speeding_events_pre",
        "speeding_share_pre",
        "dangerous_events_pre",
        "fines_last_12m_total",
    ]
    client_columns += [c for c in clients.columns if c.startswith("tag_")]
    existing = [c for c in client_columns if c in clients]
    clients[existing].to_csv(
        tables_dir / "client_behavior_clusters.csv", index=False, encoding="utf-8-sig"
    )
    clients[["client_id", "cluster_id", "cluster_label"]].to_csv(
        tables_dir / "client_clusters.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(tables_dir / "behavior_cluster_summary.csv", index=False, encoding="utf-8-sig")
    features.to_csv(tables_dir / "client_features_precrisis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([thresholds]).to_csv(
        tables_dir / "behavior_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    save_behavior_plots(clients, summary, output_dir)
    write_report(clients, summary, thresholds, output_dir)
    logging.info("Поведенческие кластеры сохранены в %s", output_dir)
    return clients, summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    project_root = PROJECT_ROOT
    sources = resolve_data_sources(args.data_dir, project_root, source=args.source)
    clients, summary = run_behavior_clustering(
        sources.directory,
        args.features_file.resolve() if args.features_file else None,
        args.output_dir.resolve() if args.output_dir else None,
        fines_file=args.fines or sources.fines,
        sources=sources,
    )
    print(f"Поведенческая кластеризация: {len(summary)} групп, {len(clients):,} клиентов.")
    print(summary[["cluster_id", "cluster_label", "clients", "share_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
