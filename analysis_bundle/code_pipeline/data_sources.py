"""Поиск очищенных источников и связывание недельной панели с группами."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

SourceMode = Literal["auto", "processed", "raw"]

RAW_FILENAMES = {
    "demographics": "clients_demographics.csv",
    "fines": "fines_2026.csv",
    "fuel": "fuel_transaction.csv",
}
PROCESSED_FILENAMES = {
    "demographics": "clients_demographics_clean.csv",
    "fines": "fines_clean.csv",
    "fuel": "fuel_clean.csv",
}
WEEKLY_PANEL_FILENAME = "client_week_panel.csv"


@dataclass(frozen=True)
class DataSources:
    """Три детальных источника модели и необязательная недельная панель."""

    directory: Path
    demographics: str
    fines: str
    fuel: str
    mode: Literal["processed", "raw"]
    weekly_panel: Path | None = None

    @property
    def demographics_path(self) -> Path:
        return self.directory / self.demographics

    @property
    def fines_path(self) -> Path:
        return self.directory / self.fines

    @property
    def fuel_path(self) -> Path:
        return self.directory / self.fuel


def detect_separator(path: Path) -> str:
    """Определить разделитель по заголовку CSV."""
    with path.open("r", encoding="utf-8-sig") as stream:
        header = stream.readline()
    return ";" if header.count(";") > header.count(",") else ","


def read_csv_detected(path: Path, **kwargs: object) -> pd.DataFrame:
    """Прочитать как исходный CSV с ``;`` либо очищенный CSV с ``,``."""
    separator = detect_separator(path)
    options: dict[str, object] = {
        "sep": separator,
        "decimal": "," if separator == ";" else ".",
        "encoding": "utf-8-sig",
        "low_memory": False,
    }
    options.update(kwargs)
    return pd.read_csv(path, **options)


def _candidate_directories(base: Path) -> list[Path]:
    """Вернуть возможные каталоги данных без повторов, в порядке приоритета."""
    candidates = [
        base / "data" / "processed",
        base / "processed",
        base,
        base / "data" / "raw",
        base / "raw",
    ]
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def _contains(directory: Path, filenames: dict[str, str]) -> bool:
    return directory.is_dir() and all(
        (directory / filename).is_file() for filename in filenames.values()
    )


def resolve_data_sources(
    explicit: Path | None,
    project_root: Path,
    source: SourceMode = "auto",
) -> DataSources:
    """Найти входы; в режиме ``auto`` очищенные файлы имеют приоритет."""
    if source not in {"auto", "processed", "raw"}:
        raise ValueError("source должен быть auto, processed или raw")

    modes = ["processed", "raw"] if source == "auto" else [source]
    directory_groups = (
        [_candidate_directories(explicit), _candidate_directories(project_root)]
        if explicit is not None and source == "auto"
        else [_candidate_directories(explicit or project_root)]
    )
    searched_directories: list[Path] = []
    for directories in directory_groups:
        unique_directories = list(dict.fromkeys(directories))
        searched_directories.extend(unique_directories)
        for mode in modes:
            filenames = PROCESSED_FILENAMES if mode == "processed" else RAW_FILENAMES
            for directory in unique_directories:
                if not _contains(directory, filenames):
                    continue
                panel = directory / WEEKLY_PANEL_FILENAME
                if not panel.is_file():
                    sibling_panel = (
                        project_root / "data" / "processed" / WEEKLY_PANEL_FILENAME
                    )
                    panel = sibling_panel if sibling_panel.is_file() else None
                return DataSources(
                    directory=directory,
                    demographics=filenames["demographics"],
                    fines=filenames["fines"],
                    fuel=filenames["fuel"],
                    mode=mode,
                    weekly_panel=panel,
                )

    searched = "\n- ".join(str(path) for path in dict.fromkeys(searched_directories))
    expected = (
        PROCESSED_FILENAMES.values()
        if source == "processed"
        else RAW_FILENAMES.values()
        if source == "raw"
        else [*PROCESSED_FILENAMES.values(), *RAW_FILENAMES.values()]
    )
    raise FileNotFoundError(
        "Не найдены таблицы для анализа. Ожидаются файлы: "
        f"{', '.join(expected)}. Проверены каталоги:\n- {searched}"
    )


def export_clustered_weekly_panel(
    panel_path: Path,
    clusters_path: Path,
    output_path: Path,
    chunksize: int = 250_000,
) -> dict[str, int]:
    """Добавить поведенческие группы к панели, не загружая её целиком в память."""
    clusters = read_csv_detected(clusters_path)
    required = {"client_id", "cluster_id"}
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"В {clusters_path.name} отсутствуют колонки: {missing}")
    if clusters["client_id"].duplicated().any():
        raise ValueError(f"В {clusters_path.name} client_id должен быть уникальным")
    cluster_columns = [
        column
        for column in ("client_id", "cluster_id", "cluster_label")
        if column in clusters.columns
    ]
    assignments = clusters[cluster_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_rows = 0
    unmatched = 0
    first_chunk = True
    try:
        for panel in pd.read_csv(
            panel_path,
            chunksize=chunksize,
            encoding="utf-8-sig",
            low_memory=False,
        ):
            if "client_id" not in panel.columns:
                raise ValueError(f"В {panel_path.name} отсутствует колонка client_id")
            merged = panel.merge(
                assignments,
                on="client_id",
                how="left",
                validate="many_to_one",
            )
            total_rows += len(merged)
            unmatched += int(merged["cluster_id"].isna().sum())
            merged.to_csv(
                temporary_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                encoding="utf-8-sig" if first_chunk else "utf-8",
            )
            first_chunk = False
        if first_chunk:
            raise ValueError(f"Панель пуста: {panel_path}")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "rows": total_rows,
        "unmatched_rows": unmatched,
    }
