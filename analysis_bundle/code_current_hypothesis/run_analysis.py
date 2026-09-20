"""Единый запуск поведенческой сегментации и анализа изменений после кризиса."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from current_hypothesis.define_groups import run_behavior_clustering
from current_hypothesis.check_hypothesis import (
    DEFAULT_ANALYSIS_END,
    DEFAULT_CRISIS_START,
)
from current_hypothesis.check_hypothesis import (
    run as run_cluster_analysis,
)
from pipeline.data_sources import export_clustered_weekly_panel, resolve_data_sources

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--source", choices=["auto", "processed", "raw"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None, help="Корневая папка результатов.")
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Повторить анализ по сохранённым поведенческим группам.",
    )
    parser.add_argument(
        "--crisis-start",
        default=DEFAULT_CRISIS_START,
        choices=[DEFAULT_CRISIS_START],
        help="Отсечка поведенческой схемы: 2026-06-01.",
    )
    parser.add_argument("--analysis-end", default=DEFAULT_ANALYSIS_END)
    parser.add_argument("--min-clients", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.20)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument(
        "--skip-behavior-analysis",
        action="store_true",
        help="Построить только поведенческие группы и их профили.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-panel-export", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    project_root = PROJECT_ROOT
    sources = resolve_data_sources(args.data_dir, project_root, source=args.source)
    data_dir = sources.directory
    logging.info("Источник: %s (%s)", sources.mode, data_dir)
    output_root = (args.output_dir or project_root / "outputs").resolve()
    behavior_dir = output_root / "behavior_clustering"
    clusters_file = behavior_dir / "tables" / "client_clusters.csv"

    if args.skip_clustering:
        if not clusters_file.is_file():
            raise FileNotFoundError(
                f"Нет поведенческих групп: {clusters_file}. Запустите без --skip-clustering."
            )
    else:
        clients, summary = run_behavior_clustering(
            data_dir,
            output_dir=behavior_dir,
            sources=sources,
        )
        print(
            f"Поведенческие группы: {len(summary)}, клиентов: {len(clients):,}.",
            flush=True,
        )

    if sources.weekly_panel is not None and not args.skip_panel_export:
        panel_path = behavior_dir / "tables" / "client_week_panel_with_clusters.csv"
        logging.info("Добавление поведенческих групп в недельную панель")
        panel_stats = export_clustered_weekly_panel(
            sources.weekly_panel,
            clusters_file,
            panel_path,
        )
        print(f"Недельная панель: {panel_stats['rows']:,} строк → {panel_path}", flush=True)

    if args.skip_behavior_analysis:
        return behavior_dir

    analysis_dir = run_cluster_analysis(
        argparse.Namespace(
            data_dir=data_dir,
            output_dir=output_root / "behavior_cluster_analysis",
            demographics=sources.demographics,
            fines=sources.fines,
            fuel=sources.fuel,
            source=sources.mode,
            clusters_file=clusters_file,
            cluster_column="cluster_id",
            crisis_start=args.crisis_start,
            analysis_end=args.analysis_end,
            min_clients=args.min_clients,
            alpha=args.alpha,
            min_effect=args.min_effect,
            anomaly_rate=args.anomaly_rate,
            log_level=args.log_level,
        )
    )
    print(f"Все графики: {analysis_dir / 'index.html'}", flush=True)
    return analysis_dir


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
