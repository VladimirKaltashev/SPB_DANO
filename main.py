"""Единый запуск кластеризации и анализа изменений после кризиса."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cluster_analys import (
    DEFAULT_ANALYSIS_END,
    DEFAULT_CRISIS_START,
    DEFAULT_DEMOGRAPHICS,
    DEFAULT_FINES,
    DEFAULT_FUEL,
)
from cluster_analys import (
    run as run_cluster_analysis,
)
from spb_clustering import locate_data_dir, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help=(
            "Фиксированное число кластеров (2 и больше); "
            "без параметра выбирается автоматически среди 3–8."
        ),
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Не переобучать кластеры, использовать уже сохранённый файл.",
    )
    parser.add_argument("--crisis-start", default=DEFAULT_CRISIS_START)
    parser.add_argument("--analysis-end", default=DEFAULT_ANALYSIS_END)
    parser.add_argument("--min-clients", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.20)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_dir = locate_data_dir(args.data_dir, project_root)
    clustering_dir = project_root / "outputs" / "clustering"
    clusters_file = clustering_dir / "tables" / "client_clusters.csv"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    if not args.skip_clustering:
        clustering_result = run_pipeline(
            data_dir,
            clustering_dir,
            requested_k=args.clusters,
        )
        print(
            f"Кластеризация завершена: {clustering_result.selected_k} кластеров, "
            f"{len(clustering_result.features):,} клиентов."
        )
    elif not clusters_file.exists():
        raise FileNotFoundError(
            "Файл кластеров не найден. Уберите --skip-clustering или сначала "
            "запустите spb_clustering.py."
        )

    analysis_args = argparse.Namespace(
        data_dir=data_dir,
        output_dir=project_root / "outputs" / "cluster_analysis",
        demographics=DEFAULT_DEMOGRAPHICS,
        fines=DEFAULT_FINES,
        fuel=DEFAULT_FUEL,
        clusters_file=clusters_file,
        cluster_column="cluster_id",
        crisis_start=args.crisis_start,
        analysis_end=args.analysis_end,
        n_clusters=args.clusters or 5,
        min_clients=args.min_clients,
        alpha=args.alpha,
        min_effect=args.min_effect,
        anomaly_rate=args.anomaly_rate,
        log_level=args.log_level,
    )
    analysis_dir = run_cluster_analysis(analysis_args)
    print(f"Анализ кластеров и графики: {analysis_dir}")


if __name__ == "__main__":
    main()
