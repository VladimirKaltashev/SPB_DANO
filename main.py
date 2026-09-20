"""Главный воспроизводимый пайплайн проекта: raw → очистка → анализы."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from current_hypothesis.run_analysis import run as run_current_hypothesis
from pipeline.clean_data import run_cleaning
from supporting_analysis.monthly_offences import run as run_monthly_offences

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Использовать уже созданные data/processed вместо пересчёта из raw.",
    )
    parser.add_argument(
        "--skip-current-hypothesis",
        action="store_true",
        help="Не запускать группировку и проверку текущей гипотезы.",
    )
    parser.add_argument(
        "--skip-monthly-offences",
        action="store_true",
        help="Не строить дополнительный помесячный отчёт по типам штрафов.",
    )
    parser.add_argument("--skip-panel-export", action="store_true")
    parser.add_argument("--analysis-end", default="2026-09-01")
    parser.add_argument("--min-clients", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.20)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--top-n", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    results: dict[str, Path] = {}

    if args.skip_cleaning:
        logging.info("Этап 1/3: очистка пропущена, используются %s", processed_dir)
    else:
        logging.info("Этап 1/3: очистка raw и построение недельной панели")
        cleaning = run_cleaning(
            raw_dir,
            processed_dir,
            reference_panel=PROJECT_ROOT / "data" / "reference" / "client_week_panel_v2.csv",
        )
        results["cleaning"] = cleaning.output_dir
        reference_text = (
            "совпала" if cleaning.reference_matches else "не совпала"
            if cleaning.reference_matches is not None
            else "не проверялась"
        )
        print(
            f"Очистка: {cleaning.clients:,} клиентов, "
            f"{cleaning.panel_rows:,} строк панели; v2-проверка: {reference_text}.",
            flush=True,
        )

    if args.skip_current_hypothesis:
        logging.info("Этап 2/3: текущая гипотеза пропущена")
    else:
        logging.info("Этап 2/3: группы и проверка текущей гипотезы")
        hypothesis_dir = run_current_hypothesis(
            argparse.Namespace(
                data_dir=processed_dir,
                source="processed",
                output_dir=output_dir,
                skip_clustering=False,
                crisis_start="2026-06-01",
                analysis_end=args.analysis_end,
                min_clients=args.min_clients,
                alpha=args.alpha,
                min_effect=args.min_effect,
                anomaly_rate=args.anomaly_rate,
                skip_behavior_analysis=False,
                log_level=args.log_level,
                skip_panel_export=args.skip_panel_export,
            )
        )
        results["current_hypothesis"] = hypothesis_dir

    if args.skip_monthly_offences:
        logging.info("Этап 3/3: дополнительный помесячный отчёт пропущен")
    else:
        logging.info("Этап 3/3: дополнительный помесячный отчёт")
        clients_file = output_dir / "behavior_cluster_analysis" / "client_behavior_features.csv"
        if not clients_file.is_file():
            raise FileNotFoundError(
                "Для помесячного отчёта сначала нужен результат текущей гипотезы: "
                f"{clients_file}"
            )
        monthly_output = output_dir / "monthly_offences"
        run_monthly_offences(
            argparse.Namespace(
                data_dir=PROJECT_ROOT,
                output_dir=monthly_output,
                clients_file=clients_file,
                start="2026-04-01",
                crisis_start="2026-06-01",
                end=args.analysis_end,
                top_n=args.top_n,
            )
        )
        results["monthly_offences"] = monthly_output

    print("\nПайплайн завершён. Результаты:", flush=True)
    for stage, path in results.items():
        print(f"- {stage}: {path}", flush=True)
    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
