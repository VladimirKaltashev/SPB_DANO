"""Run: .venv/bin/python -m fines_per_liter_analysis.run_analysis"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_data import prepare_data, sha256
from .metrics import (aggregate_metrics, attach_risk, individual_ratios,
                      liters_distribution, select_regions)
from .regressions import fit_ols, fit_three

THRESHOLDS = (10, 20, 40)
PRIMARY_THRESHOLD = 20


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root)
    parser.add_argument("--archive", type=Path, default=root / "archive_monthly_risk_clustering")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    print("Validating complete v2 cohort, archived labels and regional price shock...", flush=True)
    drivers, prices, labels, metadata = prepare_data(args.data_root, args.archive, tables)
    distribution = liters_distribution(drivers)
    distribution.to_csv(tables / "liters_distribution.csv", index=False)
    metadata.update(individual_thresholds_liters=list(THRESHOLDS), primary_threshold_liters=PRIMARY_THRESHOLD,
                    threshold_rationale="Prespecified 10/20/40L before outcome regressions; 20L excludes very small denominators while retaining most positive-purchase months.")
    overall = aggregate_metrics(drivers, prices)
    overall.to_csv(tables / "region_month.csv", index=False)
    overall[["region", "region_name", "month", "price_shock", "liters_per_driver", "fines_per_driver",
             "fines_per_1000_liters"]].to_csv(tables / "regional_dynamics.csv", index=False)
    national = overall.groupby("month", as_index=False).agg(
        n_drivers=("n_drivers", "sum"), total_fines=("total_fines", "sum"),
        total_liters=("total_liters", "sum"), total_fuel_cost_rub=("total_fuel_cost_rub", "sum"))
    national["liters_per_driver"] = national.total_liters / national.n_drivers
    national["fines_per_driver"] = national.total_fines / national.n_drivers
    national["fines_per_1000_liters"] = 1000 * national.total_fines / national.total_liters
    national["avg_fuel_price"] = national.total_fuel_cost_rub / national.total_liters
    national.to_csv(tables / "national_monthly.csv", index=False)
    selected = select_regions(prices)
    selected.to_csv(tables / "selected_regions.csv", index=False)
    print("Fitting the three primary regressions...", flush=True)
    main_rows, main_samples = fit_three(overall)
    rows = list(main_rows)
    sample_tables = [sample.assign(outcome=outcome, population="ALL", block="main")
                     for outcome, sample in main_samples.items()]
    current = attach_risk(drivers, labels)
    lagged = attach_risk(drivers, labels, lag=True)
    current_panel = aggregate_metrics(current, prices, by_risk=True)
    lagged_panel = aggregate_metrics(lagged, prices, by_risk=True)
    current_panel.to_csv(tables / "region_month_risk_current.csv", index=False)
    lagged_panel.to_csv(tables / "region_month_risk_lagged.csv", index=False)
    lagged[["client_id", "region", "month", "risk_source_month", "risk_label"]].to_csv(
        tables / "lagged_driver_risk.csv", index=False)
    print("Fitting current and previous-month NO_LITERS group regressions...", flush=True)
    for block, panel in [("current_risk", current_panel), ("lagged_risk", lagged_panel)]:
        for label in ["high_risk", "low_risk"]:
            part_rows, samples = fit_three(panel.loc[panel.risk_label.eq(label)], label.upper(), block)
            rows.extend(part_rows)
            sample_tables.extend(sample.assign(outcome=outcome, population=label.upper(), block=block)
                                 for outcome, sample in samples.items())
    regressions = pd.DataFrame(rows)
    regressions.to_csv(tables / "regression_summary.csv", index=False)
    regressions.loc[regressions.block.ne("lagged_risk")].to_csv(tables / "main_and_current_risk_results.csv", index=False)
    regressions.loc[regressions.block.eq("lagged_risk")].to_csv(tables / "lagged_risk_results.csv", index=False)
    pd.concat(sample_tables, ignore_index=True).to_csv(tables / "regression_samples.csv", index=False)
    coverage = overall[["region", "month", "n_drivers", "total_fines", "total_liters"]].merge(
        current_panel.groupby(["region", "month"], as_index=False)[["n_drivers", "total_fines", "total_liters"]].sum(),
        on=["region", "month"], suffixes=("_all", "_classified"), validate="one_to_one")
    coverage["n_unclassified"] = coverage.n_drivers_all - coverage.n_drivers_classified
    coverage.to_csv(tables / "risk_coverage.csv", index=False)
    print("Fitting prespecified individual-ratio thresholds: 10, 20, 40 liters...", flush=True)
    threshold_rows, threshold_coverage = [], []
    for threshold in THRESHOLDS:
        individual = individual_ratios(drivers, prices, threshold)
        row, _ = fit_ols(individual, "individual_fines_per_100_liters", f"LITERS_GE_{threshold}", "individual_robustness")
        row.update(threshold_liters=threshold, primary_threshold=threshold == PRIMARY_THRESHOLD,
                   share_all_driver_months=len(individual) / len(drivers),
                   share_positive_driver_months=len(individual) / drivers.fuel_liters.gt(0).sum(),
                   n_unique_drivers=individual.client_id.nunique())
        threshold_rows.append(row)
        group = individual.groupby(["region", "month"], as_index=False).agg(
            n_driver_months=("client_id", "size"), mean_individual_fines_per_100_liters=("individual_fines_per_100_liters", "mean"),
            total_fines=("fines_count", "sum"), total_liters=("fuel_liters", "sum"))
        group["threshold_liters"] = threshold
        group["ratio_of_totals_per_100_liters"] = 100 * group.total_fines / group.total_liters
        threshold_coverage.append(group)
        if threshold == min(THRESHOLDS):
            for t in THRESHOLDS:
                individual[f"eligible_{t}L"] = individual.fuel_liters.ge(t)
            individual.to_csv(tables / "individual_ratios.csv", index=False)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(tables / "individual_threshold_results.csv", index=False)
    pd.concat(threshold_coverage, ignore_index=True).to_csv(tables / "individual_threshold_cell_coverage.csv", index=False)
    metadata.update(n_current_risk_driver_months=len(current), n_lagged_risk_driver_months=len(lagged),
                    n_current_group_cells=len(current_panel), n_lagged_group_cells=len(lagged_panel),
                    minimum_drivers_per_cell=int(overall.n_drivers.min()),
                    minimum_current_group_drivers=int(current_panel.n_drivers.min()),
                    minimum_lagged_group_drivers=int(lagged_panel.n_drivers.min()),
                    software={name: importlib.metadata.version(name) for name in ["pandas", "numpy", "statsmodels", "matplotlib"]},
                    code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")})
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")
    if not args.tables_only:
        os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
        from .visualization import draw_figures
        from .report import write_report
        draw_figures(overall, regressions, selected, distribution, national, figures)
        write_report(out, metadata, regressions, thresholds, distribution, selected, national)
    print(regressions[["block", "population", "outcome", "BETA", "SE", "p_value", "R_squared", "N"]].to_string(index=False))
    print("\nIndividual thresholds:")
    print(thresholds[["threshold_liters", "BETA", "SE", "p_value", "R_squared", "N"]].to_string(index=False))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
