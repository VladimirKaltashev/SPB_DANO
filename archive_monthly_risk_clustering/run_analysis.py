"""Run with: .venv/bin/python -m archive_monthly_risk_clustering.run_analysis"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .prepare_monthly_data import prepare_data
from .clustering import FEATURES, fit_cells
from .aggregate_results import aggregate_results
from .regression import add_price_shock, fit_regression


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=root / "client_week_panel_v2.csv")
    parser.add_argument("--data-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--baseline-month", default="2026-04")
    parser.add_argument("--min-drivers", type=int, default=50)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    from .visualization import draw_figures, write_report

    print("Preparing calendar-month data and reconciling to v2...", flush=True)
    monthly, prices, metadata = prepare_data(args.data_root, args.panel, tables)
    prices = add_price_shock(prices, args.baseline_month)
    prices.to_csv(tables / "region_month_prices.csv", index=False)
    metadata.update(baseline_month=args.baseline_month, min_drivers=args.min_drivers,
                    min_cluster_size=args.min_cluster_size,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    summaries, profiles, monthly_summaries, cell_summaries = [], [], [], []
    samples, assignments_all = {}, {}
    for model in FEATURES:
        print(f"Fitting independent region-month KMeans: {model}...", flush=True)
        assignments, cells = fit_cells(monthly, model, args.min_drivers, args.min_cluster_size)
        region_month, cell_profiles, pooled, by_month = aggregate_results(assignments, cells)
        region_month = region_month.merge(prices.drop(columns="fuel_liters"), on=["region", "month"],
                                          how="left", validate="one_to_one")
        row, sample = fit_regression(region_month, model)
        assignments.to_csv(tables / f"driver_clusters_{model}.csv", index=False)
        region_month.to_csv(tables / f"region_month_{model}.csv", index=False)
        cell_profiles.to_csv(tables / f"cluster_profiles_by_cell_{model}.csv", index=False)
        pooled.to_csv(tables / f"cluster_diagnostics_{model}.csv", index=False)
        sample.to_csv(tables / f"regression_sample_{model}.csv", index=False)
        by_month.to_csv(tables / f"monthly_share_{model}.csv", index=False)
        pooled["model"], by_month["model"] = model, model
        profiles.append(pooled)
        monthly_summaries.append(by_month)
        cell_summaries.append(cells)
        summaries.append(row)
        samples[model], assignments_all[model] = sample, assignments
    regression = pd.DataFrame(summaries)
    regression.to_csv(tables / "regression_summary.csv", index=False)
    keys = ["client_id", "region", "month"]
    paired = assignments_all["full"][keys + ["risk_label", "cell_status"]].merge(
        assignments_all["no_liters"][keys + ["risk_label", "cell_status"]], on=keys,
        suffixes=("_full", "_no_liters"), validate="one_to_one")
    paired = paired[paired.cell_status_full.eq("ok") & paired.cell_status_no_liters.eq("ok")]
    paired["same_label"] = paired.risk_label_full.eq(paired.risk_label_no_liters)
    comparison = paired.groupby("month", as_index=False).agg(
        n_driver_months=("client_id", "size"), share_same_label=("same_label", "mean"))
    comparison.to_csv(tables / "model_agreement.csv", index=False)
    all_monthly, all_profiles, all_cells = map(pd.concat, [monthly_summaries, profiles, cell_summaries])
    all_cells.to_csv(tables / "cell_diagnostics.csv", index=False)
    metadata["regression_omitted_cells"] = {
        model: int(len(all_cells[all_cells.model.eq(model)]) - len(samples[model])) for model in FEATURES}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    draw_figures(samples, regression, all_monthly, figures)
    write_report(output, metadata, regression, all_profiles, all_monthly, all_cells, comparison)
    print(regression[["model", "beta", "p_value", "R_squared", "N"]].to_string(index=False))
    print(f"Report: {output / 'index.html'}")


if __name__ == "__main__":
    main()
