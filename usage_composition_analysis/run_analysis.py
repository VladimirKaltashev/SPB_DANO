"""Run: .venv/bin/python -m usage_composition_analysis.run_analysis"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_data import GROUPS, prepare_data, digest
from .metrics import group_panels, composition
from .regressions import fit


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    print("Validating baseline dates, source versions and fixed membership...", flush=True)
    drivers, prices, baseline, summary, zero, metadata = prepare_data(args.data_root, tables)
    panel, national = group_panels(drivers, prices, metadata["baseline_month"])
    panel.to_csv(tables / "region_month_usage_groups.csv", index=False)
    national.to_csv(tables / "national_usage_groups.csv", index=False)
    primary = panel.loc[panel.usage_group.isin(GROUPS)].copy()
    if primary.groupby(["region", "month"]).usage_group.nunique().ne(3).any():
        raise ValueError("Missing positive-baseline group in a region-month")
    metadata.update(n_group_cells=len(primary), minimum_group_drivers=int(primary.n_drivers.min()),
                    n_tertile_driver_months=int(drivers.usage_group.isin(GROUPS).sum()))
    single = []
    print("Fitting separately by fixed group...", flush=True)
    for outcome in ["liters_per_driver", "liters_index", "fines_per_driver"]:
        for group in GROUPS:
            fitted, _ = fit(primary.loc[primary.usage_group.eq(group)], outcome, f"{outcome}_{group}")
            fitted["usage_group"] = group
            single.append(fitted)
    single = pd.concat(single, ignore_index=True)
    single.to_csv(tables / "group_regressions.csv", index=False)
    print("Testing HIGH-LOW and MEDIUM-LOW slope differences...", flush=True)
    interaction, joint = [], []
    for period, sample in [("all_months", primary), ("post_baseline", primary.loc[primary.month.gt(metadata["baseline_month"])])]:
        for outcome in ["liters_per_driver", "liters_index"]:
            rows, test = fit(sample, outcome, f"interaction_{outcome}_{period}", interactions=True)
            rows["period"] = period
            interaction.append(rows)
            joint.append(test)
    interactions = pd.concat(interaction, ignore_index=True)
    interactions.to_csv(tables / "interaction_results.csv", index=False)
    base_positive = baseline.loc[baseline.usage_group.isin(GROUPS)]
    baseline_results, baseline_joint = fit(base_positive, "baseline_fines", "baseline_fines_by_usage", group_effects=True)
    baseline_results.to_csv(tables / "baseline_fines_comparisons.csv", index=False)
    joint.append(baseline_joint)
    pd.DataFrame(joint).to_csv(tables / "joint_tests.csv", index=False)
    print("Checking active-user composition and the predeclared sensitivities...", flush=True)
    composition_panels, composition_national, composition_fits = [], [], []
    for include_zero in [False, True]:
        population = "including_zero" if include_zero else "tertiles_only"
        for threshold in [0, 20]:
            cells, totals = composition(drivers, prices, threshold, include_zero)
            composition_panels.append(cells)
            composition_national.append(totals)
            for group in GROUPS:
                rows, _ = fit(cells, f"share_{group}", f"composition_{population}_{threshold}L_{group}")
                rows["threshold_liters"], rows["denominator"], rows["usage_group"] = threshold, population, group
                composition_fits.append(rows)
    pd.concat(composition_panels, ignore_index=True).to_csv(tables / "active_composition_region_month.csv", index=False)
    national_composition = pd.concat(composition_national, ignore_index=True)
    national_composition.to_csv(tables / "active_composition_national.csv", index=False)
    composition_results = pd.concat(composition_fits, ignore_index=True)
    composition_results.to_csv(tables / "composition_regressions.csv", index=False)
    metadata["code_sha256"] = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")
    if not args.tables_only:
        os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
        from .visualization import draw_figures
        from .report import write_report
        draw_figures(national, summary, single, interactions, baseline_results, national_composition, figures)
        write_report(out, metadata, summary, zero, national, single, interactions, baseline_results,
                     composition_results, national_composition)
    print("\nGroup slopes:")
    print(single.loc[single.term.eq("price_shock"), ["usage_group", "outcome", "beta", "SE", "p_value", "R_squared", "N"]].to_string(index=False))
    print("\nInteraction contrasts:")
    print(interactions.loc[interactions.term.str.startswith("PriceShock_x"), ["period", "outcome", "term", "beta", "SE", "p_value", "N"]].to_string(index=False))
    print("\nBaseline fines comparisons:\n", baseline_results.to_string(index=False))
    print("\nComposition slopes:\n", composition_results.loc[composition_results.term.eq("price_shock"),
          ["denominator", "threshold_liters", "usage_group", "beta", "SE", "p_value"]].to_string(index=False))
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
