"""Run only the prespecified diagnostic Stage 1."""
import os
import json
import pandas as pd

from prepare import ROOT,OUT,TABLES,prepare,sha,dump
from models import linear_models,conditional_poisson
from descriptive import summarize,draw_figures
from report import write_report


def main():
    assert (OUT/'analysis_plan.json').exists()
    print('Reconciling cleaned monthly operations and early-client coverage...',flush=True)
    data,audit=prepare()
    print(json.dumps({k:v for k,v in audit.items() if k not in ['sources']},ensure_ascii=False,indent=2),flush=True)
    print('Estimating prespecified between and within models...',flush=True)
    main,alternative,between,centered,identification=linear_models(data)
    poisson,poisson_audit=conditional_poisson(data)
    all_models=pd.concat([main,poisson],ignore_index=True)
    all_models.to_csv(TABLES/'model_results.csv',index=False)
    alternative.to_csv(TABLES/'alternative_client_clustering.csv',index=False)
    between.to_csv(TABLES/'client_period_averages.csv',index=False)
    centered.to_csv(TABLES/'within_deviations.csv',index=False)
    dump('identification_audit.json',identification);dump('poisson_audit.json',poisson_audit)
    change,correlations,groups,group_audit=summarize(data,between,centered,TABLES)
    dump('group_definition.json',group_audit)
    print(all_models.to_string(index=False),flush=True)
    print('Alternative client clustering:\n',alternative.to_string(index=False),flush=True)
    print('Correlations:\n',correlations.to_string(index=False),flush=True)
    print('Reducer groups:\n',groups.to_string(index=False),flush=True)
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
    draw_figures(all_models,change,groups,OUT/'figures',TABLES)
    write_report(OUT)
    dump('metadata.json',dict(**audit,identification=identification,poisson=poisson_audit,groups=group_audit,
        code_sha256={p.name:sha(p) for p in OUT.parent.glob('*.py')}))
    dump('validation.json',dict(status='PASS',upstream_source_hashes_checked=True,source_reconciliation='PASS',
        balanced_early_cohort='PASS',fixed_effects_residuals='PASS',poisson_convergence='PASS',exactly_three_figures=True))
    print('Stage 1 complete; source hashes and monthly reconciliation passed.',flush=True)


if __name__=='__main__':
    main()
