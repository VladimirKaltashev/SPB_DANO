"""Reconcile recorded monthly events before restricting to the early cohort."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'outputs'
TABLES=OUT/'tables'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def prepare():
    previous=ROOT/'archive_fines_per_liter_analysis/outputs'
    meta=json.loads((ROOT/'archive_usage_composition_analysis/outputs/metadata.json').read_text())
    sources=meta['sources']
    for source in sources.values(): assert sha(source['path'])==source['sha256'],source['path']
    dtype={'client_id':'string','region':'string','auto_document_id':'string'}
    monthly=pd.read_csv(sources['driver_month']['path'],dtype=dtype)
    archived=pd.read_csv(sources['client_attributes']['path'],dtype=dtype)
    prices=pd.read_csv(sources['prices']['path'],dtype=dtype)
    fuel=pd.read_csv(sources['fuel']['path'],dtype=dtype)
    fines=pd.read_csv(sources['fines']['path'],dtype=dtype)
    vehicles=pd.read_csv(sources['vehicles']['path'],dtype=dtype)
    panel=pd.read_csv(sources['panel_v2']['path'],dtype={'client_id':'string'},
        usecols=['client_id','week','observed_start','observed_end','exposure_days'])
    assert not panel.duplicated(['client_id','week']).any()
    intervals=panel[['week','observed_start','observed_end','exposure_days']].drop_duplicates().sort_values('week')
    assert not intervals.week.duplicated().any()
    starts=pd.to_datetime(intervals.observed_start);ends=pd.to_datetime(intervals.observed_end)
    assert starts.iloc[0]==pd.Timestamp('2026-04-01') and ends.iloc[-1]==pd.Timestamp('2026-08-31')
    assert starts.iloc[1:].reset_index(drop=True).eq(ends.iloc[:-1].reset_index(drop=True)+pd.Timedelta(days=1)).all()
    assert panel.groupby('client_id').exposure_days.sum().eq(153).all()
    assert monthly.groupby('client_id').size().eq(5).all() and not monthly.duplicated(['client_id','month']).any()
    assert monthly.region.nunique()==16 and monthly.client_id.nunique()==25675
    for df,date,identifier in [(fuel,'order_datetime','order_id'),(fines,'bill_offence_date','bill_id')]:
        assert df[identifier].notna().all() and df[identifier].is_unique
        df['date']=pd.to_datetime(df[date],errors='raise')
        assert df.date.notna().all()
    fuel=fuel.loc[fuel.date.ge('2026-04-01') & fuel.date.lt('2026-09-01')].copy()
    fines=fines.loc[fines.date.ge('2026-04-01') & fines.date.lt('2026-09-01')].copy()
    for frame in [fuel,fines]:
        assert frame.client_id.isin(monthly.client_id).all()
        frame['month']=frame.date.dt.to_period('M').astype(str)
    assert np.isfinite(fuel.physical_fuel_volume_main).all() and fuel.physical_fuel_volume_main.ge(0).all()
    rebuilt_liters=fuel.groupby(['client_id','month']).physical_fuel_volume_main.sum().rename('rebuilt_liters')
    rebuilt_fines=fines.groupby(['client_id','month']).size().rename('rebuilt_fines')
    checked=monthly.join(rebuilt_liters,on=['client_id','month']).join(rebuilt_fines,on=['client_id','month'])
    checked[['rebuilt_liters','rebuilt_fines']]=checked[['rebuilt_liters','rebuilt_fines']].fillna(0)
    audit=[]
    for left,right in [('fuel_liters','rebuilt_liters'),('fines_count','rebuilt_fines')]:
        difference=(checked[left]-checked[right]).abs()
        assert difference.le(1e-6).all()
        audit.append(dict(metric=left,previous_total=checked[left].sum(),rebuilt_total=checked[right].sum(),
            max_abs_difference=difference.max(),mismatched_client_months=int(difference.gt(1e-6).sum())))
    pd.DataFrame(audit).to_csv(TABLES/'source_reconciliation.csv',index=False)
    assert np.allclose(prices.price_shock,prices.avg_fuel_price/prices.baseline_price_region-1)
    assert set(prices.baseline_month)=={'2026-04'}
    attributes=archived.drop_duplicates('client_id')[['client_id','subscription_creation_date','engine_volume','horsepower','car_count']].copy()
    attributes['subscription_creation_date']=pd.to_datetime(attributes.subscription_creation_date,errors='raise')
    # Preserve history without summing possibly repeated or conflicting histories across vehicles.
    history_cols=[c for c in vehicles if c.endswith('_2025_fines') or c.startswith('fines_last_')]
    history=vehicles.groupby('client_id')[history_cols].agg(lambda s:s.dropna().iloc[0] if s.nunique()==1 else np.nan)
    conflicts=vehicles.groupby('client_id')[history_cols].nunique().gt(1).add_suffix('_multiple_values')
    attributes=attributes.join(history,on='client_id').join(conflicts,on='client_id')
    vehicles[['client_id','auto_document_id',*history_cols]].to_csv(TABLES/'historical_fines_by_vehicle.csv',index=False)
    all_data=monthly.merge(attributes,on='client_id',validate='many_to_one').merge(
        prices[['region','month','price_shock']],on=['region','month'],validate='many_to_one')
    assert all_data.price_shock.notna().all()
    eligible=all_data.subscription_creation_date.le('2026-04-01')
    data=all_data.loc[eligible].sort_values(['client_id','month']).reset_index(drop=True)
    assert data.groupby('client_id').size().eq(5).all() and data.groupby('client_id').region.nunique().eq(1).all()
    assert set(data.client_id)<=set(panel.client_id)
    data['fuel_liters_100']=data.fuel_liters/100
    data['has_fuel_purchase']=data.fuel_liters.gt(0).astype(float)
    data.to_csv(TABLES/'client_month.csv',index=False)
    zero=data.fuel_liters.eq(0);total_fines=int(data.fines_count.sum())
    zeros=data.groupby('has_fuel_purchase',as_index=False).agg(n_client_months=('client_id','size'),
        n_clients=('client_id','nunique'),total_fines=('fines_count','sum'),mean_fines=('fines_count','mean'),
        months_with_fines=('fines_count',lambda x:x.gt(0).sum()))
    zeros['share_of_fines']=zeros.total_fines/total_fines;zeros.to_csv(TABLES/'zero_purchase_audit.csv',index=False)
    distribution=dict(share_0=float(data.fines_count.eq(0).mean()),share_1=float(data.fines_count.eq(1).mean()),
        share_2=float(data.fines_count.eq(2).mean()),share_3plus=float(data.fines_count.ge(3).mean()),
        mean=float(data.fines_count.mean()),variance=float(data.fines_count.var(ddof=0)),max=int(data.fines_count.max()))
    pd.DataFrame([distribution]).to_csv(TABLES/'fines_distribution.csv',index=False)
    counts=data.groupby('client_id').agg(total_fines=('fines_count','sum'),purchase_months=('has_fuel_purchase','sum'))
    summary=dict(sources=sources,all_clients=monthly.client_id.nunique(),all_observations=len(monthly),
        eligible_clients=data.client_id.nunique(),eligible_observations=len(data),regions=data.region.nunique(),
        months=sorted(data.month.unique()),excluded_late_or_missing_subscription=all_data.loc[~eligible,'client_id'].nunique(),
        missing_subscription_clients=int(attributes.subscription_creation_date.isna().sum()),
        missing_engine_eligible_clients=data.loc[data.engine_volume.isna(),'client_id'].nunique(),
        share_zero_fuel=float(zero.mean()),zero_fuel_client_months=int(zero.sum()),
        fines_in_zero_fuel_months=int(data.loc[zero,'fines_count'].sum()),fines_share_zero_fuel=float(data.loc[zero,'fines_count'].sum()/total_fines),
        full_cohort_fines_share_zero_fuel=float(monthly.loc[monthly.fuel_liters.eq(0),'fines_count'].sum()/monthly.fines_count.sum()),
        eligible_total_fines=total_fines,eligible_total_liters=float(data.fuel_liters.sum()),
        all_zero_fines_clients=int(counts.total_fines.eq(0).sum()),
        fuel_purchase_switching_clients=int(counts.purchase_months.between(1,4).sum()),
        fines_distribution=distribution,covered_days_per_client=153,
        historical_conflict_clients=int(conflicts.any(axis=1).sum()),
        zero_event_interpretation='no recorded cleaned event in a fully gridded service observation month; not proof of complete capture or no driving')
    dump('data_audit.json',summary)
    return data,summary
