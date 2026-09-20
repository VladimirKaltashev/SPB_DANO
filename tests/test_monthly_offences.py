import numpy as np
import pandas as pd
import pytest

from supporting_analysis.monthly_offences import (
    bh_adjust,
    category_changes,
    monthly_cube,
    prepare_events,
)


def test_bh_keeps_untested_missing():
    result=bh_adjust(pd.Series([.01,.04,np.nan,.03]))
    assert np.allclose(result.dropna(),[.03,.04,.04])
    assert pd.isna(result.iloc[2])


def inputs():
    clients=pd.DataFrame({"client_id":[1,2],"cluster_id":[0,3],"cluster_label":["zero","speed"],"eligible_comparison":[True,True]})
    fines=pd.DataFrame({"client_id":[1,2],"bill_id":["a","b"],"bill_offence_date":["2026-06-01","2026-05-31 23:59:59"],"offence_short_statement":["X","Y"]})
    return fines,clients


def test_monthly_zero_cells_and_unique_counts():
    f,c=inputs()
    events,q=prepare_events(pd.concat([f,f.iloc[[0]]]),c,pd.Timestamp("2026-04-01"),pd.Timestamp("2026-09-01"))
    assert q["duplicate_bills_removed"]==1
    cube=monthly_cube(events,[0,3],["2026-04","2026-05","2026-06"],["X","Y"])
    assert len(cube)==12
    assert cube.fine_count.sum()==2
    assert cube.loc[cube.month.eq("2026-04"),"fine_count"].eq(0).all()


def test_conflicting_bill_is_rejected():
    f,c=inputs()
    f.loc[1,"bill_id"]="a"
    with pytest.raises(ValueError,match="Conflicting"):
        prepare_events(f,c,pd.Timestamp("2026-04-01"),pd.Timestamp("2026-09-01"))


def test_missing_mapping_is_rejected():
    f,c=inputs()
    with pytest.raises(ValueError,match="no cluster"):
        prepare_events(f,c.iloc[[0]],pd.Timestamp("2026-04-01"),pd.Timestamp("2026-09-01"))


def test_new_category_has_no_infinite_growth_or_significance():
    f,c=inputs()
    start,crisis,end=map(pd.Timestamp,["2026-04-01","2026-06-01","2026-09-01"])
    events,_=prepare_events(f,c,start,end)
    result=category_changes(events,c,start,crisis,end)
    row=result.loc[result.cluster_id.eq(0)&result.category.eq("X")].iloc[0]
    assert row.pre_count==0 and row.post_count==1
    assert pd.isna(row.rate_ratio) and pd.isna(row.q_value)
    assert not row.sharp_growth


def test_equal_daily_rates_are_equal_despite_different_period_lengths():
    ids=list(range(100))
    c=pd.DataFrame({"client_id":ids,"cluster_id":4,"cluster_label":"g4","eligible_comparison":True})
    dates=pd.date_range("2026-04-01","2026-08-31")
    f=pd.DataFrame([(client,str(client)+str(d),d,"X") for client in ids for d in dates],columns=["client_id","bill_id","bill_offence_date","offence_short_statement"])
    start,crisis,end=map(pd.Timestamp,["2026-04-01","2026-06-01","2026-09-01"])
    events,_=prepare_events(f,c,start,end)
    r=category_changes(events,c,start,crisis,end)
    assert r.rate_ratio.eq(1).all()
    assert r.delta_per_1000_clients_30d.eq(0).all()
