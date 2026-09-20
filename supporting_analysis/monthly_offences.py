"""Monthly fine histograms and exploratory category-change screening.

Run after main.py. Counts are unique bill_id, not monetary amounts or crashes.
The PPTX builder consumes deck_data.json and keeps its charts editable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t

from pipeline.plotting import plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTH_NAMES = {1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май", 6: "Июнь", 7: "Июль", 8: "Август", 9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"}
COLORS = ["#2B6CA3", "#EF8D32", "#20988E", "#B46393", "#8666B1", "#9CA9B3"]
CATEGORY_COLORS = {"Превышение скорости на 20-40 км/ч":"#2B6CA3", "Не пристегнут ремень безопасности":"#EF8D32", "Нарушение разметки":"#20988E", "Превышение скорости на 40-60 км/ч":"#8666B1", "Использование телефона за рулем":"#B46393", "Проезд на красный сигнал светофора":"#B03F3F", "Движение по выделенной полосе":"#657E34", "Пересечение стоп-линии":"#A57A2A"}
SHORT_NAMES = {
    "Превышение скорости на 20-40 км/ч": "Скорость +20-40 км/ч",
    "Превышение скорости на 40-60 км/ч": "Скорость +40-60 км/ч",
    "Превышение скорости на 60-80 км/ч": "Скорость +60-80 км/ч",
    "Превышение скорости более чем на 80 км/ч": "Скорость +80 км/ч",
    "Не пристегнут ремень безопасности": "Ремень безопасности",
    "Использование телефона за рулем": "Телефон за рулём",
    "Проезд на красный сигнал светофора": "Красный сигнал",
    "Остановка или стоянка в неположенном месте (Москва и Санкт-Петербург)": "Стоянка: Москва и СПб",
    "Остановка или стоянка в неположенном месте": "Стоянка: прочие регионы",
    "Движение по выделенной полосе (Москва и Санкт-Петербург)": "Выделенная полоса: Москва и СПб",
    "Движение по выделенной полосе": "Выделенная полоса",
    "Поворот (разворот) в запрещенном месте": "Запрещённый поворот / разворот",
    "Нарушение правил пользования световыми приборами, звуковыми сигналами": "Световые и звуковые приборы",
    "Выезд на полосу встречного движения или на трамвайные пути встречного направления": "Встречное движение",
}


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg with explicit preservation of untested cells."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if len(valid):
        q = valid.to_numpy() * len(valid) / np.arange(1, len(valid)+1)
        out.loc[valid.index] = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    return out


def family(name: str) -> str:
    if "скорости" in name:
        return "Скорость"
    if "стоянка" in name or "Остановка" in name:
        return "Остановка и стоянка"
    if "полос" in name or "разметки" in name or "обочине" in name or "Поворот" in name:
        return "Полосы, разметка и манёвры"
    if "светофора" in name or "стоп-линии" in name:
        return "Светофоры и стоп-линия"
    if "ремень" in name or "телефона" in name:
        return "Ремень и телефон"
    return "Прочие нарушения"


def prepare_events(fines: pd.DataFrame, clients: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    required = ["client_id", "bill_id", "bill_offence_date", "offence_short_statement"]
    if not set(required).issubset(fines):
        raise ValueError(f"Required fine fields: {required}")
    if clients.client_id.duplicated().any():
        raise ValueError("Duplicate client_id in cluster mapping")
    f = fines.copy()
    f["bill_offence_date"] = pd.to_datetime(f.bill_offence_date, errors="coerce", format="mixed")
    if f[required].isna().any().any():
        raise ValueError("Missing fine ID, client, date or offence category")
    f["category"] = f.offence_short_statement.str.replace(r"\s+", " ", regex=True).str.strip()
    key = ["client_id", "bill_offence_date", "category"]
    conflicts = f.groupby("bill_id")[key].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting repeated bill_id: {int(conflicts.sum())}")
    duplicate_count = int(f.bill_id.duplicated().sum())
    f = f.drop_duplicates("bill_id")
    f = f.loc[f.bill_offence_date.ge(start) & f.bill_offence_date.lt(end)].copy()
    merged = f.merge(clients[["client_id", "cluster_id", "cluster_label", "eligible_comparison"]], on="client_id", how="left", validate="many_to_one", indicator=True)
    unmapped = int(merged._merge.ne("both").sum())
    if unmapped:
        raise ValueError(f"{unmapped} fines have no cluster mapping")
    merged["month"] = merged.bill_offence_date.dt.to_period("M").astype(str)
    merged["family"] = merged.category.map(family)
    return merged.drop(columns="_merge"), {"duplicate_bills_removed": duplicate_count, "unmapped_events": unmapped}


def monthly_cube(events: pd.DataFrame, groups: list[int], months: list[str], categories: list[str]):
    index = pd.MultiIndex.from_product([groups, months, categories], names=["cluster_id", "month", "category"])
    count = events.groupby(["cluster_id", "month", "category"]).bill_id.nunique()
    people = events.groupby(["cluster_id", "month", "category"]).client_id.nunique()
    result = count.reindex(index, fill_value=0).rename("fine_count").to_frame()
    result["clients_with_fines"] = people.reindex(index, fill_value=0)
    return result.reset_index()


def category_changes(events, clients, start, crisis, end, column="category"):
    """Paired client rate difference. Groups 0/1 and zero baselines are untested.

    Tests remain exploratory: groups depend on baseline outcomes. FDR families:
    pooled categories separately, all positive-baseline group/category cells together.
    """
    cohort = clients.loc[clients.eligible_comparison].copy()
    ev = events.loc[events.eligible_comparison].copy()
    ev["period"] = np.where(ev.bill_offence_date.lt(crisis), "pre", "post")
    categories = sorted(events[column].unique())
    rows = []
    for group in [-1] + sorted(clients.cluster_id.unique().tolist()):
        ids = pd.Index(cohort.client_id if group == -1 else cohort.loc[cohort.cluster_id.eq(group), "client_id"])
        part = ev.loc[ev.client_id.isin(ids)]
        counts = part.groupby([column, "period", "client_id"]).bill_id.nunique()
        for category in categories:
            arrays = []
            for period in ["pre", "post"]:
                key = (category, period)
                if key in counts.index.droplevel(-1):
                    array = counts.loc[key].reindex(ids, fill_value=0).to_numpy(dtype=float)
                else:
                    array = np.zeros(len(ids))
                arrays.append(array)
            before, after = arrays
            dp, dq = (crisis-start).days, (end-crisis).days
            diff = (after/dq-before/dp)*30000  # fines / 1,000 clients / 30 days
            pre, post = before.sum(), after.sum()
            mean = float(diff.mean()) if len(ids) else np.nan
            se = float(diff.std(ddof=1)/math.sqrt(len(ids))) if len(ids)>1 else np.nan
            testable = group not in [0,1] and pre>=10 and post>=10 and (before>0).sum()>=5 and (after>0).sum()>=5 and len(ids)>=40
            p = float(ttest_1samp(diff, 0).pvalue) if testable and se>0 else np.nan
            critical = float(t.ppf(.975, len(ids)-1)) if len(ids)>1 else np.nan
            rr = float(post/dq/(pre/dp)) if pre>0 else np.nan
            rows.append({
                "cluster_id": group, "category": category, "clients_compared": len(ids),
                "pre_count": int(pre), "post_count": int(post), "pre_days": dp, "post_days": dq,
                "pre_clients_with_fines": int((before>0).sum()), "post_clients_with_fines": int((after>0).sum()),
                "pre_per_1000_clients_30d": float(before.mean()*30000/dp) if len(ids) else np.nan,
                "post_per_1000_clients_30d": float(after.mean()*30000/dq) if len(ids) else np.nan,
                "rate_ratio": rr, "change_pct": (rr-1)*100, "delta_per_1000_clients_30d": mean,
                "ci95_delta_low": mean-critical*se, "ci95_delta_high": mean+critical*se,
                "p_value": p, "baseline_zero": bool(pre == 0),
                "selection_zero_group": group in [0,1], "testable": bool(testable),
            })
    result = pd.DataFrame(rows)
    dangerous = result.category.str.contains(r"40-60|60-80|более чем на 80|красный сигнал|встречного движения", case=False, regex=True)
    result["zero_baseline_reason"] = ""
    result.loc[result.baseline_zero, "zero_baseline_reason"] = "Нет зарегистрированных событий в базовом окне"
    result.loc[result.baseline_zero & result.cluster_id.isin([0,1]), "zero_baseline_reason"] = "Ноль всех штрафов задан правилом группы"
    result.loc[result.baseline_zero & result.cluster_id.between(2,5) & dangerous, "zero_baseline_reason"] = "При таком штрафе до кризиса клиент попал бы в группу 6"
    result["q_value"] = np.nan
    for mask in [result.cluster_id.eq(-1), result.cluster_id.ge(2)]:
        result.loc[mask,"q_value"] = bh_adjust(result.loc[mask,"p_value"])
    result["sharp_growth"] = result.testable & result.q_value.lt(.05) & result.rate_ratio.ge(1.5) & result.delta_per_1000_clients_30d.ge(1) & result.post_clients_with_fines.ge(10)
    return result


def scan_anomalies(events, monthly, months):
    """Descriptive screen, not a causal or statistical anomaly detector."""
    rows = []
    for (group, category), block in monthly.groupby(["cluster_id", "category"]):
        block = block.set_index("month").reindex(months)
        for j in range(1,len(months)):
            before, after = int(block.iloc[j-1].fine_count), int(block.iloc[j].fine_count)
            ratio = (after/pd.Period(months[j]).days_in_month)/(before/pd.Period(months[j-1]).days_in_month) if before else np.nan
            if before>=10 and after>=20 and ratio>=2:
                part=events.loc[events.cluster_id.eq(group)&events.category.eq(category)&events.month.eq(months[j])]
                concentration=part.groupby("client_id").bill_id.nunique()
                rows.append({"cluster_id":int(group),"category":category,"month":months[j],"previous_count":before,"count":after,"day_adjusted_ratio":ratio,"clients":len(concentration),"largest_client_share":float(concentration.max()/after),"kind":"monthly_jump"})
    return pd.DataFrame(rows, columns=["cluster_id","category","month","previous_count","count","day_adjusted_ratio","clients","largest_client_share","kind"])


def plot_spec(spec, path, crisis_index):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12})
    fig, ax = plt.subplots(figsize=(14,7.4))
    x = np.arange(len(spec["months"]))
    bottom = np.zeros(len(x))
    grouped = spec.get("grouping") == "clustered"
    for i,series in enumerate(spec["series"]):
        vals = np.array(series["values"])
        if grouped:
            width=.8/len(spec["series"])
            ax.bar(x-.4+width*(i+.5),vals,width=width,label=series["name"],color=series["color"])
        else:
            ax.bar(x, vals, bottom=bottom, width=.68, label=series["name"], color=series["color"])
        bottom += vals
    ax.set_xticks(x, spec["month_labels"])
    ax.set_ylabel(spec.get("y_title","Число штрафов, шт."))
    maximum=max(max(s["values"]) for s in spec["series"]) if grouped else bottom.max()
    ax.set_ylim(0, max(float(maximum)*1.2,1))
    if not grouped:
        for xi, value in zip(x, bottom):
            label=f"{value:.1f}" if spec["kind"]=="category_focus" else f"{value:,.0f}".replace(","," ")
            ax.text(xi,value+max(bottom.max()*.015,.03),label,ha="center",fontsize=12)
    ax.axvline(crisis_index, color="#C04538", linestyle="--", linewidth=1.7)
    ax.text(crisis_index+.04, .97, "≈ начало кризиса\n"+spec["crisis_label"], color="#C04538", transform=ax.get_xaxis_transform(), va="top",fontsize=11)
    ax.set_title(spec["title"],loc="left",fontsize=18,pad=42)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    if len(spec["series"])>1:
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.11),ncol=3,frameon=False,fontsize=10)
    fig.text(.09,.93,spec["subtitle"],fontsize=11,color="#50616E")
    fig.text(.09,.015,spec["note"],fontsize=10,color="#50616E")
    fig.subplots_adjust(left=.09,right=.98,top=.83,bottom=.23 if len(spec["series"])>1 else .13)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def clean_json(value):
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,list): return [clean_json(v) for v in value]
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def run(args):
    root=args.data_dir.resolve()
    output=(args.output_dir or root/"outputs/monthly_offences").resolve()
    output.mkdir(parents=True,exist_ok=True)
    (output/"figures").mkdir(exist_ok=True)
    start,crisis,end=map(pd.Timestamp,[args.start,args.crisis_start,args.end])
    if not(start<crisis<end) or start.day!=1 or end.day!=1:
        raise ValueError("Use complete months and start < crisis < exclusive end")
    clients=pd.read_csv(args.clients_file or root/"outputs/behavior_cluster_analysis/client_behavior_features.csv")
    clients["subscription_creation_date"]=pd.to_datetime(clients.subscription_creation_date)
    # Recompute cohort from explicit analysis window, not serialized boolean strings.
    clients["eligible_comparison"]=clients.subscription_creation_date.le(start)
    fines=pd.read_csv(root/"data/processed/fines_clean.csv")
    if pd.to_datetime(fines.bill_offence_date).min().normalize()>start or pd.to_datetime(fines.bill_offence_date).max().normalize()<end-pd.Timedelta(days=1):
        raise ValueError("Requested period extends beyond fine-source date coverage")
    events,quality=prepare_events(fines,clients,start,end)
    months=pd.period_range(start,end-pd.Timedelta(days=1),freq="M").astype(str).tolist()
    groups=sorted(clients.cluster_id.unique().tolist())
    cats=sorted(events.category.unique())
    monthly=monthly_cube(events,groups,months,cats)
    totals=monthly.groupby(["cluster_id","month"],as_index=False).fine_count.sum()
    assert int(totals.fine_count.sum())==events.bill_id.nunique()
    fixed=events.loc[events.eligible_comparison].copy()
    fixed_monthly=monthly_cube(fixed,groups,months,cats)
    changes=category_changes(events,clients,start,crisis,end)
    families=category_changes(events,clients,start,crisis,end,column="family")
    anomalies=scan_anomalies(fixed,fixed_monthly,months)
    totals.to_csv(output/"monthly_totals.csv",index=False)
    monthly.to_csv(output/"monthly_categories.csv",index=False)
    fixed_monthly.to_csv(output/"monthly_categories_fixed_cohort.csv",index=False)
    changes.to_csv(output/"category_pre_post.csv",index=False)
    changes.loc[changes.baseline_zero & changes.post_count.gt(0)].to_csv(output/"category_appearances.csv",index=False)
    families.to_csv(output/"family_pre_post.csv",index=False)
    anomalies.to_csv(output/"monthly_anomalies.csv",index=False)
    mapping=pd.DataFrame({"category":cats,"short_label":[SHORT_NAMES.get(v,v) for v in cats],"family":[family(v) for v in cats]})
    mapping.to_csv(output/"category_dictionary.csv",index=False)
    shared=[]
    for cat,block in changes.loc[changes.cluster_id.ge(2)].groupby("category"):
        sharp=block.loc[block.sharp_growth,"cluster_id"].astype(int).tolist()
        positive=block.loc[block.rate_ratio.gt(1),"cluster_id"].astype(int).tolist()
        shared.append({"category":cat,"sharp_groups":sharp,"positive_groups":positive})
    specs=[]
    for group in groups:
        label=clients.loc[clients.cluster_id.eq(group),"cluster_label"].iloc[0]
        n=int(clients.cluster_id.eq(group).sum())
        block=monthly.loc[monthly.cluster_id.eq(group)]
        top=block.groupby("category").fine_count.sum().sort_values(ascending=False,kind="stable").head(args.top_n).index.tolist()
        counts=totals.loc[totals.cluster_id.eq(group)].set_index("month").fine_count.reindex(months).tolist()
        note="Все клиенты группы. Абсолютное число уникальных постановлений. Дата кризиса условная."
        if group in [0,1]: note="Ноль в апреле-мае задан правилом группы. Март отсутствует в очищенном источнике."
        common={"group":int(group),"group_label":label,"months":months,"month_labels":[MONTH_NAMES[pd.Period(m).month] for m in months],"crisis_label":crisis.strftime("%d.%m.%Y"),"subtitle":f"{label}. Клиентов: {n:,}".replace(","," "),"note":note,"all_clients":n}
        spec={**common,"title":f"Группа {group}: все категории штрафов по месяцам","kind":"total","series":[{"name":"Все категории штрафов","values":counts,"color":COLORS[0]}]}
        specs.append(spec)
        series=[]
        for j,category in enumerate(top):
            values=block.loc[block.category.eq(category)].set_index("month").fine_count.reindex(months,fill_value=0).tolist()
            series.append({"name":SHORT_NAMES.get(category,category),"full_name":category,"values":values,"color":CATEGORY_COLORS.get(category,COLORS[j])})
        other=(np.array(counts)-np.array([s["values"] for s in series]).sum(axis=0)).tolist()
        series.append({"name":"Остальные категории","values":other,"color":COLORS[-1]})
        assert np.array_equal(np.sum([s["values"] for s in series],axis=0),counts)
        specs.append({**common,"title":f"Группа {group}: топ-{args.top_n} категорий по месяцам","kind":"categories","series":series,"note":f"Топ-{args.top_n} по сумме за весь период, состав постоянный. "+note})
    # Additional category-specific charts show a fixed cohort. Equal days and
    # cohort sizes enter rates; main requested histograms remain absolute counts.
    focus=[("Использование телефона за рулем",[2,3,4,5,6]),("Пересечение стоп-линии",[2,3,4,5,6]),("Не пристегнут ремень безопасности",[2,3,4,5,6]),("Неоплаченный проезд по платной дороге",[4])]
    for k,(category,selected_groups) in enumerate(focus):
        series=[]
        for j,g in enumerate(selected_groups):
            n=int((clients.cluster_id.eq(g)&clients.eligible_comparison).sum())
            values=fixed_monthly.loc[fixed_monthly.cluster_id.eq(g)&fixed_monthly.category.eq(category)].set_index("month").fine_count.reindex(months).astype(float)
            rates=[float(values.loc[m]*30000/n/pd.Period(m).days_in_month) for m in months]
            series.append({"name":f"Группа {g} (n={n})","values":rates,"color":COLORS[g-2]})
        title=f"{SHORT_NAMES.get(category,category)}: по месяцам"
        subtitle=f"Постоянный состав клиентов, подключившихся не позднее {start:%d.%m.%Y}"
        if len(selected_groups)==1:
            title=f"Группа {selected_groups[0]}: штрафы за платную дорогу"
            subtitle=f"Неоплаченный проезд. Постоянный состав: {n:,} клиента".replace(","," ")
        specs.append({"group":f"focus_{k}","months":months,"month_labels":[MONTH_NAMES[pd.Period(m).month] for m in months],"crisis_label":crisis.strftime("%d.%m.%Y"),"title":title,"subtitle":subtitle,"kind":"category_focus","grouping":"clustered","series":series,"y_title":"Штрафов / 1 000 клиентов / 30 дней","note":"Сопоставимая частота. Группы 0 и 1 исключены: ноль в базовом периоде задан отбором."})
    crisis_index=months.index(crisis.to_period("M").strftime("%Y-%m"))-.5+(crisis.day-1)/crisis.days_in_month
    for i,spec in enumerate(specs,1):
        spec["file"]=f"figures/cluster_{spec['group']}_{spec['kind']}.png"
        plot_spec(spec,output/spec["file"],crisis_index)
    pooled=changes.loc[changes.cluster_id.eq(-1)].sort_values("delta_per_1000_clients_30d",ascending=False)
    sensitivity=[]
    for shift in [-14,0,14]:
        cut=crisis+pd.Timedelta(days=shift)
        if not start<cut<end: continue
        alternative=category_changes(events,clients,start,cut,end)
        alternative["crisis_date"]=str(cut.date())
        sensitivity.append(alternative)
    pd.concat(sensitivity,ignore_index=True).to_csv(output/"category_crisis_sensitivity.csv",index=False)
    result={"start":str(start.date()),"crisis":str(crisis.date()),"end_exclusive":str(end.date()),"crisis_index":crisis_index,"n_all":len(clients),"n_fixed":int(clients.eligible_comparison.sum()),"events":len(events),"categories":len(cats),"quality":quality,"chart_specs":specs,"shared":shared,"pooled":pooled.to_dict("records"),"changes":changes.to_dict("records"),"family_changes":families.to_dict("records"),"anomalies":anomalies.to_dict("records"),"monthly_fixed":fixed_monthly.to_dict("records"),"method":{"test":"two-sided paired client t-test on day-normalized counts, exploratory","fdr":"BH over testable categories pooled, separately over all testable cluster-category cells","sharp":"RR >= 1.5, q < .05, delta >= 1 per 1000 clients/30d, at least 10 post clients","zero_baseline":"No rate ratio or significance test. Groups 0/1 excluded from common-growth screen.","assumptions":"Missing event rows count as zero. Subscription date does not prove complete coverage. Baseline-defined groups have selection bias. No causal test."}}
    (output/"deck_data.json").write_text(json.dumps(clean_json(result),ensure_ascii=False,indent=2))
    print(f"Saved {len(specs)} histograms to {output}")
    print("Pooled category changes:")
    print(pooled[["category","pre_count","post_count","change_pct","q_value","sharp_growth"]].to_string(index=False))
    print("Shared sharp growth:",json.dumps([s for s in shared if len(s["sharp_groups"])>=2],ensure_ascii=False))
    print("Monthly jumps:",anomalies.to_string(index=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=PROJECT_ROOT)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--clients-file",type=Path)
    parser.add_argument("--start",default="2026-04-01")
    parser.add_argument("--crisis-start",default="2026-06-01")
    parser.add_argument("--end",default="2026-09-01",help="Exclusive end")
    parser.add_argument("--top-n",type=int,default=5,choices=range(1,6))
    run(parser.parse_args())


if __name__=="__main__":
    main()
