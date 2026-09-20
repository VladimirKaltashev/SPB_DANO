"""Descriptive pre/post comparisons; groups never enter the main FE models."""
import numpy as np
import pandas as pd


def summarize(data,between,centered,tables):
    pre=data.loc[data.month.isin(['2026-04','2026-05'])].groupby('client_id')[['fuel_liters','fines_count']].mean().add_prefix('pre_')
    post=data.loc[data.month.isin(['2026-06','2026-07','2026-08'])].groupby('client_id')[['fuel_liters','fines_count']].mean().add_prefix('post_')
    change=pre.join(post)
    change['fuel_change_abs']=change.post_fuel_liters-change.pre_fuel_liters
    change['fines_change_abs']=change.post_fines_count-change.pre_fines_count
    positive=change.pre_fuel_liters.gt(0)
    change['fuel_change_pct']=np.nan
    change.loc[positive,'fuel_change_pct']=change.loc[positive,'post_fuel_liters']/change.loc[positive,'pre_fuel_liters']-1
    q1,q2=change.loc[positive,'fuel_change_pct'].quantile([1/3,2/3])
    assert q1<q2
    change['change_group']='ZERO_PRE'
    change.loc[positive & change.fuel_change_pct.le(q1),'change_group']='STRONG_REDUCERS' if q1<0 else 'LOWEST_CHANGE'
    change.loc[positive & change.fuel_change_pct.gt(q1) & change.fuel_change_pct.le(q2),'change_group']='MIDDLE_CHANGE'
    change.loc[positive & change.fuel_change_pct.gt(q2),'change_group']='HIGHEST_CHANGE'
    correlations=[]
    for name,frame,left,right in [
        ('pre_post_absolute',change,'fuel_change_abs','fines_change_abs'),
        ('pre_post_relative_positive_pre',change.loc[positive],'fuel_change_pct','fines_change_abs'),
        ('between_client_means',between,'fuel_liters','fines_count'),
        ('within_client_only',centered,'fuel_client_demeaned','fines_client_demeaned'),
        ('within_client_and_month',centered,'fuel_twfe_residual','fines_twfe_residual')]:
        correlations.append(dict(relationship=name,N=len(frame),pearson=frame[left].corr(frame[right]),
            spearman=frame[left].corr(frame[right],method='spearman')))
    groups=change.groupby('change_group',as_index=False).agg(N=('pre_fuel_liters','size'),
        pre_fuel=('pre_fuel_liters','mean'),post_fuel=('post_fuel_liters','mean'),
        mean_individual_fuel_change_pct=('fuel_change_pct','mean'),median_individual_fuel_change_pct=('fuel_change_pct','median'),
        minimum_individual_fuel_change_pct=('fuel_change_pct','min'),maximum_individual_fuel_change_pct=('fuel_change_pct','max'),
        pre_fines=('pre_fines_count','mean'),post_fines=('post_fines_count','mean'),fines_change=('fines_change_abs','mean'))
    groups['fuel_change_pct_from_group_means']=np.where(groups.pre_fuel.gt(0),groups.post_fuel/groups.pre_fuel-1,np.nan)
    order=['STRONG_REDUCERS' if q1<0 else 'LOWEST_CHANGE','MIDDLE_CHANGE','HIGHEST_CHANGE','ZERO_PRE']
    groups=groups.set_index('change_group').reindex(order).reset_index()
    change.reset_index().to_csv(tables/'client_pre_post.csv',index=False)
    groups.to_csv(tables/'reducer_groups.csv',index=False)
    pd.DataFrame(correlations).to_csv(tables/'correlations.csv',index=False)
    zero=change.loc[~positive]
    audit=dict(tertile_q1=float(q1),tertile_q2=float(q2),positive_pre_clients=int(positive.sum()),zero_pre_clients=len(zero),
        zero_pre_post_buyers=int(zero.post_fuel_liters.gt(0).sum()),
        relative_change_definition='ratio of post 3-month average to pre 2-month average minus one; undefined for zero pre',
        group_percent_definition='report both mean individual percentage and ratio of group means; plot ratio of group means',
        max_individual_relative_change=float(change.fuel_change_pct.max()))
    return change,pd.DataFrame(correlations),groups,audit


def draw_figures(models,change,groups,folder,tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5.7))
    labels={'between':'Между клиентами: средние за 5 месяцев','within':'Тот же клиент + месяц','within_price':'Тот же клиент + месяц + PriceShock'}
    colors=['#386D9D','#B15D78','#687D49']
    for i,(model,color) in enumerate(zip(labels,colors)):
        r=models.loc[models.model.eq(model)&models.term.eq('fuel_liters_100')].iloc[0]
        ax.errorbar(r.beta,i,xerr=[[r.beta-r.ci95_low],[r.ci95_high-r.beta]],fmt='o',capsize=5,color=color,markersize=8)
        ax.annotate(f'{r.beta:+.3f}; p={r.p_value:.3g}',(r.beta,i),xytext=(0,15),textcoords='offset points',ha='center',fontsize=10)
    ax.axvline(0,color='#666',lw=1)
    ax.set_yticks(range(3),list(labels.values()));ax.set_ylim(2.6,-.6)
    ax.set_xlabel('Изменение штрафов на дополнительные 100 наблюдаемых литров; 95% ДИ')
    ax.set_title('Связь покупок топлива со штрафами: between и within')
    ax.grid(axis='x',alpha=.2)
    fig.text(.5,.025,'Линейные модели. Ошибки по 16 регионам. Покупки через сервис не равны пробегу.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,1]);fig.savefig(folder/'01_between_vs_within.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.8));saved=[]
    for ax,column,title,scale in [(axes[0],'fuel_change_abs','Абсолютное изменение покупок',1),(axes[1],'fuel_change_pct','Относительное изменение; PRE > 0',100)]:
        frame=change.loc[change[column].notna()].copy()
        frame['bin']=pd.qcut(frame[column],20,duplicates='drop')
        b=frame.groupby('bin',observed=True).agg(x=(column,'mean'),y=('fines_change_abs','mean'),N=(column,'size')).reset_index(drop=True)
        b['variable']=column;saved.append(b)
        ax.scatter(b.x*scale,b.y,s=48,c='#386D9D',edgecolor='white')
        stats_label=(f"Клиенты: r={frame[column].corr(frame.fines_change_abs):.3f}; "
                     f"ρ={frame[column].corr(frame.fines_change_abs,method='spearman'):.3f}; N={len(frame):,}")
        ax.axhline(0,c='#888',lw=.8);ax.axvline(0,c='#888',lw=.8)
        ax.set_title(title+'\n'+stats_label,fontsize=11);ax.set_xlabel('POST − PRE, л/месяц' if scale==1 else 'POST / PRE − 1, %')
        ax.set_ylabel('POST − PRE, штрафов/месяц');ax.grid(alpha=.18)
    pd.concat(saved,ignore_index=True).to_csv(tables/'change_scatter_bins.csv',index=False)
    fig.suptitle('Изменения покупок и штрафов одного клиента',fontsize=16)
    fig.text(.5,.025,'PRE: апрель–май; POST: июнь–август. Средние за месяц; точки — средние квантильных корзин, без удаления хвостов.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.93]);fig.savefig(folder/'02_fuel_change_vs_fines_change.png',dpi=180);plt.close(fig)
    g=groups.loc[groups.change_group.ne('ZERO_PRE')].copy()
    fig,axes=plt.subplots(1,2,figsize=(12,5.8));labels=['Сильное снижение','Умеренное снижение','Наименьшее снижение\nили рост']
    colors=['#A94E6D','#DC9D44','#4B86B3']
    for ax,column,title,scale in [(axes[0],'fuel_change_pct_from_group_means','Наблюдаемые покупки топлива',100),(axes[1],'fines_change','Наблюдаемые штрафы',1)]:
        values=g[column].to_numpy()*scale
        bars=ax.bar(range(3),values,color=colors,width=.6)
        ax.bar_label(bars,labels=[f'{v:+.1f}%' if scale==100 else f'{v:+.3f}' for v in values],padding=6,fontsize=12)
        ax.set_xticks(range(3),labels,fontsize=9);ax.set_title(title,fontsize=12)
        ax.set_ylabel('Изменение группового среднего, %' if scale==100 else 'POST − PRE, штрафов/клиент в месяц')
        ax.axhline(0,c='#666',lw=.8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.margins(y=.22)
    fig.suptitle('Терцили реализованного изменения топлива: только описание',fontsize=16)
    fig.text(.5,.025,'Группы определены после наблюдения изменения топлива, без использования штрафов. PRE > 0. Это не причинный тест.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.93]);fig.savefig(folder/'03_reducer_groups.png',dpi=180);plt.close(fig)
