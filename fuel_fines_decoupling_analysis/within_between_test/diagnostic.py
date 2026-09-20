"""One joint Mundlak within-between model and its equality test."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f,t

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
PREVIOUS=HERE.parent/'outputs'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def main():
    assert (OUT/'analysis_plan.json').exists()
    manifest=json.loads((OUT/'previous_files_manifest.json').read_text())
    meta=json.loads((PREVIOUS/'metadata.json').read_text())
    for source in meta['sources'].values():assert sha(source['path'])==source['sha256']
    source=PREVIOUS/'tables/client_month.csv'
    assert sha(source)==manifest[str(source.relative_to(ROOT))]
    data=pd.read_csv(source,dtype={'client_id':'string','region':'string'})
    assert len(data)==117335 and data.client_id.nunique()==23467
    assert not data.duplicated(['client_id','month']).any()
    assert data.groupby('client_id').month.nunique().eq(5).all()
    assert set(data.month)=={f'2026-{month:02d}' for month in range(4,9)}
    assert pd.to_datetime(data.subscription_creation_date).le('2026-04-01').all()
    assert data.groupby('client_id').region.nunique().eq(1).all() and data.region.nunique()==16
    assert np.allclose(data.fuel_liters_100,data.fuel_liters/100)
    data['mean_fuel']=data.groupby('client_id').fuel_liters_100.transform('mean')
    data['within_fuel']=data.fuel_liters_100-data.mean_fuel
    assert np.allclose(data.groupby('client_id').within_fuel.mean(),0,atol=1e-12)
    X=pd.DataFrame({'const':1.,'within_fuel':data.within_fuel,'mean_fuel':data.mean_fuel}).join(
        pd.get_dummies(data.month,prefix='month',drop_first=True,dtype=float))
    assert X.shape[1]==7 and np.linalg.matrix_rank(X)==7
    y=data.fines_count.to_numpy();xn=X.to_numpy()
    ols=sm.OLS(y,X).fit()
    contrast=np.zeros(len(X.columns));contrast[X.columns.get_loc('mean_fuel')]=1;contrast[X.columns.get_loc('within_fuel')]=-1
    coefficients=[];tests=[];covariances=[]
    for cluster,label in [('region','region'),('client_id','client')]:
        fit=ols.get_robustcov_results(cov_type='cluster',groups=data[cluster],
            use_correction=True,df_correction=True,use_t=True)
        ci=fit.conf_int();cov=fit.cov_params();g=data[cluster].nunique()
        for j,term in enumerate(X.columns):
            coefficients.append(dict(cluster=label,term=term,beta=fit.params[j],SE=fit.bse[j],
                p_value=fit.pvalues[j],ci95_low=ci[j,0],ci95_high=ci[j,1]))
        difference=float(contrast@fit.params);se=float(np.sqrt(contrast@cov@contrast));critical=t.ppf(.975,g-1)
        wald=fit.f_test(contrast[None,:]);w=float(np.asarray(wald.fvalue).item())
        assert np.isclose(w,(difference/se)**2)
        assert np.isclose(wald.pvalue,2*t.sf(abs(difference/se),g-1))
        tests.append(dict(cluster=label,beta_between=fit.params[2],beta_within=fit.params[1],
            difference=difference,SE_difference=se,ci95_low=difference-critical*se,ci95_high=difference+critical*se,
            Wald_F=w,df_numerator=1,df_denominator=g-1,p_value_two_sided=float(wald.pvalue),
            covariance_between_within=float(cov[2,1]),n_clusters=g,n_clients=data.client_id.nunique(),N=len(data),
            model_rank=X.shape[1],CR1_correction=g/(g-1)*(len(data)-1)/(len(data)-X.shape[1])))
        covariances.append(pd.DataFrame(cov,index=X.columns,columns=X.columns).assign(cluster=label).rename_axis('term').reset_index())
        # Independent sandwich and Wald calculation; do not combine separate-model SEs.
        b=np.linalg.lstsq(xn,y,rcond=None)[0];residual=y-xn@b
        codes,uniques=pd.factorize(data[cluster]);scores=np.zeros((len(uniques),xn.shape[1]))
        np.add.at(scores,codes,xn*residual[:,None])
        bread=np.linalg.inv(xn.T@xn)
        independent_cov=bread@(scores.T@scores)@bread*g/(g-1)*(len(data)-1)/(len(data)-X.shape[1])
        independent_se=np.sqrt(contrast@independent_cov@contrast)
        independent_w=(contrast@b/independent_se)**2
        assert np.allclose([independent_se,independent_w,f.sf(independent_w,1,g-1)],
            [se,w,float(wald.pvalue)],rtol=1e-8,atol=1e-12)
    coefficients=pd.DataFrame(coefficients);tests=pd.DataFrame(tests)
    coefficients.to_csv(OUT/'joint_coefficients.csv',index=False)
    tests.to_csv(OUT/'wald_equality_tests.csv',index=False)
    pd.concat(covariances,ignore_index=True).to_csv(OUT/'joint_covariance.csv',index=False)
    prior=pd.read_csv(PREVIOUS/'tables/model_results.csv')
    comparison=[]
    for model,term in [('between','mean_fuel'),('within','within_fuel')]:
        old=prior.loc[prior.model.eq(model)&prior.term.eq('fuel_liters_100')].iloc[0]
        new=coefficients.loc[coefficients.cluster.eq('region')&coefficients.term.eq(term)].iloc[0]
        comparison.append(dict(relationship=model,previous_beta=old.beta,joint_beta=new.beta,
            beta_difference=new.beta-old.beta,previous_SE=old.SE,joint_SE=new.SE,
            previous_parameter_rank=old.full_parameter_rank,joint_parameter_rank=X.shape[1]))
    comparison=pd.DataFrame(comparison);comparison.to_csv(OUT/'previous_estimates_comparison.csv',index=False)
    # Equality is a property of this balanced design, checked here, not an estimation constraint.
    reproduced=bool(np.allclose(comparison.previous_beta,comparison.joint_beta,atol=1e-10,rtol=1e-10))
    r=tests.loc[tests.cluster.eq('region')].iloc[0];a=tests.loc[tests.cluster.eq('client')].iloc[0]
    ratio=r.beta_between/r.beta_within;percent=100*(1-r.beta_within/r.beta_between)
    supported=bool(r.ci95_low>0 and r.p_value_two_sided<.05)
    statement=(f'В исследуемой выборке связь наблюдаемых покупок топлива через сервис с числом штрафов '
        f'статистически сильнее между разными клиентами, чем при месячных изменениях покупок одного клиента '
        f'после учёта общих месячных эффектов: {r.beta_between:.3f} против {r.beta_within:.3f} штрафа на 100 л. '
        f'Разница составляет {r.difference:.3f} штрафа на 100 л '
        f'(95% ДИ [{r.ci95_low:.3f}; {r.ci95_high:.3f}], p={r.p_value_two_sided:.4g}; ошибки по регионам). '
        'Это статистическая ассоциация; наблюдаемые покупки не равны пробегу и не устанавливают причинный эффект.')
    dump('verdict.json',dict(between_greater_than_within=supported,point_estimate_ratio=float(ratio),
        within_percent_smaller=float(percent),coefficients_reproduce_previous=reproduced,
        presentation_statement=statement,stage2_executed=False))
    table='| Кластеризация | β between | β within | Разница | SE разницы | 95% ДИ разницы | Wald F | df | p, двустороннее |\n|---|---:|---:|---:|---:|---|---:|---|---:|\n'
    for q in tests.itertuples():
        table+=f'| {q.cluster} | {q.beta_between:.6f} | {q.beta_within:.6f} | {q.difference:.6f} | {q.SE_difference:.6f} | [{q.ci95_low:.6f}; {q.ci95_high:.6f}] | {q.Wald_F:.6f} | 1; {q.df_denominator} | {q.p_value_two_sided:.8g} |\n'
    report=f'''# Формальный тест within–between: Mundlak decomposition

{table}

## Модель и тест

Та же очищенная сбалансированная когорта: 23 467 клиентов × 5 месяцев = 117 335 наблюдений, апрель–август 2026; subscription_creation_date ≤ 1 апреля. Никаких исключений по покупкам, штрафам или автомобильным характеристикам не добавлено. Проверены исходные SHA-256 и тождество FuelLiters100 = FuelLiters/100.

mean_fuel_i = среднее FuelLiters100 клиента за пять месяцев; within_fuel_it = FuelLiters100_it − mean_fuel_i. Одна совместная модель:

Fines_it = const + β_within × within_fuel_it + β_between × mean_fuel_i + MonthFE + error.

Использована OLS within–between параметризация Mundlak с кластерной ковариацией, без отдельного GLS-предположения о распределении random intercept. Это именно приведённая в задании совместная спецификация. Явные Client FE не добавляются: они поглотили бы mean_fuel_i. Клиентские постоянные компоненты ошибки допускаются кластерной структурой.

H0: β_between − β_within = 0; H1: разность не равна нулю. SE разности = sqrt(Var(β_between) + Var(β_within) − 2 Cov(β_between,β_within)). В основном расчёте ковариация коэффициентов = {r.covariance_between_within:.9g}; она взята из одной модели, а не положена равной нулю.

Основная кластеризация по 16 регионам, CR1 = G/(G−1) × (N−1)/(N−K), K=7; 95% ДИ на t(15). Wald F(1,15) = t_difference²; показанный p-value двусторонний и вычислен по F, не по асимптотическому χ². При одном ограничении численное значение квадратичного Wald равно F, но референсное распределение существенно. 16 кластеров остаются ограничением точности.

Robustness — тот же коэффициентный вектор и контраст, ошибки по 23 467 клиентам, t(23 466) / F(1,23 466). Эта альтернатива слабее защищает от общих региональных ошибок; основной вывод опирается на региональный вариант.

## Воспроизведение отдельных оценок

Совпадение с прежними коэффициентами с точностью 1e−10: **{reproduced}**. Между клиентами: {r.beta_between:.12f}; внутри клиента: {r.beta_within:.12f}. Ограничений, принудительно задающих старые коэффициенты, в модели нет.

Совпадение ожидаемо именно при пяти одинаковых месяцах для каждого клиента: клиентское среднее ортогонально внутриклиентским отклонениям, а общие месячные индикаторы сбалансированы между клиентами. Within-часть соответствует прежней оценке Client FE + Month FE; between — регрессии клиентских средних. При несбалансированной панели это тождество нельзя автоматически переносить.

SE не обязаны буквально совпадать со старыми отдельными регрессиями. Ранее within-OLS считала полное число поглощённых параметров K=23 472 в CR1, теперь оценивается совместная модель с K=7. Использован корректный ранг новой модели, а не искусственно перенесённый размер прежней модели. Основной способ кластеризации, t-инференс и уровень ДИ сохранены; прежние SE не перезаписаны. Сравнение beta/SE/rank сохранено в previous_estimates_comparison.csv.

## Ответы

1. Статистически ли between > within? **{'YES' if supported else 'INCONCLUSIVE'}**. Основной двусторонний тест равенства: p={r.p_value_two_sided:.8g}; ДИ разницы [{r.ci95_low:.6f}; {r.ci95_high:.6f}]. Клиентская кластеризация: p={a.p_value_two_sided:.8g}.
2. Разница point estimates = **{r.difference:.6f} штрафа на 100 л**. Between / within = **{ratio:.3f}**; within на **{percent:.2f}% меньше** between. Разница — результат теста; отношение 2,57 и 61% — описательные точечные оценки без отдельного ДИ отношения.
3. Можно ли писать «существенно сильнее»? {'Да, в смысле статистически более сильной связи в этой выборке и спецификации; лучше заменить «существенно» на «статистически», чтобы не заявлять неустановленную практическую или причинную значимость.' if supported else 'Нет: основной тест не установил положительную разницу.'}
4. Точная формулировка:

> {statement}

Эта проверка не делает within-связь нулевой и не повышает её объясняющую способность: она формально проверяет только разницу наклонов. Вывод относится к наблюдаемым сервисным покупкам и штрафам, а не к доказанной разнице эффекта пробега.

## Проверки и файлы

Коэффициенты, SE разности и Wald перепроверены независимым NumPy-расчётом OLS и регионального/клиентского sandwich. Сохранены joint_coefficients.csv, joint_covariance.csv, wald_equality_tests.csv, previous_estimates_comparison.csv. Старые файлы проверяются по перечню и SHA-256. Дополнительных моделей исходов и Stage 2 нет.

Запуск из корня проекта: `.venv/bin/python fuel_fines_decoupling_analysis/within_between_test/diagnostic.py`.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    current={str(p.relative_to(ROOT)):sha(p) for d in ['archive_monthly_risk_clustering','fines_per_liter_analysis','usage_composition_analysis','vehicle_sensitivity_analysis','fuel_fines_decoupling_analysis']
        for p in (ROOT/d).rglob('*') if p.is_file() and HERE not in p.parents}
    assert current==manifest
    dump('validation.json',dict(status='PASS',unchanged_previous_files=len(manifest),source_hashes_verified=len(meta['sources']),
        balanced_cohort_verified=True,previous_coefficients_reproduced=reproduced,
        independent_OLS_sandwich_and_Wald='PASS for region and client',code_sha256=sha(Path(__file__))))
    print(tests.to_string(index=False))
    print(comparison.to_string(index=False))
    print(statement)


if __name__=='__main__':main()
