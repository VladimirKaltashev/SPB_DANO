"""Six prespecified margin models; no fines models or automatic Stage 2.

Run from project root: .venv/bin/python vehicle_sensitivity_analysis/margins_diagnostic.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = Path(__file__).resolve().parent / "outputs"
OUT = STAGE1 / "margins"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def verify_stage1():
    manifest = json.loads((OUT / "stage1_manifest.json").read_text())
    for path, digest in manifest.items():
        assert sha(ROOT / path) == digest, path
    return len(manifest)


def write_report(results, contrasts, audit):
    def table(frame):
        def fmt(value):
            if pd.isna(value):
                return '—'
            return f'{value:.6g}' if isinstance(value, (float, np.floating)) else str(value).replace('|','/')
        return '\n'.join(['| '+' | '.join(frame.columns)+' |',
            '| '+' | '.join(['---']*len(frame.columns))+' |',
            *['| '+' | '.join(map(fmt, row))+' |' for row in frame.itertuples(index=False,name=None)]])
    labels={'fuel_liters':'FuelLiters','log_fuel':'log(1 + FuelLiters)',
            'fuel_purchase':'P(FuelLiters > 0)','log_positive_fuel':'log(FuelLiters), purchasers'}
    comparison=results[['margin','outcome','vehicle_variable','interaction_beta','SE','p_value','ci95_low','ci95_high','N']].copy()
    comparison['outcome']=comparison.outcome.map(labels)
    effects=results.loc[results.margin.ne('ALL'),['margin','outcome','vehicle_variable',
        'interaction_per_10pp_shock','effect_10pp_ci95_low','effect_10pp_ci95_high','p_value']].copy()
    effects['unit']='liters or log units'
    ext=effects.margin.eq('EXTENSIVE')
    effects.loc[ext,['interaction_per_10pp_shock','effect_10pp_ci95_low','effect_10pp_ci95_high']]*=100
    effects.loc[ext,'unit']='percentage points of probability'
    ext_slopes=pd.DataFrame(contrasts).query("margin == 'EXTENSIVE'").copy()
    for col in ['change_per_10pp_shock','SE','ci95_low','ci95_high']:
        ext_slopes[col]*=100
    markdown=f'''# Короткая диагностика extensive / intensive margins

## Сопоставление 10 оценок

{table(comparison)}

Interaction — PriceShock × характеристика в единицах исходной SD Stage 1. PriceShock записан долей: 0,10 = +10 п.п. Все модели содержат const, PriceShock, характеристику, interaction, Region FE и Month FE. Ошибки сгруппированы по 16 регионам с finite-sample correction и t(15); число кластеров небольшое. Доверительные интервалы 95%, p-value номинальные.

Четыре строки ALL прочитаны из неизменённого Stage 1. Оценены только шесть новых моделей: два LPM на всех наблюдениях и четыре модели среди покупателей. Ни штрафные модели, ни Stage 2 не выполнялись.

## Масштаб interaction: +10 п.п. PriceShock и +1 SD характеристики

{table(effects)}

Это **разница ценовых наклонов между характеристиками**, а не сам ценовой наклон. Положительный interaction LPM означает более слабое снижение вероятности покупки в показанных ниже точках; он не означает, что вероятность сама по себе растёт.

### Изменение вероятности покупки при +10 п.п. PriceShock

Все изменения и ДИ в следующей таблице — процентные пункты вероятности. Характеристика берётся на −1 SD / среднем / +1 SD; SE и ДИ учитывают ковариацию главного эффекта цены и interaction.

{table(ext_slopes[['vehicle_variable','vehicle_z','vehicle_value','change_per_10pp_shock','SE','ci95_low','ci95_high']])}

Аналогичные условные изменения литров и log(литры) среди покупателей сохранены в marginal_effects.csv. Для log(литры) это логарифмические единицы, не литры и не автоматически процентное изменение арифметического среднего.

## Диагноз: C — оба направления, с противоположными знаками

1. **Extensive:** положительный interaction значим для двигателя и мощности. При большей характеристике вероятность наблюдать хотя бы одну покупку снижается слабее. Этот результат согласован между двумя proxy в пределах проведённой диагностики.
2. **Intensive в литрах:** среди наблюдаемых покупателей больший двигатель и мощность связаны с более отрицательным ценовым наклоном объёма. В обеих моделях interaction значим. Это не меньшая чувствительность объёма среди покупателей.
3. **Intensive в логарифме:** для двигателя знак отрицательный, но p≈0,107; для мощности отрицательный и значимый. Устойчивая относительная неоднородность среди покупателей именно по объёму двигателя не установлена.

**Положительный interaction в log(1+x) прежде всего согласуется с extensive margin (A): сохранением ненулевых покупок.** После исключения нулей положительный знак исчезает у обеих характеристик. Среди покупателей связь объёма с характеристикой направлена в противоположную сторону; поэтому слабый итоговый interaction в литрах на всей выборке совместим с взаимным ослаблением этих двух связей.

Это диагностическое объяснение, не численное разложение коэффициента на проценты вклада. Для L≥0 верны тождества E[L|X] = P(L>0|X) × E[L|L>0,X] и E[log(1+L)|X] = P(L>0|X) × E[log(1+L)|L>0,X]. Их произведения, изменение состава покупателей и разные преобразования исхода означают, что коэффициенты наших отдельных OLS нельзя просто складывать. Условная модель здесь использует log(L), а не log(1+L), поэтому дополнительного точного тождества для её коэффициента также не заявляем. Log(1+L) сжимает большие покупки и иначе учитывает переходы между нулём и положительным объёмом.

**Решение:** широкая формулировка «большие двигатели слабее сокращают объём топлива» не поддерживается. Узкую гипотезу о неоднородности вероятности наблюдаемой покупки закрывать как пустую не следует: здесь есть ясный согласованный сигнал по двигателю и мощности. Полный механизм изменения объёма остаётся ограниченным условным отбором и неустойчивым относительным interaction двигателя. Автоматического продолжения нет: Stage 2 не запущен; никаких новых спецификаций после этой диагностики не добавлено.

## Выборки, очистка и ограничения

- Вся выборка: {audit['all_clients']:,} клиента, {audit['N_all']:,} клиент-месяцев.
- Положительные покупки: {audit['observed_purchaser_clients']:,} клиент хотя бы с одной покупкой, {audit['N_positive']:,} клиент-месяца.
- Нули: {audit['N_zero']:,} клиент-месяцев ({audit['share_zero']:.2%}).
- Обе выборки сохраняют все 16 регионов и месяцы апрель–август 2026. Стандартизация двигателя и мощности не пересчитывалась среди покупателей.
- Источники — та же очищенная и сверенная с v2 панель Stage 1; SHA-256 исходных таблиц, ранее сохранённый PriceShock и исходные z проверены. Сохранены прежние правила для нескольких машин и подозрительных экстремальных характеристик, новых cutoff нет.
- Условная выборка L>0 — только observed fuel purchasers в данном месяце. Её состав может зависеть от цены и характеристик; это не оценка для всех водителей и не выделенный причинный intensive effect одних и тех же людей.
- Предсказания обоих LPM на фактических наблюдениях лежат внутри [0,1]; аудит в lpm_prediction_audit.csv. Это не гарантирует корректности линейной модели за пределами данных.
- Покупки в сервисе не равны поездкам или пробегу. Интерпретации страхового риска и новых выводов по штрафам нет.

## Воспроизведение

Из корня проекта: `.venv/bin/python vehicle_sensitivity_analysis/margins_diagnostic.py`.
План сохранён в analysis_plan.json до оценки моделей; шесть новых interaction и кластерных SE перепроверены независимым NumPy-расчётом. Результаты Stage 1 не изменены.
'''
    (OUT/'diagnostic_summary.md').write_text(markdown,encoding='utf-8')
    dump('diagnosis.json',dict(answer='C',positive_log1p_signal='primarily consistent with extensive margin A',
        engine_relative_intensive='INCONCLUSIVE',broad_weaker_volume_response='NOT SUPPORTED',
        narrow_purchase_probability_signal='SUPPORTED within this diagnostic',stage_2_executed=False,
        limitation='conditional purchaser selection; not an exact additive or causal decomposition'))


def main():
    assert (OUT / "analysis_plan.json").exists()
    protected = verify_stage1()
    metadata = json.loads((STAGE1 / "metadata.json").read_text())
    for source in metadata['sources'].values():
        assert sha(source['path']) == source['sha256']
    data = pd.read_csv(STAGE1 / 'diagnostic_client_month.csv',
        dtype={'client_id': 'string', 'region': 'string'})
    assert not data.duplicated(['client_id', 'month']).any()
    assert len(data) == metadata['analytical_client_months']
    assert data.fuel_liters.ge(0).all() and data.region.nunique() == 16
    assert set(data.month) == set(metadata['months'])
    standard = pd.read_csv(STAGE1 / 'standardization.csv').set_index('variable')
    for variable in ['engine_volume', 'horsepower']:
        assert np.allclose(data[variable+'_z'],
            (data[variable]-standard.loc[variable, 'mean'])/standard.loc[variable, 'std_ddof0'])
    data['fuel_purchase'] = data.fuel_liters.gt(0).astype(float)
    purchasers = data.loc[data.fuel_purchase.eq(1)].copy().reset_index(drop=True)
    purchasers['log_positive_fuel'] = np.log(purchasers.fuel_liters)
    previous = pd.read_csv(STAGE1 / 'diagnostic_results.csv')
    previous = previous.loc[previous.outcome.isin(['fuel_liters', 'log_fuel'])].copy()
    previous['margin'] = 'ALL'
    previous['estimate_origin'] = 'reused_stage1'
    rows, contrasts, terms, validations, lpm_audit = [], [], [], [], []
    for margin, sample, outcomes in [('EXTENSIVE', data, ['fuel_purchase']),
                                     ('INTENSIVE', purchasers, ['fuel_liters', 'log_positive_fuel'])]:
        fe = pd.get_dummies(sample[['region', 'month']], drop_first=True, dtype=float)
        for variable in ['engine_volume', 'horsepower']:
            z = sample[variable+'_z']
            X = pd.DataFrame({'const': 1., 'price_shock': sample.price_shock,
                'vehicle_z': z, 'interaction': sample.price_shock*z}).join(fe)
            xn = X.to_numpy()
            assert np.isfinite(xn).all() and np.linalg.matrix_rank(xn) == xn.shape[1]
            for outcome in outcomes:
                ols = sm.OLS(sample[outcome], X).fit()
                fit = ols.get_robustcov_results(cov_type='cluster', groups=sample.region,
                    use_correction=True, df_correction=True, use_t=True)
                ci = fit.conf_int(); j = X.columns.get_loc('interaction')
                rows.append(dict(margin=margin, outcome=outcome, vehicle_variable=variable,
                    interaction_beta=fit.params[j], SE=fit.bse[j], p_value=fit.pvalues[j],
                    ci95_low=ci[j,0], ci95_high=ci[j,1], N=int(ols.nobs),
                    n_clients=sample.client_id.nunique(), n_regions=sample.region.nunique(),
                    R_squared=ols.rsquared, sign='positive' if fit.params[j]>0 else 'negative',
                    interaction_per_10pp_shock=fit.params[j]*.1, estimate_origin='new_margin_model'))
                for i, term in enumerate(X.columns):
                    terms.append(dict(margin=margin, outcome=outcome, vehicle_variable=variable,
                        term=term, beta=fit.params[i], SE=fit.bse[i], p_value=fit.pvalues[i],
                        ci95_low=ci[i,0], ci95_high=ci[i,1]))
                for value in [-1, 0, 1]:
                    contrast = np.zeros(X.shape[1]); contrast[1] = .1; contrast[j] = .1*value
                    effect = contrast @ fit.params
                    se = np.sqrt(contrast @ fit.cov_params() @ contrast)
                    half = t.ppf(.975, sample.region.nunique()-1)*se
                    contrasts.append(dict(margin=margin, outcome=outcome, vehicle_variable=variable,
                        vehicle_z=value, vehicle_value=standard.loc[variable,'mean']+value*standard.loc[variable,'std_ddof0'],
                        change_per_10pp_shock=effect, SE=se, ci95_low=effect-half, ci95_high=effect+half,
                        probability_change_pp=effect*100 if margin=='EXTENSIVE' else np.nan))
                # Verify the interaction with a separately computed cluster sandwich and t(G-1).
                b = np.linalg.lstsq(xn, sample[outcome].to_numpy(), rcond=None)[0]
                residual = sample[outcome].to_numpy()-xn@b
                score = pd.DataFrame(xn*residual[:,None]).groupby(sample.region.to_numpy()).sum().to_numpy()
                bread = np.linalg.inv(xn.T@xn); g,n,k=len(score),len(sample),X.shape[1]
                cov = bread@(score.T@score)@bread*g/(g-1)*(n-1)/(n-k)
                se = np.sqrt(cov[j,j]); pv = 2*t.sf(abs(b[j]/se),g-1)
                assert np.allclose([b[j],se,pv],[fit.params[j],fit.bse[j],fit.pvalues[j]],rtol=1e-7,atol=1e-9)
                validations.append(dict(margin=margin,outcome=outcome,variable=variable,independent_inference='PASS'))
                if margin=='EXTENSIVE':
                    predicted=ols.predict(X)
                    lpm_audit.append(dict(variable=variable,min_fitted_probability=predicted.min(),
                        max_fitted_probability=predicted.max(),share_fitted_outside_0_1=((predicted<0)|(predicted>1)).mean()))
                print(f'{margin} {outcome} {variable}: beta={fit.params[j]:.8g}, SE={fit.bse[j]:.8g}, p={fit.pvalues[j]:.8g}',flush=True)
    results = pd.concat([previous,pd.DataFrame(rows)],ignore_index=True)
    results['effect_10pp_ci95_low'] = results.ci95_low*.1
    results['effect_10pp_ci95_high'] = results.ci95_high*.1
    results['interaction_10pp_probability_pp'] = np.where(results.margin.eq('EXTENSIVE'),results.interaction_beta*10,np.nan)
    results.to_csv(OUT/'margin_comparison.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'marginal_effects.csv',index=False)
    pd.DataFrame(terms).to_csv(OUT/'new_model_coefficients.csv',index=False)
    pd.DataFrame(lpm_audit).to_csv(OUT/'lpm_prediction_audit.csv',index=False)
    data.groupby('month',as_index=False).agg(n_client_months=('client_id','size'),
        n_purchasers=('fuel_purchase','sum'),share_purchase=('fuel_purchase','mean')).to_csv(OUT/'sample_by_month.csv',index=False)
    audit=dict(N_all=len(data),N_positive=len(purchasers),N_zero=int(data.fuel_purchase.eq(0).sum()),
        share_zero=float(data.fuel_purchase.eq(0).mean()),all_clients=data.client_id.nunique(),
        observed_purchaser_clients=purchasers.client_id.nunique(),regions=data.region.nunique(),
        positive_regions=purchasers.region.nunique(),months=sorted(data.month.unique()),
        mean_positive_liters=float(purchasers.fuel_liters.mean()),standardization='unchanged Stage 1 client-wise mean and SD',
        six_new_models_only=True,stage_2_executed=False,fines_analyzed=False)
    dump('sample_audit.json',audit)
    write_report(results,contrasts,audit)
    assert len(rows)==6 and len(results)==10 and verify_stage1()==protected
    dump('validation.json',dict(status='PASS',protected_stage1_files=protected,source_hashes_verified=len(metadata['sources']),
        independent_inference=validations,stage1_sample_and_z_preserved=True,code_sha256=sha(Path(__file__))))
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    print(results[['margin','outcome','vehicle_variable','interaction_beta','SE','p_value','ci95_low','ci95_high','N']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
