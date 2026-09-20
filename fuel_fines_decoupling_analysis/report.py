"""Report associations and their limited explanatory power without a causal story."""
from html import escape
import json

import numpy as np
import pandas as pd


def md_table(frame):
    def fmt(value):
        if pd.isna(value):return '—'
        return f'{value:.6g}' if isinstance(value,(float,np.floating)) else str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(map(fmt,r))+' |' for r in frame.itertuples(index=False,name=None)]])


def write_report(out):
    tables=out/'tables'
    read=lambda n:pd.read_csv(tables/f'{n}.csv')
    audit=json.loads((out/'data_audit.json').read_text())
    identification=json.loads((out/'identification_audit.json').read_text())
    group_meta=json.loads((out/'group_definition.json').read_text())
    poisson_meta=json.loads((out/'poisson_audit.json').read_text())
    results=read('model_results');alternative=read('alternative_client_clustering')
    groups=read('reducer_groups');correlations=read('correlations');zeros=read('zero_purchase_audit')
    pick=lambda model,term:results.loc[results.model.eq(model)&results.term.eq(term)].iloc[0]
    between=pick('between','fuel_liters_100');within=pick('within','fuel_liters_100')
    controlled=pick('within_price','fuel_liters_100');price=pick('within_price','price_shock')
    extensive=pick('extensive','has_fuel_purchase');poisson=pick('conditional_poisson','fuel_liters_100')
    scale=within.beta/between.beta;mean=audit['fines_distribution']['mean']
    selected=results.loc[results.term.isin(['fuel_liters_100','has_fuel_purchase'])].copy()
    interpretations={
        'between':f'+{between.beta:.3f} штрафа/месяц при различии средних покупок на 100 л между клиентами',
        'within':f'+{within.beta:.3f} штрафа/месяц на +100 л у того же клиента, после Month FE',
        'within_price':f'+{controlled.beta:.3f} штрафа/месяц на +100 л после учёта PriceShock',
        'extensive':f'+{extensive.beta:.3f} штрафа в месяц с покупкой против месяца без покупки, внутри клиента',
        'conditional_poisson':f'log-rate coefficient; IRR={poisson.IRR_per_100_liters:.4f} на +100 л, не аддитивное число штрафов'}
    selected['interpretation']=selected.model.map(interpretations)
    selected=selected[['model','relationship','beta','SE','p_value','ci95_low','ci95_high','n_clients','N','interpretation']]
    selected.to_csv(tables/'main_results_table.csv',index=False)
    group_show=groups[['change_group','N','pre_fuel','post_fuel','fuel_change_pct_from_group_means',
        'mean_individual_fuel_change_pct','pre_fines','post_fines','fines_change']].copy()
    group_show['fines_change_pct']=group_show.post_fines/group_show.pre_fines-1
    strong=groups.loc[groups.change_group.eq('STRONG_REDUCERS')].iloc[0]
    strong_pct=strong.post_fines/strong.pre_fines-1
    zero_row=zeros.loc[zeros.has_fuel_purchase.eq(0)].iloc[0]
    verdict={
        'between':'YES','within':'YES','within_to_between_point_estimate_ratio':float(scale),
        'within_percent_smaller':float(100*(1-scale)),'after_month_FE_and_PriceShock':'YES',
        'strong_reducers_fines_drop_more':'YES, descriptive only',
        'explains_aggregate_fuel_fines_gap':'PARTIALLY, compatible but not an identified explanation',
        'insurance_analytical_story':'YES, measurement and within-vs-between distinction; no causal risk or tariff claim',
        'diagnosis':'positive statistically distinguishable within association, smaller than between and low incremental explanatory power',
        'stage':1,'further_analysis_executed':False}
    (out/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2),encoding='utf-8')
    narrative=f'''# Fuel–fines decoupling: диагностический Stage 1

## Главный результат

**Полного разрыва внутри клиента не обнаружено.** Связь положительная и статистически различима: +100 наблюдаемых литров у того же клиента связаны с **+{within.beta:.4f} штрафа/месяц**, p={within.p_value:.6g}, после клиентских и месячных эффектов. Между клиентами оценка +{between.beta:.4f}; within по точечным оценкам на {100*(1-scale):.1f}% меньше (в {1/scale:.2f} раза). Это сравнение оценок, не отдельный формальный тест разности коэффициентов.

В то же время покупки объясняют лишь **{within.R_squared*100:.3f}% вариации штрафов, оставшейся после Client FE и Month FE**. Статистическая значимость не превращает наблюдаемые покупки в точный показатель пробега или сильный индивидуальный предиктор. Нельзя описывать этот результат как отсутствие within-связи, но и история о сильной объясняющей способности покупок не подтверждается.

## Главная таблица

{md_table(selected)}

В первых трёх строках β измеряется в штрафах/месяц на 100 наблюдаемых литров; в extensive — разница месяца с покупкой и без неё. В Poisson β — логарифмический коэффициент: IRR на +100 л = {poisson.IRR_per_100_liters:.4f}, 95% ДИ [{poisson.IRR_ci95_low:.4f}; {poisson.IRR_ci95_high:.4f}]. Нельзя сравнивать 0,095 Poisson с 0,066 OLS как одинаковые аддитивные эффекты.

## Выборка и проверка очищенных источников

Основная выборка: **{audit['eligible_clients']:,} клиентов, {audit['eligible_observations']:,} клиент-месяцев, {audit['regions']} регионов, апрель–август 2026**. У всех subscription_creation_date ≤ 2026-04-01. Из полной панели {audit['all_clients']:,} клиентов исключены {audit['excluded_late_or_missing_subscription']:,} с более поздней датой; пропусков даты — {audit['missing_subscription_clients']}. {audit['missing_engine_eligible_clients']} клиентов с отсутствующим engine_volume оставлены: характеристики автомобиля не нужны основным моделям.

Источники: fuel_clean.csv с physical_fuel_volume_main, fines_clean.csv с датой нарушения, clients_demographics_clean.csv. Все SHA-256 совпали с прошлыми анализами. Суммы топлива и число штрафов повторно восстановлены из очищенных операций по каждому клиенту × календарному месяцу и совпали с прежней полной панелью; таблица сверки source_reconciliation.csv. Исходный PriceShock сохранён: региональная средневзвешенная цена / апрельская цена региона − 1.

В v2 подтверждены все {audit['covered_days_per_client']} календарных дня на клиента, непрерывные недельные интервалы, уникальные client-week и пять client-month. Ноль ставится при отсутствии **зарегистрированного очищенного события** в этом месяце. Это корректный ноль для наблюдаемого сервиса, но полноту регистрации всех реальных заправок или поездок мы подтвердить не можем. Сама сбалансированная сетка не доказывает отсутствие пропусков наблюдения вне сервиса. Дата подписки — консервативный критерий включения, не доказательство фактической активности каждый день.

Аналитические суммы: {audit['eligible_total_liters']:,.2f} л, {audit['eligible_total_fines']:,} штраф. Доля нулевых покупок {audit['share_zero_fuel']:.2%}; доля нулевых штрафов {audit['fines_distribution']['share_0']:.2%}. Данные о подписке, двигателе, мощности и истории штрафов сохранены, но в модели автоматически не включены. История по автомобилям сохранена отдельно; клиентское значение оставлено только при единственном непротиворечивом известном значении по его автомобилям, иначе NaN + флаг. Никакого суммирования неоднозначной истории нескольких машин нет.

{md_table(read('source_reconciliation'))}

### Распределение штрафов

{md_table(read('fines_distribution'))}

Высокая доля нулей и дисперсия выше среднего не скрываются. Главная прозрачная модель — линейная FE; ровно одна заранее определённая count-robustness — conditional Poisson. По типам штрафов модели не оценивались.

## Как оценены within-модели

Главная модель: Fines_it = α_i + β × (FuelLiters_it / 100) + MonthFE_t + ε_it. Вторая добавляет PriceShock_rt. Extensive заменяет литры на HasFuelPurchase_it. Region FE не добавлены: регион клиента постоянен и поглощён Client FE.

В сбалансированной панели применяется точное двойное вычитание средних: x_it − x̄_i − x̄_t + x̄. Это эквивалентно OLS с явными индикаторами клиентов и месяцев. Остатки имеют нулевые средние по клиенту и месяцу. {audit['all_zero_fines_clients']:,} клиентов без единого штрафа оставлены в линейных моделях. Для extensive изменения 0↔1 есть у {audit['fuel_purchase_switching_clients']:,} клиентов; постоянный статус поглощается клиентскими эффектами.

Кластеризация **по региону** — основная для всех таблиц: она допускает произвольную зависимость во времени и между клиентами внутри региона, включая общие региональные изменения и повторные наблюдения клиента. Все клиенты вложены в регион. Кластеров только 16; используется t(15), 95% ДИ и CR1-поправка G/(G−1) × (N−1)/(N−K). K включает полное число оценённых параметров с поглощёнными Client FE и Month FE; поправка не занижается из-за малого числа столбцов после вычитания средних. Даже с этой поправкой 16 кластеров ограничивают точность inference.

Альтернатива — кластеризация по клиентам, которая учитывает повторные наблюдения клиента, но не общие региональные ошибки. Её меньшие p-value не выбираются вместо основных. Двухстороннюю client × region кластеризацию не добавляем: клиенты вложены в регионы, это не независимое второе измерение кластеров.

{md_table(alternative[['model','term','beta','SE','p_value','ci95_low','ci95_high','n_clusters']])}

Положительный вывод для FuelLiters и HasFuelPurchase сохраняется при обоих вариантах SE. Для самого PriceShock inference различается: при основной региональной кластеризации β={price.beta:.4f}, p={price.p_value:.4f}; меньший p-value клиентской альтернативы не считается подтверждением, поскольку региональные общие ошибки она не учитывает.

### PriceShock и идентификация

После Client FE и Month FE SD PriceShock = {identification['price_shock_twfe_sd']:.6g} (около {identification['price_shock_twfe_sd']*100:.3f} п.п.). Корреляция с residual FuelLiters = {identification['fuel_price_twfe_correlation']:.4f}; condition number после масштабирования столбцов = {identification['scaled_condition_number']:.4f}. Матрица полного ранга, PriceShock не поглощён полностью и проблемы коллинеарности не обнаружено. Добавление цены меняет оценку топлива с {within.beta:.6f} до {controlled.beta:.6f}; вывод сохраняется.

### Единственная count-robustness

Conditional Poisson условится на общее число штрафов каждого клиента, исключая его индивидуальный интерсепт из likelihood; Month FE остаются. Используются FuelLiters/100 и четыре индикатора месяца, без подбора преобразований. Клиенты с нулевой суммой штрафов не информативны для условного likelihood: исключены {poisson_meta['removed_zero_total_clients']:,}, остались {poisson_meta['retained_clients']:,} клиентов и {int(poisson.N):,} наблюдений. Это ограничение выборки count-модели явно отличается от OLS на всех клиентах.

Ковариация — sandwich по региональным суммам условного score, поправка G/(G−1), t(15). Градиент в решении {poisson_meta['converged_gradient_inf']:.3g}; информация положительно определена. Коэффициент положителен, IRR={poisson.IRR_per_100_liters:.4f}, p={poisson.p_value:.6g}. Направление не зависит от выбора OLS/Poisson, но размер эффекта и выборки различаются. Счётная модель предполагает Poisson conditional mean; региональные robust SE сами по себе не проверяют это предположение.

## PRE/POST и корреляции

PRE = апрель + май, POST = июнь + июль + август, во всех случаях **среднее за месяц**, а не сравнение сумм двух и трёх месяцев. Окна выбраны до расчёта по заданию. Конец мая уже может содержать кризис, поэтому это условное описательное разделение, не чистое причинное pre/post сравнение.

{md_table(correlations)}

В строке within_client_only вычтены только собственные средние клиента; в within_client_and_month — также общие месячные эффекты. Pearson/Spearman рассчитаны по исходным клиентам или клиент-месяцам, а не по точкам binned scatter. Обычные независимые p-value для корреляций повторных наблюдений не используются. Сглаженные корзины визуально подчёркивают тренд, но не отменяют слабую корреляцию индивидуальных изменений.

## Реализованные терцили сокращения топлива — только описание

Среди {group_meta['positive_pre_clients']:,} клиентов с PRE > 0 границы изменения: {group_meta['tertile_q1']:.2%} и {group_meta['tertile_q2']:.2%}. Равные значения остаются вместе. Топливные изменения определяют группы после наблюдения POST; штрафы для границ не используются. Это не причинная модель, отбор может отражать регрессию к среднему и случайность момента покупок. Верхний терциль содержит преимущественно рост, поэтому не назван stable.

{md_table(group_show)}

В таблице проценты записаны долями: −0,83 = −83%. fuel_change_pct_from_group_means — отношение групповых средних; mean_individual_fuel_change_pct — среднее индивидуальных процентов. Эти показатели не обязаны совпадать, особенно при малом PRE. На графике используется отношение групповых средних. Хвосты не удалены; максимальное индивидуальное изменение {group_meta['max_individual_relative_change']:.2%} явно сохранено.

У strong reducers покупки снизились на {strong.fuel_change_pct_from_group_means:.1%}, штрафы с {strong.pre_fines:.3f} до {strong.post_fines:.3f}: {strong.fines_change:+.3f} штрафа/месяц ({strong_pct:.1%}). Средний терциль: почти неизменные штрафы, верхний — рост. Таким образом, описательно более сильное снижение покупок сопровождается большим снижением штрафов, но далеко не пропорциональным. Это не основание для причинной интерпретации.

У {group_meta['zero_pre_clients']:,} клиентов PRE fuel = 0; процентное изменение для них не рассчитывается, они выделены отдельно. {group_meta['zero_pre_post_buyers']:,} из них покупают топливо в POST. Эти клиенты остаются в FE-моделях и анализе абсолютных изменений.

## Что происходит без наблюдаемой покупки

{md_table(zeros)}

В {audit['zero_fuel_client_months']:,} нулевых топливных месяцах есть {audit['fines_in_zero_fuel_months']:,} штрафов — {audit['fines_share_zero_fuel']:.2%} всех штрафов выбранной когорты; исходные 25,33% полной когорты также воспроизведены. В среднем без покупки {zero_row.mean_fines:.3f} штрафа/месяц; ноль топлива не означает ноль штрафов или отсутствие езды. Основной within extensive β={extensive.beta:.4f}, p={extensive.p_value:.6g}: месяцы с покупкой связаны с несколько большим числом штрафов у того же клиента после Month FE.

## Ответы Stage 1

1. BETWEEN relationship — **YES**: +{between.beta:.3f} штрафа на различие средних покупок в 100 л; корреляция при этом невысока.
2. WITHIN relationship — **YES**: −100 наблюдаемых литров связаны с −{within.beta:.3f} штрафа/месяц у того же клиента, после Month FE. Это ассоциация, не эффект вмешательства.
3. WITHIN меньше BETWEEN на **{100*(1-scale):.1f}%** по точечным оценкам, в **{1/scale:.2f} раза**; нулевым within не является.
4. После Month FE и PriceShock — **YES**, коэффициент {controlled.beta:.4f}, p={controlled.p_value:.6g}. Альтернативная клиентская кластеризация и count-модель сохраняют положительное направление топлива.
5. В нулевые топливные месяцы штрафы продолжают наблюдаться: **{audit['fines_share_zero_fuel']:.2%} всех штрафов** когорты. При наличии покупки within-разница +{extensive.beta:.3f} штрафа.
6. Strong reducers — **YES, описательно**: штрафы снизились на {strong_pct:.1%}, при падении наблюдаемых покупок на {strong.fuel_change_pct_from_group_means:.1%}; в других терцилях падение меньше или есть рост.
7. Объясняет ли менее выраженная within-связь прежний агрегатный разрыв? **PARTIALLY**: это совместимо с небольшим откликом штрафов на изменения сервисных покупок, но не установленное разложение PriceShock → Fines. Нельзя перемножать старый агрегированный ценовой коэффициент и новый within β и выдавать это за mediation. Слабая значимость агрегированного результата и различия выборок/контролей остаются отдельными ограничениями.
8. Содержательная история для Т-Страхования — **YES на уровне аналитики измерения**: связь между разными клиентами не равна связи изменений у одного клиента; падение сервисных покупок не означает пропорционального изменения штрафной активности. Это не доказательство изменения пробега, стиля вождения, страхового риска или основания для тарифа.

**Stage 1 завершён; дальнейший механизм и Stage 2 автоматически не запускались.**

## Три графика

![Between vs within](figures/01_between_vs_within.png)

![Changes](figures/02_fuel_change_vs_fines_change.png)

![Reducer groups](figures/03_reducer_groups.png)

## Воспроизводимость

`.venv/bin/python fuel_fines_decoupling_analysis/run_analysis.py`

`.venv/bin/python fuel_fines_decoupling_analysis/test_models.py -v`

FE-коэффициенты и региональные SE проверены против явной OLS с клиентскими/месячными индикаторами на тестовой панели. Conditional Poisson проверен против Poisson GLM с явными клиентскими эффектами. Прежние файлы защищены перечнем и контрольными суммами в previous_files_manifest.json. План анализа сохранён до моделей в analysis_plan.json. Источники и параметры — metadata.json; все CSV — tables/.
'''
    (out/'diagnostic_summary.md').write_text(narrative,encoding='utf-8')
    # Small standalone HTML view; no new plots or analytical calculations.
    pieces=[];lines=narrative.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('| '):
            block=[]
            while i<len(lines) and lines[i].startswith('| '):block.append(lines[i]);i+=1
            cells=[[escape(c.strip()) for c in r.strip('|').split('|')] for r in block]
            pieces.append('<div class="table"><table><thead><tr>'+''.join('<th>'+c+'</th>' for c in cells[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+c+'</td>' for c in r)+'</tr>' for r in cells[2:])+'</tbody></table></div>');continue
        if line.startswith('!['):
            path=line.split('](')[1].rstrip(')');pieces.append('<img src="'+escape(path)+'" alt="Диагностический график">')
        elif line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));pieces.append(f'<h{level}>'+escape(line[level:].strip())+f'</h{level}>')
        elif line.strip():pieces.append('<p>'+escape(line).replace('**','')+'</p>')
        i+=1
    (out/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Fuel–fines: Stage 1</title><style>body{font:16px/1.6 system-ui;color:#243246;max-width:1150px;margin:40px auto;padding:0 24px}h1,h2{color:#234d6b}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce2e8;text-align:left}th{background:#eef3f8}.table{overflow:auto}img{width:100%;height:auto;margin:20px 0}p{max-width:1050px}</style>'+''.join(pieces)+'</html>',encoding='utf-8')
    return verdict
