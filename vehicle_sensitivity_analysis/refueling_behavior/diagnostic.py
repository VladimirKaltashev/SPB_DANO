"""Transaction identity audit. Stop before models if physical refuels are ambiguous."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    stage = ROOT / "vehicle_sensitivity_analysis/outputs"
    metadata = json.loads((stage / "metadata.json").read_text())
    for source in metadata['sources'].values():
        assert sha(source['path']) == source['sha256']
    source = Path(metadata['sources']['fuel']['path'])
    fuel = pd.read_csv(source, dtype={'client_id':'string','order_id':'string','matched_order_id':'string'})
    assert not fuel[['client_id','order_id']].isna().any().any()
    assert fuel.order_id.is_unique
    fuel['timestamp'] = pd.to_datetime(fuel.order_datetime, errors='raise')
    assert not fuel.timestamp.isna().any()
    assert np.isfinite(fuel.physical_fuel_volume_main).all() and fuel.physical_fuel_volume_main.ge(0).all()
    assert fuel.physical_fuel_transaction_main.eq(fuel.physical_fuel_volume_main.gt(0).astype(int)).all()
    assert fuel.loc[fuel.order_fuel_volume.lt(0),'physical_fuel_volume_main'].eq(0).all()
    assert fuel.loc[fuel.refund_type.eq('full_cancellation'),'physical_fuel_volume_main'].eq(0).all()
    # Every matched return refers reciprocally to its positive order; neither return is an extra event.
    indexed = fuel.set_index('order_id')
    negative_matches = fuel.loc[fuel.order_fuel_volume.lt(0) & fuel.refund_match_status.eq('matched')]
    for r in negative_matches.itertuples():
        positive = indexed.loc[r.matched_order_id]
        assert positive.matched_order_id == r.order_id
        assert positive.client_id == r.client_id and positive.order_fuel_volume > 0
        expected = max(positive.order_fuel_volume+r.order_fuel_volume,0)
        assert np.isclose(positive.physical_fuel_volume_main,expected,atol=.02)
    scoped = fuel.loc[fuel.timestamp.ge('2026-04-01') & fuel.timestamp.lt('2026-09-01')].copy()
    orders = scoped.loc[scoped.physical_fuel_volume_main.gt(0)].copy()
    orders = orders.sort_values(['client_id','timestamp','order_id']).reset_index(drop=True)
    attributes = pd.read_csv(stage/'client_attributes.csv',dtype={'client_id':'string'})
    orders = orders.merge(attributes[['client_id','car_count','analytical_eligible']], on='client_id',validate='many_to_one')
    assert orders.car_count.notna().all()
    columns=['order_id','timestamp','physical_fuel_volume_main','order_fuel_price_1liter','refund_match_status']
    for column in columns:
        orders['previous_'+column] = orders.groupby('client_id')[column].shift()
    orders['gap_seconds'] = (orders.timestamp-orders.previous_timestamp).dt.total_seconds()
    assert orders.gap_seconds.dropna().ge(0).all()
    # Time windows are audit summaries only. They do not define sessions or alter any source row.
    close = orders.loc[orders.gap_seconds.le(300)].copy()
    close['same_liters'] = close.physical_fuel_volume_main.eq(close.previous_physical_fuel_volume_main)
    close['same_price'] = close.order_fuel_price_1liter.eq(close.previous_order_fuel_price_1liter)
    selected = ['client_id','car_count','analytical_eligible','order_id','previous_order_id',
        'timestamp','previous_timestamp','gap_seconds','physical_fuel_volume_main',
        'previous_physical_fuel_volume_main','order_fuel_price_1liter','previous_order_fuel_price_1liter',
        'refund_match_status','previous_refund_match_status','same_liters','same_price']
    close[selected].to_csv(OUT/'close_order_pairs.csv',index=False)
    close.loc[close.car_count.eq(1)].sort_values('gap_seconds')[selected].head(20).to_csv(OUT/'examples_single_car.csv',index=False)
    windows=[]
    for seconds in [60,300,3600]:
        for population, frame in [('all_clean_clients',orders),('stage1_analytical_clients',orders.loc[orders.analytical_eligible])]:
            matches=frame.loc[frame.gap_seconds.le(seconds)]
            windows.append(dict(population=population,window_seconds=seconds,positive_orders=len(frame),
                close_successor_orders=len(matches),share_of_positive_orders=len(matches)/len(frame),
                affected_clients=matches.client_id.nunique(),single_car_pairs=int(matches.car_count.eq(1).sum())))
    pd.DataFrame(windows).to_csv(OUT/'timestamp_audit.csv',index=False)
    # Confirm that this audit uses the exact same cleaned physical liters and analytical cohort.
    analytical=orders.loc[orders.analytical_eligible].copy()
    analytical['month']=analytical.timestamp.dt.to_period('M').astype(str)
    rebuilt=analytical.groupby(['client_id','month']).physical_fuel_volume_main.sum()
    previous=pd.read_csv(stage/'diagnostic_client_month.csv',usecols=['client_id','month','fuel_liters'],dtype={'client_id':'string'})
    previous=previous.join(rebuilt.rename('rebuilt_liters'),on=['client_id','month'])
    previous['rebuilt_liters']=previous.rebuilt_liters.fillna(0)
    assert np.allclose(previous.fuel_liters,previous.rebuilt_liters)
    assert np.isclose(orders.physical_fuel_volume_main.sum(),metadata['full_total_liters'])
    summary=dict(source=str(source),source_sha256=sha(source),rows_all=len(fuel),rows_april_august=len(scoped),
        positive_orders_april_august=len(orders),positive_orders_analytical=len(analytical),
        unique_order_ids=bool(fuel.order_id.is_unique),missing_identifiers=0,
        exact_client_timestamp_duplicate_rows=int(orders.duplicated(['client_id','timestamp'],keep=False).sum()),
        exact_client_time_volume_price_duplicate_rows=int(orders.duplicated(['client_id','timestamp','physical_fuel_volume_main','order_fuel_price_1liter'],keep=False).sum()),
        paired_refunds_full_source=len(negative_matches),full_cancellations_full_source=int(negative_matches.refund_type.eq('full_cancellation').sum()),
        partial_returns_full_source=int(negative_matches.refund_type.eq('partial_refund').sum()),
        unresolved_negative_rows_full_source=int(fuel.refund_match_status.eq('unresolved').sum()),
        positive_event_flag_matches_cleaned_volume=True,
        close_pairs_5min=len(close),close_pairs_share_of_positive_orders=len(close)/len(orders),
        close_pairs_affected_clients=close.client_id.nunique(),close_pairs_single_car=int(close.car_count.eq(1).sum()),
        close_pairs_single_car_clients=close.loc[close.car_count.eq(1),'client_id'].nunique(),
        close_pairs_same_price=int(close.same_price.sum()),close_pairs_same_volume=int(close.same_liters.sum()),
        close_pairs_same_price_and_volume=int((close.same_price & close.same_liters).sum()),
        close_pairs_neither_matched_return=int((close.refund_match_status.eq('not_applicable') & close.previous_refund_match_status.eq('not_applicable')).sum()),
        minimum_positive_gap_seconds=float(orders.gap_seconds.min()),
        all_positive_timestamps_nonmidnight=bool(orders.timestamp.ne(orders.timestamp.dt.normalize()).all()),
        share_subsecond_timestamps=float(orders.timestamp.dt.microsecond.gt(0).mean()),
        total_liters=float(orders.physical_fuel_volume_main.sum()),analytical_monthly_reconciliation='PASS',
        available_raw_fields=['order_id','order_datetime','order_fuel_volume','order_fuel_price_1liter','client_id'],
        unavailable_session_fields=['station_id','pump_id','refueling_session_id','parent_order_id','receipt_id','order_status','auto_document_id'])
    dump('transaction_audit.json',summary)
    # Gate required by the user: order identity is not sufficient evidence of distinct physical visits.
    decision=dict(status='STOPPED_AT_EVENT_DEFINITION',reliable_unit='distinct cleaned positive-volume order',
        physical_refuel_event_identified=False,reason='distinct close-time orders cannot be reliably mapped to separate physical refuels',
        models_run=0,plots_created=0,fines_analyzed=False,stage2_executed=False,
        more_frequent_refuels='INCONCLUSIVE',count_price_response='NO CLEAR DIFFERENCE — not estimated',
        size_price_response='NO CLEAR DIFFERENCE — not estimated',decomposition='NOT TESTED',
        proceed_to_refueling_exposure_fines='NO on the evidence currently available')
    dump('decision.json',decision)
    report=f'''REFUEL EVENT DEFINITION: отдельная физическая заправка / посещение АЗС надёжно не идентифицированы.

Проверяемая единица, которую дают данные: **уникальный order_id с physical_fuel_volume_main > 0 после существующей очистки**, отнесённый к client_id и времени исходного положительного заказа. Это очищенный заказ на покупку топлива. Он не гарантированно соответствует одной физической заправке или отдельному посещению АЗС.

## Решение по первому шагу

**STOPPED_AT_EVENT_DEFINITION.** Согласно условию задания «Если из данных невозможно надёжно определить отдельные заправки — остановиться», регрессии, client-month метрики реальных заправок, small-refuel threshold и два графика не строились. Штрафы и полный Stage 2 не анализировались. Это не отрицательный результат проверки механизма: до его статистической проверки не дошли.

## Что подтверждено

- Исходник: data/processed/fuel_clean.csv, SHA-256 `{summary['source_sha256']}`; совпадает с источником Stage 1.
- Во всём файле {len(fuel):,} строк; order_id уникален, пропусков client_id / order_id нет.
- За апрель–август: {len(orders):,} положительных очищенных заказов, из них {len(analytical):,} у клиентов аналитической выборки Stage 1.
- Нет точных совпадений client_id + timestamp и client_id + timestamp + литры + цена среди положительных заказов.
- Во всём исходнике проверены {len(negative_matches)} пар возвратов: {summary['full_cancellations_full_source']} полных отмен, {summary['partial_returns_full_source']} частичных возвратов. Полная отмена не создаёт событие; частичный возврат оставляет один положительный заказ с остаточными литрами; отрицательная строка не создаёт дополнительное событие.
- {summary['unresolved_negative_rows_full_source']:,} несопоставленных отрицательных строк не считаются физическими покупками по существующим правилам очистки. Это правило сохранено, но само по себе не восстанавливает реальную сессию заправки.
- physical_fuel_transaction_main полностью совпадает с индикатором положительных очищенных литров. Следовательно, этот флаг тоже считает положительные заказы; независимого признака физической сессии он не добавляет.
- Все положительные заказы имеют время внутри суток, {summary['share_subsecond_timestamps']:.2%} — доли секунды. Точность записи есть, но она не подтверждает время начала или окончания физической заправки.
- Сумма {summary['total_liters']:,.2f} л совпадает с предыдущей полной панелью; для аналитической выборки совпадение проверено отдельно по каждому клиенту × месяцу.

## Почему каждый order_id нельзя автоматически считать посещением

Документация `Задание.md`, таблица 3, определяет order_id как «Уникальный идентификатор заказа (транзакции)», order_datetime — как время заказа. Гарантии «один заказ = одна физическая заправка / один визит» там нет. Полей станции, колонки, сессии, родительского заказа, чека или статуса заказа в топливном источнике нет; автомобиль также не указан.

Обнаружено **{len(close):,} соседних пар положительных заказов одного клиента с интервалом ≤5 минут**, затрагивающих {close.client_id.nunique():,} клиентов. Это {len(close)/len(orders):.2%} положительных заказов как последующих элементов пары; это не доля доказанных дубликатов. **{summary['close_pairs_single_car']:,} пар приходится на клиентов с одной машиной** ({summary['close_pairs_single_car_clients']:,} клиентов), поэтому проблема не сводится к нескольким автомобилям.

В {summary['close_pairs_same_price_and_volume']:,} близких парах совпадают и литры, и цена; минимальный интервал — {summary['minimum_positive_gap_seconds']:.3f} секунды. Все {summary['close_pairs_neither_matched_return']:,} близких пар состоят из строк, которые не отмечены как сопоставленные возвраты. Уже обработанные возвраты поэтому не объясняют эту неоднозначность.

Например, у клиента с одной машиной 13 мая есть два разных order_id на 7,62 л по одинаковой цене в 04:52:04.903 и 04:52:11.492; интервал 6,589 секунды. У другого клиента 17 мая заказы на 50,62 и 55,62 л записаны с интервалом 9,680 секунды. Точные идентификаторы и исходные строки сохранены в examples_single_car.csv; полный перечень близких пар — close_order_pairs.csv.

Такие записи могут быть отдельными заказами в рамках одного посещения, повторным проведением операции, частями одной покупки либо иными транзакциями. Аудит не доказывает, какой вариант верен, и не разрешает удалять или объединять их. Окна 1/5/60 минут используются только для описания временной близости в timestamp_audit.csv, не для подбора модели или определения новой сессии.

Если кризис меняет дробление заказа, число строк может расти, а литры на строку — падать даже при неизменном числе посещений. Это способно имитировать ровно искомый frequency/size-паттерн. Аналогично секунды между заказами нельзя уверенно интерпретировать как интервал между заправками, а малый заказ — как самостоятельную небольшую заправку.

## Ответы на пять вопросов

1. Большие двигатели заправляются чаще? **INCONCLUSIVE** — счётчик физических заправок не установлен.
2. Изменение числа заправок при PriceShock? **NO CLEAR DIFFERENCE** — модель не оценивалась; это не нулевой эффект.
3. Изменение размера одной заправки? **NO CLEAR DIFFERENCE** — размер физической заправки не установлен.
4. Объясняет ли frequency/size decomposition прежний результат? **Не проверено**, поскольку первый шаг не пройден.
5. Есть ли основание переходить к refueling exposure × crisis → fines? **NO на текущем этапе** — exposure к отдельным физическим посещениям не измерен.

Для продолжения нужен идентификатор сессии/посещения или подтверждённое поставщиком правило связи нескольких заказов с одной заправкой. Отдельная диагностика числа **заказов в сервисе** технически возможна, но ответит на более узкий вопрос и не установит частоту посещений, очереди, стресс или дорожную экспозицию. Предыдущие месячные результаты по литрам и вероятности покупки здесь не пересчитывались и не объявляются опровергнутыми: аудит выявил ограничение нового перехода от заказов к физическим событиям, а не исправленные дубликаты.

## Воспроизведение

`.venv/bin/python vehicle_sensitivity_analysis/refueling_behavior/diagnostic.py`
'''
    (OUT/'diagnostic_summary.md').write_text(report,encoding='utf-8')
    manifest=json.loads((OUT/'previous_files_manifest.json').read_text())
    current={str(p.relative_to(ROOT)):sha(p) for d in ['archive_monthly_risk_clustering','fines_per_liter_analysis','usage_composition_analysis','vehicle_sensitivity_analysis']
        for p in (ROOT/d).rglob('*') if p.is_file() and HERE not in p.parents}
    assert current==manifest, 'Previous results changed'
    dump('validation.json',dict(status='PASS',previous_files_unchanged=len(manifest),
        cleaned_source_hashes_verified=len(metadata['sources']),refund_pair_checks='PASS',
        prior_client_month_liters_reconciled=True,diagnostic_code_sha256=sha(Path(__file__))))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
