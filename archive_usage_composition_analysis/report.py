"""Report each link separately; no conclusion selected from a favorable specification."""

from html import escape
import json

import numpy as np

from .prepare_data import GROUPS, ZERO
from .visualization import p_text


def md_table(table):
    def fmt(value):
        return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value).replace("|", "/")
    return "\n".join(["| " + " | ".join(table.columns) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |",
                      *["| " + " | ".join(map(fmt, row)) + " |" for row in table.itertuples(index=False, name=None)]])


def diagnose(slopes, interactions, baseline, composition_results):
    import pandas as pd
    primary = interactions.loc[interactions.outcome.eq("liters_index") & interactions.term.eq("PriceShock_x_HIGH")]
    main, post = [primary.loc[primary.period.eq(period)].iloc[0] for period in ["all_months", "post_baseline"]]
    high_fines = baseline.loc[baseline.term.eq("HIGH")].iloc[0]
    c = composition_results.loc[composition_results.term.eq("price_shock") & composition_results.denominator.eq("tertiles_only")]
    low, high = [c.loc[c.threshold_liters.eq(0) & c.usage_group.eq(g)].iloc[0] for g in ["LOW", "HIGH"]]
    high20 = c.loc[c.threshold_liters.eq(20) & c.usage_group.eq("HIGH")].iloc[0]
    low20 = c.loc[c.threshold_liters.eq(20) & c.usage_group.eq("LOW")].iloc[0]
    h1_support = all(row.beta > 0 and row.p_value < .05 for row in [main, post])
    h1_against = all(row.beta < 0 and row.p_value < .05 for row in [main, post])
    h2_support = high_fines.beta > 0 and high_fines.p_value < .05
    h3_main = (low.beta < 0 and low.p_value < .05) or (high.beta > 0 and high.p_value < .05)
    h3_sensitivity = (low20.beta < 0 and low20.p_value < .05) or (high20.beta > 0 and high20.p_value < .05)
    h3_support = h3_main and h3_sensitivity
    diagnosis = {
        "H1": {"answer": "YES" if h1_support else "NO" if h1_against else "INCONCLUSIVE",
               "result": "SUPPORTED" if h1_support else "NOT SUPPORTED" if h1_against else "MIXED / INCONCLUSIVE",
               "main_high_minus_low_beta_index": float(main.beta), "main_p_value": float(main.p_value),
               "post_baseline_high_minus_low_beta_index": float(post.beta), "post_baseline_p_value": float(post.p_value)},
        "H2": {"answer": "YES" if h2_support else "NO" if high_fines.beta < 0 and high_fines.p_value < .05 else "INCONCLUSIVE",
               "result": "SUPPORTED" if h2_support else "NOT SUPPORTED", "high_minus_low_fines": float(high_fines.beta),
               "p_value": float(high_fines.p_value)},
        "H3": {"answer": "YES" if h3_support else "INCONCLUSIVE", "result": "SUPPORTED" if h3_support else "NOT SUPPORTED",
               "low_share_beta": float(low.beta), "low_p_value": float(low.p_value),
               "high_share_beta": float(high.beta), "high_p_value": float(high.p_value)},
        "overall": "SUPPORTED" if h1_support and h2_support and h3_support else "PARTIALLY SUPPORTED" if h3_support and (h1_support or h2_support) else "NOT SUPPORTED",
        "meaning": "NOT SUPPORTED means the proposed mechanism has not received the required joint support; it is not proof that every possible composition effect is absent.",
    }
    rows = []
    for group in GROUPS:
        row = slopes.loc[slopes.outcome.eq("liters_per_driver") & slopes.term.eq("price_shock") & slopes.usage_group.eq(group)].iloc[0]
        rows.append({"TEST": f"Price → fuel activity {group}", "RESULT": "SIGNIFICANT" if row.p_value < .05 else "NOT SIGNIFICANT",
                     "EFFECT": f"{.1 * row.beta:+.2f} л/водитель при +10 п.п. PriceShock", "P-VALUE": p_text(row.p_value),
                     "INTERPRETATION": "Абсолютная реакция; не тест относительной чувствительности"})
    rows.extend([
        {"TEST": "HIGH − LOW: relative interaction", "RESULT": diagnosis["H1"]["result"],
         "EFFECT": f"{.1 * main.beta:+.2f} п.п. индекса при +10 п.п. PriceShock", "P-VALUE": p_text(main.p_value),
         "INTERPRETATION": f"Май–август: {.1 * post.beta:+.2f} п.п.; p={p_text(post.p_value)}. Знак меняется."},
        {"TEST": "Baseline fines HIGH − LOW", "RESULT": diagnosis["H2"]["result"],
         "EFFECT": f"{high_fines.beta:+.3f} штрафа/водитель за апрель", "P-VALUE": p_text(high_fines.p_value),
         "INTERPRETATION": "HIGH имеет больше исходных штрафов на человека"},
        {"TEST": "Price → share LOW among active", "RESULT": "SUPPORTED" if low.beta < 0 and low.p_value < .05 else "NOT SUPPORTED",
         "EFFECT": f"{10 * low.beta:+.3f} п.п. доли при +10 п.п. PriceShock", "P-VALUE": p_text(low.p_value),
         "INTERPRETATION": "Ожидаемое снижение доли LOW не установлено"},
        {"TEST": "Price → share HIGH among active", "RESULT": "SUPPORTED" if high.beta > 0 and high.p_value < .05 else "NOT SUPPORTED",
         "EFFECT": f"{10 * high.beta:+.3f} п.п. доли при +10 п.п. PriceShock", "P-VALUE": p_text(high.p_value),
         "INTERPRETATION": "Ожидаемый рост доли HIGH не установлен"},
    ])
    return diagnosis, pd.DataFrame(rows)


def write_report(out, meta, baseline, zero, national, slopes, interactions, baseline_results, composition_results, composition):
    diagnosis, tests = diagnose(slopes, interactions, baseline_results, composition_results)
    tests.to_csv(out / "tables/hypothesis_tests.csv", index=False)
    (out / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    base = baseline.set_index("usage_group")
    main_national = national.loc[national.usage_group.isin(GROUPS)]
    activity_table = main_national.pivot(index="month", columns="usage_group", values="liters_index")[GROUPS].reset_index()
    fines_table = main_national[["month", "usage_group", "fines_per_driver", "total_fines"]]
    zero_eligible = zero.loc[zero.population.eq("eligible_zero_baseline")].iloc[0]
    cmain = composition.loc[composition.denominator_population.eq("positive_baseline_tertiles") & composition.active_threshold_liters.eq(0)]
    first, last = cmain.sort_values("month").iloc[[0, -1]].itertuples(index=False)
    h1 = diagnosis["H1"]
    h2 = diagnosis["H2"]
    h3 = diagnosis["H3"]
    interpretation = (
        f"H1: {h1['answer']} — LOW сильнее сокращает наблюдаемые покупки, чем HIGH? "
        f"Основной относительный interaction HIGH−LOW = {h1['main_high_minus_low_beta_index']:.3f}, p={p_text(h1['main_p_value'])}. "
        f"В заранее указанной проверке май–август: {h1['post_baseline_high_minus_low_beta_index']:.3f}, p={p_text(h1['post_baseline_p_value'])}. "
        "Знак зависит от включения месяца формирования групп; устойчивой поддержки H1 нет.\n\n"
        f"H2: {h2['answer']} — HIGH имеет больше штрафов на водителя. Апрельские средние LOW / MEDIUM / HIGH: "
        f"{base.loc['LOW', 'baseline_fines_per_driver']:.3f} / {base.loc['MEDIUM', 'baseline_fines_per_driver']:.3f} / "
        f"{base.loc['HIGH', 'baseline_fines_per_driver']:.3f}. HIGH−LOW = {h2['high_minus_low_fines']:.3f}, p={p_text(h2['p_value'])}.\n\n"
        f"H3: {h3['answer']} — роста доли HIGH в составе наблюдаемых покупателей не установлено: "
        f"β_HIGH={h3['high_share_beta']:.4f}, p={p_text(h3['high_p_value'])}; "
        f"β_LOW={h3['low_share_beta']:.4f}, p={p_text(h3['low_p_value'])}. "
        f"Национальная доля HIGH с апреля по август изменилась с {first.share_HIGH:.2%} до {last.share_HIGH:.2%}, "
        f"LOW — с {first.share_LOW:.2%} до {last.share_LOW:.2%}. Это не ожидаемый сдвиг в сторону HIGH.\n\n"
        f"OVERALL COMPOSITION HYPOTHESIS: {diagnosis['overall']}. "
        "Поддержано звено об исходных штрафах HIGH, но не установлено главное изменение состава покупателей; "
        "относительная ценовая реакция групп чувствительна к окну. Поэтому объяснять прежний разрыв между литрами и штрафами "
        "этим механизмом на имеющихся результатах нельзя. NOT SUPPORTED не означает доказанного отсутствия любого composition effect.")
    methodology = [
        f"Baseline — полный апрель 2026. Ограничения по Задание.md начинаются в конце мая; март представлен не полностью, "
        "поэтому не используется. Май–август не участвуют в назначении групп. Существующие региональные цены и PriceShock "
        "прочитаны без изменения из предыдущего анализа: средневзвешенная цена месяца / апрельская цена данного региона − 1.",
        f"Из {meta['n_all_drivers']:,} клиентов {meta['n_eligible_drivers']:,} имеют 30 дней в апрельской панели и дату подписки не позже 1 апреля. "
        f"{meta['n_insufficient_baseline']:,} отнесены в INSUFFICIENT_BASELINE. Это заранее выбранный консервативный критерий "
        "наблюдаемости, а не доказанная дата начала поездок. Панель сама содержит заполненные нулями недели, а подписка "
        "не гарантирует полноты покупок: критерий ограничивает выборку и отличается от прошлого анализа всей когорты.",
        f"Основные терцили рассчитаны глобально среди {meta['n_tertile_drivers']:,} клиентов с положительными апрельскими литрами. "
        f"Границы: LOW ≤ {meta['q1_liters']:.2f} л; MEDIUM > {meta['q1_liters']:.2f} и ≤ {meta['q2_liters']:.2f} л; HIGH > {meta['q2_liters']:.2f} л. "
        "Равные значения не разрывались по идентификаторам; группы не пересчитывались в последующие месяцы. "
        "Нули позже остаются в той же группе и в знаменателе liters/driver.",
        f"Групповая панель: {meta['n_group_cells']} наблюдений = 16 регионов × 5 месяцев × 3 группы. "
        f"Минимум в ячейке — {meta['minimum_group_drivers']} клиентов. В каждой отдельной регрессии 80 регион-месяцев. "
        "Состав каждой региональной группы постоянен во времени. Национальная динамика получена из сумм литров/штрафов "
        "и числа клиентов, а не из невзвешенного среднего региональных отношений.",
        "Абсолютный исход — liters_per_driver. Относительный исход — 100 × liters_per_driver(r,g,t) / liters_per_driver(r,g,April). "
        "Процентное изменение = индекс − 100. Регионально-групповой знаменатель фиксирован до шока. "
        "Это нормированная реакция относительно апреля, не оценка точной логарифмической эластичности и не среднее индивидуальных отношений.",
        "Основная interaction-модель построена на регионе × месяце × фиксированной группе: "
        "Y = α + β PriceShock + γ Medium + δ High + θM PriceShock×Medium + θH PriceShock×High. LOW — reference. "
        "Такое агрегирование корректно для выбранного исхода — средних по региональным группам. "
        "Оценка невзвешенная: каждый регион имеет одинаковый вес, как в предыдущих регрессиях. "
        "Модель не выдаёт групповые средние за индивидуальные наблюдения. Групповые наклоны совпадают с отдельными OLS, "
        "а тест θ учитывает ковариацию реакций групп через общую региональную кластеризацию ошибок.",
        "Все ошибки сгруппированы по 16 регионам, с поправкой малой выборки и t(G−1). "
        "Отдельно сохранены совместные F-тесты θM=θH=0. Положительный θH в модели индекса соответствует более слабому "
        "сокращению HIGH относительно LOW; отрицательный — обратному направлению. Основная проверка включает все пять месяцев. "
        "Заранее указанная robustness исключает апрель из регрессии, сохраняя группы и baseline; оба результата показаны.",
        "H2 оценивается по клиентам в апреле: fines_count ~ 1 + Medium + High, без нормирования на литры, "
        "с ошибками по региону. Сохранены HIGH−LOW, MEDIUM−LOW и совместный тест. "
        "Штрафы не использовались для назначения usage-групп.",
    ]
    zeros_text = (
        f"В основной baseline-eligible выборке {int(zero_eligible.n_drivers):,} ZERO-клиентов "
        f"({zero_eligible.share_of_population:.2%}); {int(zero_eligible.baseline_drivers_with_fines):,} уже имеют штрафы в апреле "
        f"(всего {int(zero_eligible.baseline_fines):,} штрафов). "
        f"{int(zero_eligible.future_buyers):,} ({zero_eligible.share_buying_later:.2%}) покупают топливо в последующие месяцы. "
        "Поэтому ZERO не называется неактивным водителем. Эти клиенты не смешиваются с LOW. "
        "Их возвращение влияет на знаменатель состава покупателей и показано отдельной проверкой.")
    composition_text = (
        "Основной active fuel user — положительные физические литры в данном месяце. Основной знаменатель долей — "
        "активные клиенты из фиксированной положительной baseline-когорты LOW/MEDIUM/HIGH; их доли суммируются в 100%. "
        "Sensitivity: активность ≥20 л. Дополнительно расширен знаменатель на baseline-eligible ZERO-клиентов; "
        "в этом случае доли LOW+MEDIUM+HIGH+ZERO = 100%. Люди с недостаточным baseline-наблюдением не включаются ни в один "
        "из этих двух знаменателей. Правила фиксированы до оценки моделей.\n\n"
        "При добавлении ZERO все три исходных терциля теряют долю активных покупателей; это возвращение клиентов "
        "с нулём наблюдаемых покупок в апреле, а не перераспределение от LOW к HIGH. "
        "При положительных покупках β доли HIGH становится отрицательным. Поэтому смена знаменателя не спасает заявленный механизм.")
    limitations = [
        "Группы описывают объём покупок одного месяца, а не доказанную постоянную интенсивность использования автомобиля. "
        "Отбор по низкому/высокому апрельскому значению создаёт риск регрессии к среднему: "
        "LOW в мае имеет индекс 180,8, HIGH — 71,9. Это совместимо со случайностью момента покупок, запасами топлива "
        "и неполным наблюдением. Именно поэтому основной и post-baseline interaction интерпретируются совместно.",
        "Апрель = 100 — нормировка, а не внешний контроль. Процент относительно апреля зависит от baseline-знаменателя. "
        "Post-baseline результат не является разрешением выбрать май в качестве нового baseline или скрыть основной результат.",
        "Все модели описательные pooled OLS, без фиксированных эффектов региона и месяца. "
        "Сезонность, общие временные изменения и региональные различия могут объяснять часть связей. "
        "Clustered SE учитывают зависимость наблюдений внутри региона, но не устраняют смешивающие факторы. "
        "Пять месяцев и 16 регионов ограничивают точность; p-value вторичных проверок номинальные, без множественной поправки.",
        "Наблюдаемые покупки не равны езде. В предыдущем анализе 25,3% штрафов относились к клиент-месяцам без покупки "
        "топлива в сервисе. Отсутствие покупки здесь не означает отсутствие поездок. " + meta["actual_driving_measure"],
        "HIGH имеет больше штрафов в baseline, но это не доказательство более опасного поведения: "
        "больше штрафов может отражать большую экспозицию. В этой проверке штрафы намеренно не делятся на литры.",
        "Незначимый результат H3 не доказывает неизменность долей. Он означает, что направленное предсказание гипотезы "
        "не получило требуемой статистической поддержки. Разница между значимостью отдельных групп также не заменяет interaction-тест.",
    ]
    sections = [
        ("Краткая таблица проверок", "Эффекты цены приведены для +10 п.п. PriceShock. В H2 — разница средних за апрель.", tests),
        ("Диагноз механизма", interpretation, None),
        ("Baseline и фиксированные группы", "\n\n".join(methodology), baseline),
        ("Нулевые покупки в baseline", zeros_text, zero),
        ("Абсолютная и относительная реакция групп", "В таблице β на единицу PriceShock; N — регион-месяцы. Нормированная динамика ниже: апрель = 100.",
         slopes.loc[slopes.term.eq("price_shock"), ["usage_group", "outcome", "beta", "SE", "p_value", "R_squared", "N"]]),
        ("Нормированная динамика", "Изменение в процентах относительно апреля равно индексу минус 100.", activity_table),
        ("Прямые тесты различий реакции", "LOW — reference. Для относительной гипотезы H1 основной исход — liters_index. Разница в абсолютных литрах не является разницей эластичностей.",
         interactions.loc[interactions.term.str.startswith("PriceShock_x"), ["period", "outcome", "term", "beta", "SE", "p_value", "N"]]),
        ("Исходные различия штрафов", "const — среднее LOW; MEDIUM и HIGH — разницы с LOW. Ошибки сгруппированы по регионам; N — клиенты.", baseline_results),
        ("Динамика штрафов в фиксированных группах", "Вторичная проверка; группы остаются апрельскими.", fines_table),
        ("Состав наблюдаемо активных покупателей", composition_text,
         composition_results.loc[composition_results.term.eq("price_shock"), ["denominator", "threshold_liters", "usage_group", "beta", "SE", "p_value", "N"]]),
        ("Доли терцилей среди наблюдаемых покупателей", "Основной критерий: положительные литры. Доли рассчитаны из общих численностей покупателей.", cmain[["month", "n_active_total", "share_LOW", "share_MEDIUM", "share_HIGH"]]),
        ("Ограничения", "\n\n".join(limitations), None),
    ]
    md, html = ["# Проверка гипотезы composition effect\n"], []
    for title, body, table in sections:
        md.extend([f"## {title}\n", body + "\n"])
        block = f"<section><h2>{escape(title)}</h2>" + "".join(f"<p>{escape(p)}</p>" for p in body.split("\n\n"))
        if table is not None:
            md.append(md_table(table) + "\n")
            block += '<div class="table">' + table.to_html(index=False, border=0, float_format=lambda x: f"{x:.5g}") + '</div>'
        html.append(block + "</section>")
    names = ["01_usage_groups_liters_over_time", "02_usage_groups_normalized_activity", "03_price_effect_by_usage_group",
             "04_baseline_usage_vs_fines", "05_active_user_composition", "06_usage_groups_fines_over_time", "07_interaction_period_sensitivity"]
    md.extend(f"![{name}](figures/{name}.png)\n" for name in names)
    md.append("## Источники\n")
    md.extend(f"- {name}: `{item['path']}`; SHA-256 `{item['sha256']}`" for name, item in meta["sources"].items())
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    images = "".join(f'<img src="figures/{name}.png" alt="{name}" loading="lazy">' for name in names)
    (out / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Проверка composition effect</title><style>body{margin:0;background:#f3f5f8;color:#20344a;font:16px/1.6 system-ui,sans-serif}'
        'main{max-width:1250px;margin:35px auto;padding:0 24px}h1{font-size:34px}h2{font-size:24px}'
        'section{background:white;padding:25px 30px;margin:24px 0;border-radius:12px}p{max-width:110ch}'
        '.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #e1e6ee}'
        'th{background:#edf2f8}img{display:block;width:100%;border-radius:10px;margin:24px 0}a{color:#286398;margin-right:16px}</style>'
        '<main><h1>Проверка гипотезы composition effect</h1><p>Фиксированные апрельские терцили · 16 регионов · апрель–август 2026</p>'
        '<p><a href="report.md">Полный отчёт</a><a href="tables/hypothesis_tests.csv">Таблица проверок</a><a href="analysis_plan.json">Правила до регрессий</a></p>'
        + "".join(html[:2]) + images + "".join(html[2:]) + '</main></html>', encoding="utf-8")
