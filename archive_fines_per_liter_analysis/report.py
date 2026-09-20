"""Source-grounded economic interpretation, without forcing the proposed mechanism."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .visualization import FILENAMES, LABELS, p_text


def markdown_table(table):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "/")
    return "\n".join(["| " + " | ".join(table.columns) + " |",
                      "| " + " | ".join(["---"] * len(table.columns)) + " |",
                      *["| " + " | ".join(map(fmt, row)) + " |" for row in table.itertuples(index=False, name=None)]])


def result_table(regressions):
    result = regressions[["population", "outcome", "BETA", "SE", "p_value", "R_squared", "N"]].copy()
    result["TEST"] = result.outcome.map(LABELS) + " — " + result.population.map(
        {"ALL": "все", "HIGH_RISK": "HIGH", "LOW_RISK": "LOW"})
    return result[["TEST", "BETA", "SE", "p_value", "R_squared", "N"]].rename(
        columns={"p_value": "P-VALUE", "R_squared": "R²"})


def economic_conclusion(regressions):
    rows = regressions.loc[regressions.block.eq("main")].set_index("outcome")
    activity, fines, intensity = [rows.loc[name] for name in LABELS if name != "price_shock"]
    answers = []
    for row in [activity, fines, intensity]:
        direction = "отрицательная" if row.BETA < 0 else "положительная"
        significant = "статистически значима" if row.p_value < .05 else "статистически незначима"
        answers.append(f"{LABELS[row.name]}: связь {direction}, {significant} на уровне 5%. "
                       f"β = {row.BETA:.4f}, p = {p_text(row.p_value)}. "
                       f"Рост PriceShock на 10 п.п. соответствует изменению показателя на {row.change_per_10pct_price:+.3f} "
                       f"(95% ДИ {row.ci95_low * .1:+.3f}…{row.ci95_high * .1:+.3f}).")
    if all(row.BETA < 0 and row.p_value < .05 for row in [activity, fines, intensity]):
        classification = "B — данные согласуются с EXPOSURE + BEHAVIOUR EFFECT как описательной связью. Причинный психологический механизм не установлен."
    elif activity.BETA < 0 and activity.p_value < .05 and fines.BETA < 0 and fines.p_value < .05 and intensity.p_value >= .05:
        classification = ("Паттерн согласуется с A — EXPOSURE EFFECT. Однако незначимость интенсивности не доказывает её неизменность: "
                          "для вывода «примерно не меняется» нужны заранее заданные границы практической эквивалентности.")
    elif all(row.p_value >= .05 for row in [activity, fines, intensity]):
        classification = "C — NO ROBUST EFFECT в трёх основных проверках. Незначимость не является доказательством отсутствия эффекта."
    elif activity.BETA < 0 and activity.p_value < .05 and intensity.BETA > 0 and intensity.p_value < .05:
        classification = (
            "Результат находится вне предложенных категорий A–D: наблюдаемая топливная активность снижается, "
            "а штрафов на 1000 наблюдаемых литров становится больше. A требует снижения штрафов и примерно неизменной интенсивности; "
            "B — снижения интенсивности. C («устойчивой связи нет») также неточна, потому что две основные связи значимы. "
            "Для D нет основания утверждать отсутствие общей связи, а различие p-value групп не является тестом различия их коэффициентов. "
            "Поэтому результат не подгоняется под одну из этих категорий. Гипотеза снижения интенсивности штрафов не подтверждается.")
    else:
        classification = "Смешанный результат: предложенные категории A–D не описывают его однозначно. Следует отдельно интерпретировать каждую оценку и её доверительный интервал."
    return answers, classification


def write_report(out: Path, meta, regressions, thresholds, distribution, selected, national):
    answers, classification = economic_conclusion(regressions)
    main = regressions.loc[regressions.block.eq("main")]
    main_and_groups = regressions.loc[regressions.block.ne("lagged_risk")]
    lagged = regressions.loc[regressions.block.eq("lagged_risk")]
    current = regressions.loc[regressions.block.eq("current_risk")]
    primary_threshold = thresholds.loc[thresholds.primary_threshold].iloc[0]
    same_sign = np.sign(thresholds.BETA).nunique() == 1
    significant = thresholds.p_value.lt(.05).all()
    zero_share = meta["zero_liter_driver_months"] / meta["n_driver_months"]
    zero_fine_share = meta["fines_in_zero_liter_months"] / meta["total_fines"]
    selected_table = selected[["price_growth_group", "region_name", "price_shock"]].copy()
    selected_table["price_shock_percent"] = 100 * selected_table.pop("price_shock")
    threshold_table = thresholds[["threshold_liters", "BETA", "SE", "p_value", "R_squared", "N",
                                  "share_positive_driver_months"]].rename(columns={"p_value": "P-VALUE", "R_squared": "R²"})
    methods = [
        f"Период {meta['start']}–{meta['end']}: {meta['n_regions']} регионов × {meta['n_months']} месяцев, "
        f"{meta['n_cells']} ячеек, {meta['n_drivers']:,} клиентов и {meta['n_driver_months']:,} клиент-месяцев. "
        "Основная выборка включает всех клиентов v2, в том числе без покупок топлива и без характеристик двигателя. "
        f"Минимум в ячейке — {meta['minimum_drivers_per_cell']} водитель.",
        "Использована полная календарная панель из архивного driver_region_month.csv ДО исключения пропусков двигателя. "
        "Состав клиентов и регионы сверены с data/reference/client_week_panel_v2.csv, каждый клиент-месяц — с fines_clean.csv и fuel_clean.csv. "
        "Хеши всех исходных файлов совпадают с источниками архивной кластеризации. Метки NO_LITERS проверены на совпадение исходных штрафов и литров.",
        f"Контрольные суммы: {meta['total_fines']:,} штрафов и {meta['total_liters']:,.2f} физических литров. "
        "Штрафы относятся к фактическому календарному месяцу нарушения, покупки — к месяцу операции. "
        "Недельные итоги не переносились целиком между месяцами. Новая очистка исходников не выполнялась.",
        "Регион — kladr_code регистрации клиента, одинаковый для числителя, знаменателя и цены. Место АЗС неизвестно. "
        "n_drivers — все уникальные клиенты данной когорты в ячейке, а не только покупатели или оштрафованные.",
        "Основная интенсивность = 1000 × SUM(fines) / SUM(liters). "
        "Литры на водителя = SUM(liters) / n_drivers; штрафы на водителя = SUM(fines) / n_drivers. "
        "Среднее индивидуальных отношений не используется как основная метрика. Нулевой суммарный знаменатель обозначался бы пропуском; в реальных 80 ячейках таких случаев нет.",
        f"PriceShock полностью воспроизводит архивную формулу: avg_fuel_price / baseline_price_region − 1. "
        f"Baseline — {meta['baseline_month']}; по Задание.md ограничения начинаются в конце мая. "
        "Цена = SUM(physical_liters × transaction_price) / SUM(physical_liters). "
        "Совпадение цен и PriceShock проверено для всех 80 ячеек. Для HIGH и LOW используется одна общая региональная цена.",
        "Все запрошенные регрессии — невзвешенные pooled OLS с константой. Стандартные ошибки сгруппированы по регионам, "
        "используются поправка малой выборки и t(G−1), G = 16. β дан на единицу PriceShock, то есть на 100 п.п.; "
        "изменение при +10 п.п. равно 0,1 × β. R² относится к исходной OLS и не меняется от способа расчёта ошибок. "
        "В индивидуальной проверке N — клиент-месяцы, но независимых групп для расчёта ошибок по-прежнему 16.",
    ]
    cluster_text = (
        f"Использованы сохранённые NO_LITERS метки из archive_monthly_risk_clustering, без переобучения. "
        f"В группах {meta['n_current_risk_driver_months']:,} клиент-месяцев; {meta['unclassified_drivers']} клиент "
        f"({meta['unclassified_driver_months']} клиент-месяцев) отсутствует из-за пропусков характеристик в старой модели. "
        "В основном анализе эти клиенты сохранены. Суммы HIGH + LOW сверены с классифицированной частью полной выборки. "
        "NO_LITERS не использует литры, но использует штрафы текущего месяца: различия fines/liters между текущими группами "
        "частично механически заданы самим способом их формирования. Это exploratory/descriptive анализ.")
    current_summary = []
    for outcome, part in current.groupby("outcome", sort=False):
        current_summary.append(LABELS[outcome] + ": " + "; ".join(
            f"{row.population}: β={row.BETA:.3f}, p={p_text(row.p_value)}" for row in part.itertuples()) + ".")
    lag_text = (
        "Метка берётся строго из предыдущего календарного месяца, исходы — из текущего. Для мая используются апрельские группы, "
        "для августа — июльские. При отсутствии предыдущего месяца метка не переносится через разрыв. "
        f"Получено {meta['n_lagged_risk_driver_months']:,} клиент-месяцев и по 64 ячейки на группу за май–август. "
        "Эта проверка убирает прямое использование штрафов текущего месяца при назначении его группы, "
        "но не устраняет отбор по прошлому поведению и не обеспечивает причинную идентификацию. "
        "Она также отличается от основного расчёта отсутствием апреля, поэтому изменение значимости нельзя приписывать только лагу.")
    lag_details = " ".join(f"{row.population}, {LABELS[row.outcome].lower()}: β={row.BETA:.3f}, "
                           f"p={p_text(row.p_value)} ({'значимо' if row.p_value < .05 else 'незначимо'} на 5%)."
                           for row in lagged.itertuples())
    robustness = (
        "Пороги 10, 20 и 40 литров зафиксированы в analysis_plan.json до оценки регрессий. "
        f"Основной порог — 20 л: он сохраняет {primary_threshold.share_positive_driver_months:.2%} положительных клиент-месяцев "
        "и убирает самые маленькие знаменатели. При 20 л один штраф соответствует 5 штрафам на 100 л; "
        "при 10 л — 10, при 40 л — 2,5. Это объясняет чувствительность индивидуального отношения к малым объёмам. "
        "Все три порога показаны, лучший p-value не выбирался. "
        f"Знак {'одинаков' if same_sign else 'меняется'}; оценки {'значимы при всех порогах' if significant else 'не значимы при всех порогах'} на уровне 5%. "
        "Для каждого порога оценивается driver-month OLS индивидуального 100 × fines_i / liters_i на региональный PriceShock. "
        "Эта проверка относится к покупателям с достаточным объёмом, меняет состав выборки и не заменяет отношение сумм по полной когорте.")
    limitations = [
        f"Нулевые покупки у {meta['zero_liter_driver_months']:,} клиент-месяцев ({zero_share:.1%}); "
        f"в {meta['zero_liters_with_fines']:,} из них есть штрафы. В таких месяцах зарегистрировано {meta['fines_in_zero_liter_months']:,} "
        f"штрафа ({zero_fine_share:.1%} всех штрафов). Покупки топлива в сервисе не являются полным учётом поездок: "
        "возможны покупки вне сервиса и использование ранее купленного топлива. Эти штрафы остаются в основном числителе, как требует отношение сумм всей когорты.",
        "Литры — proxy наблюдаемой топливной активности, не точный пробег. Положительная связь штрафов на литр с ценой "
        "может возникать, когда знаменатель сокращается быстрее числителя. Она не доказывает ни ухудшения вождения, ни психологической реакции водителей.",
        "Pooled OLS смешивает межрегиональные и временные различия; фиксированные эффекты региона и месяца не включены, "
        "как и в архивной простой регрессии. Нет экзогенного ценового воздействия, контроля сезонности и состава топлива. "
        "Группировка стандартных ошибок не устраняет эти ограничения. Всего пять месяцев и 16 регионов.",
        f"Сохранена наблюдаемая когорта v2. В источнике {meta['fines_before_subscription']} штрафов до subscription_creation_date "
        f"и {meta['clients_subscription_after_window']} клиентов с подпиской после конца окна. "
        "Дата подписки не использована как начало вождения: в данных реально наблюдаются операции до неё. "
        "Это отличается от отбора для некоторых регрессий check_def.ipynb.",
        "Литры взяты из очищенного physical_fuel_volume_main с обработкой возвратов; отрицательные операции не преобразовывались через abs(). "
        "Цена взвешена покупками жителей региона, не обязательно внутри региона, и может изменяться из-за состава покупаемого топлива.",
        "Групповые и пороговые p-value являются номинальными, без поправки на множественные проверки. "
        "Основные три теста отделены от exploratory и robustness. Разница между «значимо» и «незначимо» у HIGH/LOW сама по себе "
        "не доказывает статистического различия двух коэффициентов.",
        "Незначимый коэффициент штрафов на водителя не означает точного нулевого эффекта: его доверительный интервал допускает снижение. "
        "Но заявлять подтверждённую цепочку «цена → меньше активности → меньше штрафов» на этой основе нельзя.",
    ]
    regional_text = (
        "Регионы выбраны по финальному (август к апрелю) росту цены: минимум, верхняя медианная позиция и максимум; "
        "при равенстве — по коду региона. Исходы не использовались при выборе. "
        "Дополнительно построены четыре сетки динамики всех 16 регионов: цена, литры на водителя, штрафы на водителя, штрафы на 1000 л. "
        "Внутри каждой сетки единый масштаб Y, регионы упорядочены по росту цены. Национальная динамика рассчитана из общих сумм, "
        "а не как невзвешенное среднее региональных отношений. Она не обязана быть монотонной: месячная динамика и pooled коэффициент — разные описания данных.")
    presentation = (
        "Для основной презентации: 01_price_vs_liters_per_driver.png показывает значимое снижение наблюдаемых покупок; "
        "02_price_vs_fines_per_driver.png показывает, что снижение штрафов не установлено; "
        "03_price_vs_fines_per_1000_liters.png показывает рост отношения штрафов к литрам. "
        "Три графика вместе отделяют изменение активности от нормированной интенсивности. "
        "Кластеризацию рекомендовано перенести в robustness / appendix: основной вывод виден без неё, "
        "а текущие метки частично определены теми же штрафами. Лаговые результаты стоит показать рядом как ограничение устойчивости группового вывода, "
        "не как доказательство различий реакции HIGH и LOW. Региональная динамика подходит для дополнительного слайда.")
    sections = [
        ("Три экономических ответа", "\n\n".join(answers), result_table(main)),
        ("Как классифицировать результат", classification, None),
        ("Выборка, источники и формулы", "\n\n".join(methods), None),
        ("Итоговая таблица: основной анализ и текущие группы", cluster_text + "\n\n" + "\n\n".join(current_summary), result_table(main_and_groups)),
        ("Группы предыдущего месяца", lag_text + "\n\n" + lag_details, result_table(lagged)),
        ("Распределение литров", "Квантили рассчитаны до выбора порогов. Основная выборка включает нулевые покупки.", distribution),
        ("Индивидуальные отношения: проверка порогов", robustness, threshold_table),
        ("Региональная динамика", regional_text, selected_table),
        ("Общая динамика по месяцам", "Числители и знаменатели суммируются по всем регионам.",
         national[["month", "n_drivers", "liters_per_driver", "fines_per_driver", "fines_per_1000_liters", "avg_fuel_price"]]),
        ("Ограничения и интерпретация", "\n\n".join(limitations), None),
        ("Что вынести в презентацию", presentation, None),
        ("Архив и воспроизводимость", "monthly_risk_analysis переименована в archive_monthly_risk_clustering. "
         "Все прежние таблицы, графики и аналитический код сохранены. Изменены только начало README, команды запуска и docstring запуска с новым именем пакета. "
         "Предыдущая проверка PriceShock → ShareHighRisk остаётся зафиксированным статистически незначимым результатом. "
         "Архивный запуск: .venv/bin/python -m archive_fines_per_liter_analysis.run_analysis. "
         "Состав источников и SHA-256 находятся в metadata.json; правила выбора порогов и регионов — в analysis_plan.json.", None),
    ]
    md = ["# Штрафы относительно наблюдаемой топливной активности\n"]
    html_sections = []
    for title, body, table in sections:
        md.extend([f"## {title}\n", body + "\n"])
        html = f"<section><h2>{escape(title)}</h2>" + "".join(f"<p>{escape(p)}</p>" for p in body.split("\n\n"))
        if table is not None:
            md.append(markdown_table(table) + "\n")
            html += '<div class="table">' + table.to_html(index=False, border=0, float_format=lambda x: f"{x:.5g}") + '</div>'
        html_sections.append(html + "</section>")
    figure_names = list(FILENAMES.values())
    md.extend(f"![{name}](figures/{name}.png)\n" for name in figure_names)
    md.append("## Источники и версии\n")
    md.extend(f"- {name}: `{item['path']}`; SHA-256 `{item['sha256']}`" for name, item in meta["sources"].items())
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    summary_figures = "".join(f'<img src="figures/{name}.png" alt="{name}" loading="lazy">' for name in figure_names)
    extra_figures = "".join(f'<img src="figures/{name}.png" alt="{name}" loading="lazy">' for name in
                            ["04_selected_regional_dynamics", "05_national_monthly_dynamics", "06_liters_distribution"])
    links = " ".join(f'<a href="figures/regional_{metric}.png">{escape(label)}</a>' for metric, label in LABELS.items())
    (out / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Штрафы и топливная активность</title><style>body{margin:0;color:#1b344f;background:#f2f5f8;font:16px/1.6 system-ui,sans-serif}'
        'main{max-width:1220px;margin:36px auto;padding:0 24px}h1{font-size:34px;line-height:1.25}h2{font-size:24px}'
        'section{background:white;padding:24px 30px;border-radius:12px;margin:24px 0}p{max-width:108ch}'
        'img{display:block;width:100%;margin:24px auto;border-radius:10px}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}'
        'th,td{text-align:left;padding:10px;border-bottom:1px solid #e3e9f0}th{background:#edf2f8}a{color:#216397;margin-right:12px}</style>'
        '<main><h1>Штрафы относительно наблюдаемой топливной активности</h1><p>Апрель–август 2026 · 16 регионов · 80 наблюдений</p>'
        '<p><a href="report.md">Полный отчёт</a><a href="tables/regression_summary.csv">Все регрессии CSV</a>'
        '<a href="tables/region_month.csv">Региональная панель CSV</a></p>'
        + "".join(html_sections[:2]) + summary_figures + "".join(html_sections[2:]) + extra_figures
        + '<section><h2>Графики всех регионов</h2><p>' + links + '</p></section></main></html>', encoding="utf-8")
