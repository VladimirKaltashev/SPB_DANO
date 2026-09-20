"""Standalone research figures and a local report; no network assets."""

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

COLORS = {"full": "#2364AA", "no_liters": "#C35A28"}
MONTH_LABELS = {
    "2026-04": "Апр", "2026-05": "Май", "2026-06": "Июн",
    "2026-07": "Июл", "2026-08": "Авг",
}


def summarize_no_liters_regions(sample: pd.DataFrame) -> pd.DataFrame:
    """Build one readable change-summary row per region."""
    rows = []
    for (region, region_name), group in sample.groupby(
            ["region", "region_name"], sort=False, dropna=False):
        group = group.sort_values("month")
        first, last = group.iloc[0], group.iloc[-1]
        rows.append({
            "region": region,
            "region_name": region_name,
            "start_month": first.month,
            "end_month": last.month,
            "risk_start": first.share_high_risk,
            "risk_end": last.share_high_risk,
            "risk_change_pp": 100 * (last.share_high_risk - first.share_high_risk),
            "price_start_rub_l": first.avg_fuel_price,
            "price_end_rub_l": last.avg_fuel_price,
            "price_change_pct": 100 * (last.avg_fuel_price / first.avg_fuel_price - 1),
            "within_region_correlation": group.share_high_risk.corr(group.avg_fuel_price),
        })
    return pd.DataFrame(rows).sort_values("region_name").reset_index(drop=True)


def _padded_limits(values: pd.Series, include_zero: bool = False) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    padding = max((high - low) * 0.22, 0.8)
    return low - padding, high + padding


def draw_no_liters_regions(sample: pd.DataFrame, figures: Path) -> None:
    """Draw risk share and regional price change together for every region."""
    sample = sample.sort_values(["region_name", "month"]).copy()
    regions = list(sample.region_name.drop_duplicates())
    if not regions:
        return
    ncols = 4
    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 4.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, region_name in zip(axes, regions):
        part = sample.loc[sample.region_name.eq(region_name)].sort_values("month")
        months = np.arange(len(part))
        risk_pct = 100 * part.share_high_risk
        price_change_pct = 100 * part.price_shock
        ax_price = ax.twinx()

        ax.plot(months, risk_pct, color=COLORS["no_liters"], marker="o",
                markersize=5.5, linewidth=2.2, zorder=3)
        ax_price.plot(months, price_change_pct, color="#333333", marker="s",
                      markersize=4.5, linewidth=1.8, linestyle="--", zorder=2)
        weak = part.fines_ratio.lt(1.5)
        if weak.any():
            ax.scatter(months[weak], risk_pct[weak], s=105, facecolors="none",
                       edgecolors="#9A1B1B", linewidths=1.5, zorder=4)

        for x, risk, price in zip(months, risk_pct, price_change_pct):
            ax.annotate(f"{risk:.1f}", (x, risk), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7.3,
                        color=COLORS["no_liters"])
            ax_price.annotate(f"{price:+.1f}", (x, price), xytext=(0, -11),
                              textcoords="offset points", ha="center", fontsize=7.1,
                              color="#333333")

        risk_delta = risk_pct.iloc[-1] - risk_pct.iloc[0]
        price_delta = price_change_pct.iloc[-1]
        ax.set_title(region_name, fontsize=10.5, fontweight="bold", pad=20)
        ax.text(0.5, 1.015, f"апр→авг: риск {risk_delta:+.1f} п.п. · цена {price_delta:+.1f}%",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2,
                color="#555555")
        ax.set_xticks(months, [MONTH_LABELS.get(month, month) for month in part.month])
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        ax.tick_params(axis="y", colors=COLORS["no_liters"], labelsize=8)
        ax_price.tick_params(axis="y", colors="#333333", labelsize=8)
        ax.set_ylim(*_padded_limits(risk_pct))
        ax_price.set_ylim(*_padded_limits(price_change_pct, include_zero=True))
        ax.grid(axis="both", alpha=0.22)
        ax.spines[["top"]].set_visible(False)
        ax_price.spines[["top"]].set_visible(False)
        ax.spines["left"].set_color(COLORS["no_liters"])
        ax_price.spines["right"].set_color("#333333")
        ax.set_ylabel("high-risk, %", color=COLORS["no_liters"], fontsize=8.5)
        ax_price.set_ylabel("цена к апрелю, %", color="#333333", fontsize=8.5)

    for ax in axes[len(regions):]:
        ax.remove()
    handles = [
        Line2D([0], [0], color=COLORS["no_liters"], marker="o", lw=2.2,
               label="Доля high-risk, NO_LITERS (левая ось)"),
        Line2D([0], [0], color="#333333", marker="s", lw=1.8, ls="--",
               label="Изменение цены к апрелю (правая ось)"),
        Line2D([0], [0], color="#9A1B1B", marker="o", markerfacecolor="none",
               markersize=9, lw=0, label="Слабое различие кластеров по штрафам (<1,5×)"),
    ]
    fig.suptitle("NO_LITERS: доля high-risk и изменение цены в каждом регионе",
                 fontsize=18, y=0.992)
    fig.text(0.5, 0.965,
             "Оранжевые подписи — доля high-risk, %; серые — изменение цены к апрелю, %",
             ha="center", va="top", fontsize=10, color="#555555")
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.004), fontsize=10)
    fig.tight_layout(rect=[0.01, 0.045, 0.99, 0.94])
    for ext in ["png", "svg"]:
        fig.savefig(figures / f"no_liters_risk_and_price_by_region.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_figures(samples: dict, regression: pd.DataFrame,
                 monthly: pd.DataFrame, figures: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    for model, sample in samples.items():
        row = regression.loc[regression.model.eq(model.upper())].iloc[0]
        fig, ax = plt.subplots(figsize=(9, 6))
        for month, group in sample.groupby("month"):
            ax.scatter(group.price_shock, group.share_high_risk, label=month,
                       s=42, alpha=0.8, edgecolor="white", linewidth=0.4)
        x = np.linspace(sample.price_shock.min(), sample.price_shock.max(), 100)
        ax.plot(x, row.intercept + row.beta * x, color=COLORS[model], lw=2.2, label="Pooled OLS")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        baseline = sample.baseline_month.iloc[0]
        ax.set(xlabel=f"Изменение цены относительно {baseline}", ylabel="Доля high-risk")
        ax.set_title(f"{model.upper()}: цена топлива и доля high-risk\n"
                     f"β = {row.beta:.4f}   p = {row.p_value:.4f}   R² = {row.R_squared:.4f}   "
                     f"N = {int(row.N)}   регионов = {int(row.n_regions)}", fontsize=12, pad=12)
        for point in sample.loc[sample.share_high_risk.gt(0.5)].itertuples():
            ax.annotate(f"{point.region_name}, {point.month}",
                        (point.price_shock, point.share_high_risk), xytext=(14, -4),
                        textcoords="offset points", fontsize=9)
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=9)
        fig.text(0.12, 0.015, "Одна точка — регион × месяц. p: ошибки сгруппированы по региону. Связь описательная.", fontsize=8)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        for ext in ["png", "svg"]:
            fig.savefig(figures / f"price_shock_{model}.{ext}", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, group in monthly.groupby("model", sort=False):
        ax.plot(group.month, group.share_high_risk, marker="o", lw=2.2,
                color=COLORS[model], label=model.upper())
        for month, share in zip(group.month, group.share_high_risk):
            ax.annotate(f"{share:.1%}", (month, share), xytext=(0, 9 if model == "full" else -16),
                        textcoords="offset points", ha="center", color=COLORS[model], fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="Как меняется доля high-risk по месяцам", ylabel="Доля среди включённых водителей", xlabel="Месяц")
    ax.margins(y=0.25)
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(figures / f"monthly_high_risk_share.{ext}", dpi=180)
    plt.close(fig)
    draw_no_liters_regions(samples["no_liters"], figures)


def markdown_table(data: pd.DataFrame) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "/")
    return "\n".join(["| " + " | ".join(data.columns) + " |",
                      "| " + " | ".join(["---"] * len(data.columns)) + " |",
                      *["| " + " | ".join(map(fmt, row)) + " |" for row in data.itertuples(index=False, name=None)]])


def write_report(output: Path, metadata: dict, regression: pd.DataFrame,
                 profiles: pd.DataFrame, monthly: pd.DataFrame, cells: pd.DataFrame,
                 comparison: pd.DataFrame, regional_changes: pd.DataFrame) -> None:
    m = metadata
    result_table = regression[["model", "features", "beta", "p_value", "R_squared", "N", "n_regions"]]
    profile_table = profiles[["model", "risk_label", "n_driver_months", "mean_fines", "median_fines",
                              "mean_liters", "mean_engine_volume", "mean_horsepower"]]
    month_table = monthly.pivot(index="month", columns="model", values="share_high_risk").reset_index()
    ratios = []
    for model, part in profiles.groupby("model"):
        risk = part.set_index("risk_label")
        low = risk.loc["low_risk", "mean_fines"]
        ratio = risk.loc["high_risk", "mean_fines"] / low if low else np.inf
        local = cells.loc[cells.model.eq(model) & cells.status.eq("ok"), "fines_ratio"]
        ratios.append(f"{model.upper()}: среднее число штрафов high/low = {ratio:.2f}×; "
                      f"медиана отношения внутри ячеек {local.median():.2f}×, диапазон {local.min():.2f}–{local.max():.2f}×.")
    same_sign = np.sign(regression.beta).nunique() == 1
    negative_both = regression.beta.lt(0).all()
    significant_both = regression.p_value.lt(0.05).all()
    conclusion = (f"Направление коэффициентов {'совпадает' if same_sign else 'различается'}. "
                  f"Снижение доли high-risk при росте цены {'наблюдается' if negative_both else 'не наблюдается'} в обеих моделях. "
                  f"Обе оценки {'значимы' if significant_both else 'не являются одновременно значимыми'} на уровне 5% "
                  "при группировке ошибок по регионам. Причинный эффект эта регрессия не устанавливает.")
    start_month = regional_changes.start_month.iloc[0]
    end_month = regional_changes.end_month.iloc[0]
    risk_up = int(regional_changes.risk_change_pp.gt(1e-9).sum())
    risk_down = int(regional_changes.risk_change_pp.lt(-1e-9).sum())
    risk_flat = len(regional_changes) - risk_up - risk_down
    price_up = int(regional_changes.price_change_pct.gt(0).sum())
    corr_positive = int(regional_changes.within_region_correlation.gt(0).sum())
    national_no_liters = monthly.loc[monthly.model.eq("no_liters")].set_index("month")
    national_delta = 100 * (
        national_no_liters.loc[end_month, "share_high_risk"]
        - national_no_liters.loc[start_month, "share_high_risk"])
    regional_body = (
        f"На общем графике показана доля, а не количество водителей. Между {start_month} и {end_month} "
        f"национальная доля NO_LITERS изменилась всего на {national_delta:+.2f} п.п. "
        f"Цена выросла в {price_up} из {len(regional_changes)} регионов, но доля high-risk снизилась "
        f"только в {risk_down}, выросла в {risk_up} и почти не изменилась в {risk_flat}. "
        f"Корреляция уровней цены и доли внутри региона положительна в {corr_positive} из "
        f"{len(regional_changes)} регионов. При пяти месяцах это только наглядная диагностика, "
        "но она показывает отсутствие единой отрицательной реакции на рост цены. Красная обводка "
        "на графике отмечает месяцы, где два кластера слабо различаются по среднему числу штрафов."
    )
    regional_table = regional_changes[[
        "region_name", "risk_start", "risk_end", "risk_change_pp",
        "price_start_rub_l", "price_end_rub_l", "price_change_pct",
        "within_region_correlation",
    ]].rename(columns={
        "region_name": "region",
        "risk_start": f"high_risk_{start_month}",
        "risk_end": f"high_risk_{end_month}",
        "price_start_rub_l": f"price_{start_month}_rub_l",
        "price_end_rub_l": f"price_{end_month}_rub_l",
        "within_region_correlation": "corr_price_risk",
    })
    size = cells.groupby("model").agg(cells=("status", "size"), min_drivers=("n_drivers", "min"),
                                    max_drivers=("n_drivers", "max"), min_cluster_n=("min_cluster_n", "min"))
    excluded = cells.loc[cells.status.ne("ok"), ["model", "region", "month", "n_drivers", "status"]]
    weak = cells.loc[cells.status.eq("ok") & cells.fines_ratio.lt(1.5),
                     ["model", "region", "month", "mean_fines_high", "mean_fines_low", "fines_ratio"]]
    problems = [
        f"У {m['multi_car_clients']:,} клиентов несколько автомобилей. Использованы медианы объёма и мощности по известным значениям; это характеристики набора автомобилей клиента, а не установленного автомобиля каждой поездки.",
        f"Не удалось извлечь характеристики из {m['unparsed_vehicle_rows']} строк автомобилей. Исключено {m['unique_clients'] - m['eligible_clients']} клиентов ({m['driver_months'] - m['eligible_driver_months']} клиент-месяцев) без полной пары характеристик. Обе модели используют одинаковую выборку; импутации нет.",
        f"В {m['extreme_vehicle_rows']} строках автомобилей объём > 8 л или мощность > 1000 л.с. Значения сохранены и отмечены в аудите. StandardScaler не устраняет влияние выбросов на KMeans.",
        f"В периоде {m['negative_fuel_rows_in_window']} отрицательных топливных операций, из них {m['unresolved_negative_fuel_rows']} с неразрешённым возвратом. Использовано готовое physical_fuel_volume_main; отрицательные объёмы не превращались в покупки через abs().",
        f"{m['fines_before_subscription']} штрафов датированы до subscription_creation_date; у {m['clients_subscription_after_window']} клиентов эта дата позже конца периода. Сохранена наблюдаемая когорта v2 и все её штрафы. Дата подписки не трактуется как доказанная дата начала вождения; поведение до подписки есть в самих данных. В архивном check_def.ipynb другая выборка: штрафы до подписки исключаются.",
        "Регион — регистрация клиента (kladr_code), единая для штрафов, топлива и знаменателя. Регион АЗС отсутствует: цена отражает покупки жителей региона, а не обязательно цены АЗС внутри него.",
        "Полная когорта v2 включает месяцы без штрафов и заправок. Ноль означает отсутствие зарегистрированной операции в сервисе, а не доказанное отсутствие поездок. Пробег и покупки вне сервиса не наблюдаются.",
        "Кластеры заново обучаются в каждом месяце. High-risk — относительная метка по среднему числу штрафов внутри ячейки, а не единый абсолютный порог риска. Различие по штрафам частично задано способом построения и не служит независимой валидацией риска.",
        "Самарская область, май: около 78,5% водителей получили high-risk, хотя разница средних штрафов лишь 6–11%. Это следствие переназначения меток по среднему штрафов при слабом разделении. Ячейка не удалялась из оценки после просмотра результата; её следует интерпретировать осторожно.",
        "Pooled OLS не контролирует сезонность, постоянные региональные различия и изменение состава топлива. Всего 5 месяцев и 16 регионов; группировка ошибок учитывает зависимость внутри региона, но не делает вывод причинным. Средневзвешенная цена может меняться и из-за состава покупок.",
    ]
    methods = [
        f"Источник когорты: client_week_panel_v2.csv ({m['panel_rows']:,} строк). Реальный период {m['start']}–{m['end']}. По каждой неделе значения fines_count и fuel_liters полностью сверены с детальными очищенными файлами: расхождений нет.",
        f"Календарные месяцы построены по датам нарушений и заправок. В панели {m['month_crossing_week_rows']:,} строк недель пересекают границы месяцев; перенос всей недели в месяц её начала не использовался.",
        f"Получено {m['region_month_cells']} ячеек, {m['unique_clients']:,} уникальных клиентов и {m['driver_months']:,} клиент-месяцев до исключений. В моделях {m['eligible_clients']:,} клиентов и {m['eligible_driver_months']:,} клиент-месяцев.",
        f"В каждой ячейке отдельно: StandardScaler и KMeans(n_clusters=2, random_state=42, n_init=30). Минимум {m['min_drivers']} водителей в ячейке и {m['min_cluster_size']} в каждом полученном кластере. При равных средних штрафах риск не назначается. Пропущенные ячейки не входят в регрессию.",
        f"Baseline — {m['baseline_month']}, первый полный докризисный месяц; по Задание.md ограничения начались в конце мая. Цена = сумма(физические литры × цена операции) / сумма(физические литры), по всей когорте региона. PriceShock = цена месяца / апрельская цена региона − 1.",
        "Основная оценка: невзвешенная OLS share_high_risk ~ 1 + price_shock. Одна строка — регион × месяц. p-value использует стандартные ошибки с группировкой по регионам, поправку малой выборки и t-распределение с G−1 степенями свободы. IID p-value приведён отдельно только для справки.",
        "β измерен в долях на единицу PriceShock. Для роста цены на 10% изменение доли в процентных пунктах равно 10 × β. Национальная месячная доля взвешена количеством водителей: сумма high-risk / сумма водителей, а не среднее региональных долей.",
    ]
    notebook_review = (
        "Проверены все 38 ячеек archive_exploratory_notebooks/check_def.ipynb. "
        "В нём есть справочник регионов, очищенные продажи, средневзвешенные цены и панель 16 × 5. "
        "KMeans, StandardScaler, high_risk, share_high_risk, engine_volume, horsepower и price_shock отсутствуют. "
        "cov_type=\"cluster\" — стандартные ошибки регрессии с группировкой по регионам. "
        "В сохранённом выводе ячейки с оцениванием (индекс 35) есть ConvergenceWarning у Negative Binomial. "
        "Сохранённые оценки: β цены = 0.0183, p = 0.3282 у NB; β литров на клиента = −4.9741, p ≈ 0.0001. "
        "NB требует проверки сходимости. Эти результаты не являются новой кластеризацией. "
        "archive_insurance_driver_clustering содержит другую модель: кластеры по апрелю–маю и валидация на июне–августе. "
        "Старые файлы не изменялись. Логика региона и цены из check_def воспроизведена в автономном модуле; "
        "исполнение старого notebook не требуется.")
    sections = [
        ("Что показали две модели", conclusion, result_table),
        ("Что видно по отдельным регионам", regional_body, regional_table),
        ("Выборка и методика", "\n\n".join(methods), None),
        ("Размеры ячеек", "", size.reset_index()),
        ("Различие кластеров", "\n\n".join(ratios), profile_table),
        ("Ячейки со слабым различием по штрафам", "Отношение средних high/low < 1,5. Это диагностический флаг, а не правило исключения из регрессии.", weak),
        ("Доля high-risk по месяцам", "Значения в долях: 0.20 = 20%.", month_table),
        ("Согласие моделей", "Сравнение меток на одних и тех же клиент-месяцах.", comparison),
        ("Исключённые ячейки", "Нет." if excluded.empty else "Причины исключения:", None if excluded.empty else excluded),
        ("Проблемы данных и границы вывода", "\n\n".join(problems), None),
        ("Проверка архивного check_def.ipynb", notebook_review, None),
    ]
    md, html = ["# Месячная кластеризация риска\n"], []
    for title, body, table in sections:
        md.extend([f"## {title}\n", body + "\n"])
        section = f"<section><h2>{escape(title)}</h2>" + "".join(
            f"<p>{escape(p)}</p>" for p in body.split("\n\n") if p)
        if table is not None:
            md.append(markdown_table(table) + "\n")
            section += '<div class="table">' + table.to_html(index=False, float_format=lambda x: f"{x:.5g}", border=0) + "</div>"
        html.append(section + "</section>")
    figures = ["no_liters_risk_and_price_by_region", "monthly_high_risk_share",
               "price_shock_no_liters", "price_shock_full"]
    md.extend(f"![{name}](figures/{name}.png)\n" for name in figures)
    sources = "\n".join(f"- {name}: `{entry['path']}`; SHA-256 `{entry['sha256']}`" for name, entry in m["sources"].items())
    md.extend(["## Источники\n", sources, "\nВерсия Git: " + m["git_commit"]])
    (output / "report.md").write_text("\n".join(md), encoding="utf-8")
    charts = "".join(f'<img src="figures/{name}.png" alt="{name}" loading="lazy">' for name in figures)
    (output / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Месячная кластеризация риска</title><style>body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#172b42}'
        'main{max-width:1180px;margin:40px auto;padding:0 24px}h1{font-size:34px}h2{font-size:24px}'
        'section{background:white;padding:24px 30px;margin:20px 0;border-radius:12px}p{max-width:100ch}'
        '.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #e2e6ed}'
        'th{background:#eef2f7}img{display:block;width:100%;max-width:1000px;margin:24px auto;background:white;border-radius:10px}'
        'a{color:#2364aa}</style><main><h1>Месячная кластеризация риска</h1>'
        '<p>Апрель–август 2026 · FULL и NO_LITERS · '
        '<a href="report.md">Полный отчёт и источники</a> · <a href="tables/regression_summary.csv">Результаты CSV</a></p>'
        + html[0] + charts + "".join(html[1:]) + '</main></html>', encoding="utf-8")
