"""Локальный HTML-отчёт и проверка файлов изображений."""

from html import escape
from pathlib import Path

import pandas as pd
from PIL import Image

SUMMARY_CHARTS = {
    "cluster_pre_post_comparison.png": "Топливо и штрафы до и после кризиса",
    "cluster_monthly_trends.png": "Месячная динамика по группам",
    "cluster_change_heatmap.png": "Размеры изменений; н/д — недостаточно данных",
    "cluster_new_patterns.png": "Изменения, прошедшие статистические пороги",
    "cluster_fines_2025_vs_2026.png": "Штрафы: апрель–август 2025 и 2026",
    "fuel_price_timeline.png": "Цена топлива и дата разделения",
    "cluster_offence_type_heatmap.png": "Какие нарушения чаще встречаются в каждой группе",
}


def write_gallery(
    output_dir: Path,
    baseline: pd.DataFrame,
    offence_manifest: pd.DataFrame,
    changes: pd.DataFrame,
) -> list[str]:
    charts = list(SUMMARY_CHARTS.items())
    if not offence_manifest.empty:
        charts.extend(
            (row.chart_file, f"{row.cluster}. {row.cluster_label}")
            for row in offence_manifest.itertuples()
        )
    # Эти два изображения строит сегментация; отдельный анализ может получать
    # внешний CSV групп без соседней папки с их профилями.
    for filename, title in [
        ("behavior_cluster_sizes.png", "Размеры поведенческих групп"),
        ("behavior_cluster_profiles.png", "Профили поведенческих групп"),
    ]:
        relative = f"../behavior_clustering/figures/{filename}"
        if (output_dir / relative).is_file():
            charts.insert(0, (relative, title))
    for relative, _ in charts:
        with Image.open(output_dir / relative) as image:
            image.verify()

    groups = (
        baseline.groupby(["cluster", "cluster_label"], sort=True).size().reset_index(name="clients")
    )
    groups.columns = ["ID на графике", "Группа", "Клиентов"]
    highlights = changes.loc[changes["is_new_pattern"]].head(8)
    findings = (
        "".join(
            f"<li>{escape(str(row.cluster_label))}: {escape(str(row.metric_label))} "
            f"{row.pre_mean:.2f} → {row.post_mean:.2f}; q={row.q_value:.3g}.</li>"
            for row in highlights.itertuples()
        )
        or "<li>При заданных порогах устойчивых изменений не найдено.</li>"
    )
    cards = "\n".join(
        f'<section><h2>{escape(title)}</h2><a href="{escape(relative, quote=True)}">'
        f'<img src="{escape(relative, quote=True)}" alt="{escape(title, quote=True)}" '
        'loading="lazy"></a></section>'
        for relative, title in charts
    )
    document = f"""<!doctype html>
<html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Поведенческие группы — результаты анализа</title>
<style>
body{{max-width:1200px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#193044;background:#f4f7fa}}
h1,h2{{line-height:1.25}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}
img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;background:white}}
td,th{{text-align:left;padding:9px 14px;border-bottom:1px solid #dce3ea}}a{{color:#21649c}}
</style>
<h1>Поведенческие группы водителей</h1>
<p>Клиентов: {len(baseline):,}. Групп: {len(groups)}. Проверено изображений: {len(charts)}.</p>
<p>Группа определяется правилами по штрафам до 1 июня 2026 года и фиксируется
для дальнейшего сравнения. В основной группе действует приоритет:
опасные нарушения → разные типы → хронические → скорость → остальные.
Пересекающиеся признаки сохранены в тегах.</p>
<p>Частоты приведены к 30 дням. До/после сравниваются клиенты с достаточной
историей наблюдения. Годовое сравнение использует всю базу.
Изменения после кризиса сами по себе не доказывают его причинное влияние.</p>
<p><a href="analysis_summary.md">Полный текстовый отчёт</a> ·
<a href="cluster_new_patterns.csv">Таблица изменений</a></p>
{groups.to_html(index=False, border=0)}
<section><h2>Что изменилось</h2><ul>{findings}</ul></section>
{cards}
</html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    return [relative for relative, _ in charts]
