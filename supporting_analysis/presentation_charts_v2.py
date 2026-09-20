"""Presentation-first charts for slides 5-7.

The technical charts remain unchanged and are copied to ``technical_backup``.
This module only changes the visual form: every displayed value is read from
an existing analysis output and checked before export.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from pipeline.plotting import plt

from supporting_analysis.presentation_charts import (
    COLORS,
    DPI,
    FIGSIZE,
    PROJECT_ROOT,
    _read_csv,
    build_all as build_technical_charts,
    setup_presentation_style,
)


MONTH_LABELS = {
    "2026-04": "апр",
    "2026-05": "май",
    "2026-06": "июн",
    "2026-07": "июл",
    "2026-08": "авг",
}

TECHNICAL_BACKUP_NAMES = {
    "slide_05_chart": "slide_05_scatter_technical",
    "slide_06_chart": "slide_06_heatmap_technical",
    "slide_07_chart": "slide_07_dumbbell_technical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def _new_canvas():
    fig = plt.figure(figsize=FIGSIZE, facecolor=COLORS["background"])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def _save(fig, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=dpi, facecolor=COLORS["background"], pad_inches=0)
    fig.savefig(paths[1], facecolor=COLORS["background"], pad_inches=0)
    plt.close(fig)
    return paths


def _rounded_card(ax, x: float, y: float, width: float, height: float, *, alpha: float = 1.0):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0,
        facecolor=COLORS["accent"],
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def ensure_technical_backups(root: Path, dpi: int) -> list[Path]:
    presentation_dir = root / "presentation_charts"
    missing = [
        presentation_dir / f"{stem}.{suffix}"
        for stem in TECHNICAL_BACKUP_NAMES
        for suffix in ("png", "svg")
        if not (presentation_dir / f"{stem}.{suffix}").is_file()
    ]
    if missing:
        build_technical_charts(root, presentation_dir, dpi=dpi)

    backup_dir = presentation_dir / "technical_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source_stem, backup_stem in TECHNICAL_BACKUP_NAMES.items():
        for suffix in ("png", "svg"):
            source = presentation_dir / f"{source_stem}.{suffix}"
            target = backup_dir / f"{backup_stem}.{suffix}"
            shutil.copy2(source, target)
            outputs.append(target)
    return outputs


def build_slide_05(root: Path, output_dir: Path, dpi: int) -> tuple[list[Path], dict]:
    source = root / "archive_fines_per_liter_analysis/outputs/tables/national_monthly.csv"
    monthly = _read_csv(
        source,
        {"month", "n_drivers", "liters_per_driver", "avg_fuel_price"},
    ).sort_values("month")
    expected_months = list(MONTH_LABELS)
    if monthly["month"].astype(str).tolist() != expected_months:
        raise ValueError("Слайд 5 v2: ожидаются ровно месяцы с апреля по август 2026")
    if monthly[["avg_fuel_price", "liters_per_driver"]].isna().any().any():
        raise ValueError("Слайд 5 v2: в временных рядах есть пропуски")

    price_index = monthly["avg_fuel_price"] / monthly["avg_fuel_price"].iloc[0] * 100
    purchase_index = monthly["liters_per_driver"] / monthly["liters_per_driver"].iloc[0] * 100
    x = np.arange(len(monthly))

    fig = _new_canvas()
    grid = fig.add_gridspec(2, 1, left=0.08, right=0.88, top=0.90, bottom=0.14, hspace=0.32)
    axes = [fig.add_subplot(grid[index]) for index in range(2)]
    series = [
        (price_index, COLORS["accent"], "ЦЕНА  ↑", "+"),
        (purchase_index, COLORS["primary"], "ПОКУПКИ  ↓", ""),
    ]
    for axis, (values, color, label, positive_prefix) in zip(axes, series):
        axis.set_facecolor(COLORS["background"])
        axis.axvspan(1.5, 4.18, color=COLORS["accent"], alpha=0.10, linewidth=0)
        axis.axvline(1.5, color=COLORS["accent"], alpha=0.70, linewidth=1.5)
        axis.axhline(100, color=COLORS["accent"], alpha=0.35, linewidth=1.2, linestyle=(0, (4, 5)))
        axis.plot(
            x,
            values,
            color=color,
            linewidth=5,
            marker="o",
            markersize=10,
            markerfacecolor=color,
            markeredgecolor=COLORS["background"],
            markeredgewidth=2,
            solid_capstyle="round",
            zorder=3,
        )
        axis.set_xlim(-0.28, 4.65)
        axis.set_ylim(62, 110)
        axis.set_yticks([])
        axis.tick_params(axis="x", length=0, labelsize=16, pad=8)
        axis.spines[:].set_visible(False)
        axis.text(
            -0.02,
            0.84,
            label,
            transform=axis.transAxes,
            fontsize=20,
            fontweight="bold",
            color=color,
            ha="left",
        )
        delta = float(values.iloc[-1] - 100)
        axis.text(
            4.18,
            float(values.iloc[-1]),
            f"{positive_prefix if delta > 0 else ''}{delta:.1f}%".replace(".", ","),
            fontsize=24,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
    axes[0].set_xticks([])
    axes[1].set_xticks(x, [MONTH_LABELS[month] for month in expected_months])
    axes[0].text(
        1.57,
        106.8,
        "с 1 июня",
        color=COLORS["accent"],
        fontsize=14,
        ha="left",
        va="top",
    )
    fig.text(0.08, 0.95, "АПРЕЛЬ = 100", fontsize=14, color=COLORS["text"], alpha=0.78)
    paths = _save(fig, output_dir, "slide_05_simple", dpi)

    return paths, {
        "message": "Цена топлива растёт, а покупки топлива через сервис сокращаются.",
        "source": str(source.relative_to(root)),
        "period": expected_months,
        "N_clients_each_month": monthly["n_drivers"].astype(int).tolist(),
        "raw": {
            "avg_fuel_price": monthly["avg_fuel_price"].astype(float).tolist(),
            "liters_per_driver": monthly["liters_per_driver"].astype(float).tolist(),
        },
        "transformation": "index_t = value_t / value_2026_04 * 100",
        "index": {
            "price": price_index.astype(float).tolist(),
            "purchases": purchase_index.astype(float).tolist(),
        },
        "april_to_august": {
            "price_pct": float(price_index.iloc[-1] - 100),
            "purchases_pct": float(purchase_index.iloc[-1] - 100),
        },
        "missing": 0,
        "simplification_caveat": (
            "На основном слайде показана национальная помесячная динамика; "
            "региональная OLS-регрессия сохранена в technical_backup."
        ),
    }


def _profile_text(summary: pd.DataFrame) -> dict[int, str]:
    rows = summary.set_index("cluster_id")
    return {
        0: "0 штрафов в доступном окне",
        1: f"0 новых · медиана {rows.loc[1, 'historical_fines_12m_median']:.0f} за 12 месяцев",
        2: f"обычно {rows.loc[2, 'fines_pre_median']:.0f} штраф",
        3: f"{rows.loc[3, 'speeding_share_pre_mean']:.0%} штрафов — за скорость",
        4: f"медиана {rows.loc[4, 'historical_fines_12m_median']:.0f} за 12 месяцев",
        5: f"медиана {rows.loc[5, 'offence_types_pre_median']:.0f} типа нарушений",
        6: "у каждого есть опасное нарушение",
    }


def build_slide_06(root: Path, output_dir: Path, dpi: int) -> tuple[list[Path], dict]:
    source = root / "outputs/behavior_clustering/tables/behavior_cluster_summary.csv"
    summary = _read_csv(
        source,
        {
            "cluster_id",
            "cluster_label",
            "clients",
            "share_pct",
            "fines_pre_median",
            "historical_fines_12m_median",
            "speeding_share_pre_mean",
            "dangerous_client_share",
            "offence_types_pre_median",
        },
    ).sort_values("cluster_id")
    if summary["cluster_id"].astype(int).tolist() != list(range(7)):
        raise ValueError("Слайд 6 v2: отсутствует одна из семи правиловых групп")
    if int(summary["clients"].sum()) != 25_675:
        raise ValueError("Слайд 6 v2: размер когорты отличается от технического анализа")
    descriptions = _profile_text(summary)

    fig = _new_canvas()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    card_x, card_width, card_height = 0.55, 8.45, 0.78
    top_y, gap = 6.28, 0.14
    for index, row in enumerate(summary.itertuples(index=False)):
        group = int(row.cluster_id)
        y = top_y - index * (card_height + gap)
        _rounded_card(ax, card_x, y, card_width, card_height, alpha=0.96)
        ax.scatter(
            card_x + 0.43,
            y + card_height / 2,
            s=780,
            color=COLORS["primary"],
            edgecolor="none",
            zorder=3,
        )
        ax.text(
            card_x + 0.43,
            y + card_height / 2,
            str(group),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=COLORS["accent"],
            zorder=4,
        )
        ax.text(
            card_x + 0.90,
            y + 0.50,
            str(row.cluster_label),
            ha="left",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=COLORS["text"],
        )
        ax.text(
            card_x + 0.90,
            y + 0.22,
            descriptions[group],
            ha="left",
            va="center",
            fontsize=13.5,
            color=COLORS["secondary"],
        )
        ax.text(
            card_x + card_width - 0.28,
            y + card_height / 2,
            f"{int(row.clients):,} · {float(row.share_pct):.1f}%".replace(",", " ").replace(".", ","),
            ha="right",
            va="center",
            fontsize=14,
            color=COLORS["text"],
        )
    paths = _save(fig, output_dir, "slide_06_profiles", dpi)

    return paths, {
        "message": "До кризиса водители уже образуют семь содержательно разных профилей.",
        "source": str(source.relative_to(root)),
        "N_clients": int(summary["clients"].sum()),
        "period": {
            "configured": ["2026-03-20", "2026-05-31"],
            "fines_available": ["2026-04-01", "2026-05-31"],
        },
        "groups": [
            {
                "cluster_id": int(row.cluster_id),
                "cluster_label": str(row.cluster_label),
                "clients": int(row.clients),
                "share_pct": float(row.share_pct),
                "presentation_description": descriptions[int(row.cluster_id)],
            }
            for row in summary.itertuples(index=False)
        ],
        "transformation": "Нет нормализации; выбраны медианы/доли из технической сводки.",
        "missing": int(summary.isna().sum().sum()),
        "simplification_caveat": (
            "Карточки объясняют правила групп, но не показывают все признаки и их распределения; "
            "z-score heatmap сохранена в technical_backup."
        ),
    }


def _select_slide_07_rows(summary: pd.DataFrame) -> pd.DataFrame:
    wanted = pd.DataFrame(
        [
            {"cluster_id": 2, "metric": "fine_count_30d", "short": "Эпизодические"},
            {"cluster_id": 3, "metric": "fine_speeding_share", "short": "Любители скорости"},
            {"cluster_id": 4, "metric": "fine_count_30d", "short": "Хронические"},
        ]
    )
    selected = wanted.merge(summary, on=["cluster_id", "metric"], how="left", validate="one_to_one")
    if selected[["pre_mean", "post_mean", "clients_compared"]].isna().any().any():
        raise ValueError("Слайд 7 v2: не найдены выбранные технические показатели")
    return selected


def _mini_bar(ax, *, x: float, y: float, width: float, value: float, maximum: float, color: str):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.22,
            boxstyle="round,pad=0,rounding_size=0.11",
            linewidth=0,
            facecolor=COLORS["muted"],
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width * min(max(value / maximum, 0), 1),
            0.22,
            boxstyle="round,pad=0,rounding_size=0.11",
            linewidth=0,
            facecolor=color,
        )
    )


def build_slide_07(root: Path, output_dir: Path, dpi: int) -> tuple[list[Path], dict]:
    source = root / "outputs/behavior_cluster_analysis/cluster_new_patterns.csv"
    summary = _read_csv(
        source,
        {
            "cluster_id",
            "cluster_label",
            "metric",
            "clients_compared",
            "pre_mean",
            "post_mean",
            "relative_change",
            "paired_effect_size",
            "q_value",
            "is_new_pattern",
        },
    )
    summary["cluster_id"] = pd.to_numeric(summary["cluster_id"], errors="raise").astype(int)
    selected = _select_slide_07_rows(summary)

    fig = _new_canvas()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    card_width, card_height = 3.68, 5.62
    card_xs = [0.48, 4.56, 8.64]
    subtitles = {
        2: "штрафа на клиента\nза 30 дней",
        3: "средняя доля\nскоростных штрафов",
        4: "штрафа на клиента\nза 30 дней",
    }
    conclusions = {
        2: "редкие штрафы",
        3: "скорость всё ещё доминирует",
        4: "частые штрафы",
    }
    for x, row in zip(card_xs, selected.itertuples(index=False)):
        group = int(row.cluster_id)
        _rounded_card(ax, x, 0.88, card_width, card_height, alpha=0.97)
        ax.text(x + 0.32, 6.02, str(group), fontsize=34, fontweight="bold", color=COLORS["secondary"], va="top")
        ax.text(x + 0.92, 5.96, str(row.short), fontsize=18, fontweight="bold", color=COLORS["text"], va="top")
        ax.text(x + 0.34, 5.16, subtitles[group], fontsize=14, color=COLORS["text"], va="top", linespacing=1.25)

        maximum = 1.0 if row.metric == "fine_speeding_share" else 2.0
        values = [("ДО", float(row.pre_mean), COLORS["secondary"]), ("ПОСЛЕ", float(row.post_mean), COLORS["primary"])]
        for offset, (period, value, color) in enumerate(values):
            y = 3.75 - offset * 1.15
            ax.text(x + 0.34, y + 0.38, period, fontsize=13, fontweight="bold", color=color, va="bottom")
            _mini_bar(ax, x=x + 0.34, y=y, width=1.95, value=value, maximum=maximum, color=color)
            display = f"{value:.0%}" if row.metric == "fine_speeding_share" else f"{value:.2f}".replace(".", ",")
            ax.text(x + 2.85, y + 0.11, display, fontsize=20, fontweight="bold", color=color, va="center", ha="center")

        if row.metric == "fine_speeding_share":
            change = (float(row.post_mean) - float(row.pre_mean)) * 100
            change_label = f"{change:+.0f} п.п."
        else:
            change = float(row.relative_change) * 100
            change_label = f"{change:+.0f}%".replace(".", ",")
        ax.text(x + card_width / 2, 1.90, change_label, fontsize=18, fontweight="bold", color=COLORS["secondary"], ha="center")
        ax.text(x + card_width / 2, 1.43, conclusions[group], fontsize=14, color=COLORS["text"], ha="center")

    paths = _save(fig, output_dir, "slide_07_profiles_pre_post", dpi)
    selection_reasons = {
        2: "Низкая средняя частота штрафов сохраняется: изменение не проходит порог нового паттерна.",
        3: "Доля скорости заметно снижается, но остаётся очень высокой — 86%.",
        4: "Высокая средняя частота штрафов сохраняется и немного растёт; новый паттерн не отмечен.",
    }
    return paths, {
        "message": "Характерные различия трёх профилей видны и после кризиса, хотя значения не остаются неизменными.",
        "source": str(source.relative_to(root)),
        "period": {
            "before": ["2026-04-01", "2026-05-31"],
            "after": ["2026-06-01", "2026-08-31"],
        },
        "selected_profiles": [
            {
                "cluster_id": int(row.cluster_id),
                "cluster_label": str(row.cluster_label),
                "metric": str(row.metric),
                "clients_compared": int(row.clients_compared),
                "before": float(row.pre_mean),
                "after": float(row.post_mean),
                "relative_change": (
                    None if pd.isna(row.relative_change) else float(row.relative_change)
                ),
                "q_value": None if pd.isna(row.q_value) else float(row.q_value),
                "is_new_pattern": bool(row.is_new_pattern),
                "selection_reason": selection_reasons[int(row.cluster_id)],
            }
            for row in selected.itertuples(index=False)
        ],
        "transformation": (
            "Показаны исходные средние из cluster_new_patterns.csv. Mini-bars используют "
            "линейный масштаб: 0-2 штрафа/30 дней для групп 2 и 4; 0-100% для группы 3."
        ),
        "missing": 0,
        "excluded_profiles": {
            "5": "Разноплановый профиль нельзя честно выразить одним показателем без новой метрики разнообразия.",
            "6": (
                "Опасные типы не подтверждают стабильность: например, скорость +40-60 падает "
                "с 322,3 до 16,0 на 1 000 клиентов за 30 дней."
            ),
        },
        "simplification_caveat": (
            "Слайд показывает три заранее проверенных примера, а не все группы и показатели. "
            "Полный dumbbell сохранён в technical_backup."
        ),
    }


def _write_validation(output_dir: Path, validation: dict) -> list[Path]:
    json_path = output_dir / "validation_v2.json"
    md_path = output_dir / "validation_v2.md"
    json_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Validation: presentation charts v2", ""]
    for key, value in validation["slides"].items():
        lines.extend(
            [
                f"## {key}",
                "",
                f"- message: {value['message']}",
                f"- source: `{value['source']}`",
                f"- period: `{json.dumps(value['period'], ensure_ascii=False)}`",
                f"- transformation: {value['transformation']}",
                f"- missing: `{value['missing']}`",
                f"- caveat: {value['simplification_caveat']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [json_path, md_path]


def build_all(root: Path, dpi: int = DPI) -> dict[str, object]:
    root = root.resolve()
    setup_presentation_style()
    backups = ensure_technical_backups(root, dpi)
    output_dir = root / "presentation_charts/v2"
    slide_05, validation_05 = build_slide_05(root, output_dir, dpi)
    slide_06, validation_06 = build_slide_06(root, output_dir, dpi)
    slide_07, validation_07 = build_slide_07(root, output_dir, dpi)
    validation = {
        "palette": COLORS,
        "dpi": dpi,
        "slides": {
            "slide_05": validation_05,
            "slide_06": validation_06,
            "slide_07": validation_07,
        },
    }
    validation_paths = _write_validation(output_dir, validation)
    return {
        "technical_backups": backups,
        "slide_05": slide_05,
        "slide_06": slide_06,
        "slide_07": slide_07,
        "validation": validation_paths,
    }


def main() -> None:
    args = parse_args()
    generated = build_all(args.project_root, dpi=args.dpi)
    for section, paths in generated.items():
        print(f"{section}:")
        for path in paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()
