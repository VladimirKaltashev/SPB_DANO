"""Presentation-ready charts for slides 5–7 without changing analysis logic.

The module consumes tables produced by the existing technical analyses.  It
does not refit groups, redefine periods, or replace the technical metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline.plotting import plt
from matplotlib import font_manager, ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Exact colors used by the current Figma Slides template.
COLORS = {
    "background": "#BF8ED5",
    "text": "#1E0833",
    "primary": "#1E0833",
    "secondary": "#AB5998",
    "accent": "#FFFFFF",
    "muted": "#D1ABE3",
    "grid": "#AB5998",
}

FIGSIZE = (12.8, 7.2)  # 16:9
DPI = 300
CRISIS_START = pd.Timestamp("2026-06-01")

GROUP_LABELS = {
    0: "0. Без недавних нарушений",
    1: "1. Без новых штрафов, с историей",
    2: "2. Эпизодические",
    3: "3. Любители скорости",
    4: "4. Хронические",
    5: "5. Разноплановые",
    6: "6. Опасные",
}

SLIDE_07_GROUP_LABELS = {
    2: "2. Эпизодические",
    3: "3. Скоростные",
    4: "4. Хронические",
    5: "5. Разноплановые",
    6: "6. Опасные",
}

PROFILE_COLUMNS = {
    "fines_pre": "Штрафы\nдо кризиса",
    "fines_last_12m_total": "Штрафы\nза 12 месяцев",
    "speeding_share_pre": "Доля\nскорости",
    "dangerous_events_pre": "Опасные\nнарушения",
    "offence_types_pre": "Типы\nнарушений",
}

OFFENCE_LABELS = {
    "Превышение скорости на 20-40 км/ч": "Скорость +20–40",
    "Не пристегнут ремень безопасности": "Ремень",
    "Нарушение разметки": "Разметка",
    "Остальные": "Остальные",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="По умолчанию <project-root>/presentation_charts.",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def setup_presentation_style() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in installed else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 16,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "svg.fonttype": "none",
        }
    )
    return family


def new_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    return fig, ax


def format_axis(ax, *, grid: bool = True, remove_left: bool = False) -> None:
    ax.spines["bottom"].set_alpha(0.45)
    if remove_left:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_alpha(0.45)
    if grid:
        ax.grid(
            axis="y",
            color=COLORS["grid"],
            linewidth=0.8,
            alpha=0.22,
        )
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=15, width=0.8)


def format_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def save_presentation_chart(fig, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    fig.savefig(
        paths[0],
        dpi=dpi,
        bbox_inches="tight",
        facecolor=COLORS["background"],
    )
    fig.savefig(
        paths[1],
        bbox_inches="tight",
        facecolor=COLORS["background"],
    )
    plt.close(fig)
    return paths


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден источник технического анализа: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"В {path} отсутствуют столбцы: {', '.join(missing)}")
    return frame


def build_slide_05(root: Path, output_dir: Path, dpi: int) -> tuple[list[Path], dict]:
    """What to see in 3 seconds: higher price growth accompanies fewer liters."""
    tables = root / "archive_fines_per_liter_analysis/outputs/tables"
    samples = _read_csv(
        tables / "regression_samples.csv",
        {
            "region",
            "month",
            "price_shock",
            "liters_per_driver",
            "fitted",
            "outcome",
            "population",
            "block",
        },
    )
    plotted = samples.loc[
        samples["block"].eq("main")
        & samples["population"].eq("ALL")
        & samples["outcome"].eq("liters_per_driver")
    ].copy()
    regression = _read_csv(
        tables / "regression_summary.csv",
        {
            "block",
            "population",
            "outcome",
            "BETA",
            "SE",
            "p_value",
            "R_squared",
            "N",
            "n_regions",
            "ci95_low",
            "ci95_high",
            "intercept",
            "change_per_10pct_price",
        },
    )
    row = regression.loc[
        regression["block"].eq("main")
        & regression["population"].eq("ALL")
        & regression["outcome"].eq("liters_per_driver")
    ].squeeze()
    if len(plotted) != int(row["N"]):
        raise ValueError("Слайд 5: N не совпадает с технической регрессией")
    fitted_check = row["intercept"] + row["BETA"] * plotted["price_shock"]
    if not np.allclose(plotted["fitted"], fitted_check, rtol=0, atol=1e-9):
        raise ValueError("Слайд 5: fitted не совпадает с технической регрессией")

    national = _read_csv(
        tables / "national_monthly.csv",
        {"month", "n_drivers", "liters_per_driver", "avg_fuel_price"},
    ).sort_values("month")
    before = float(national.iloc[0]["liters_per_driver"])
    after = float(national.iloc[-1]["liters_per_driver"])
    delta = after - before
    delta_pct = delta / before * 100

    fig, ax = new_figure()
    ax.scatter(
        plotted["price_shock"],
        plotted["liters_per_driver"],
        s=64,
        color=COLORS["secondary"],
        alpha=0.55,
        edgecolor=COLORS["accent"],
        linewidth=0.7,
        zorder=3,
    )
    ordered = plotted.sort_values("price_shock")
    ax.plot(
        ordered["price_shock"],
        ordered["fitted"],
        color=COLORS["primary"],
        linewidth=4.2,
        zorder=4,
    )

    target_x = min(0.085, float(plotted["price_shock"].max()) * 0.65)
    target_y = float(row["intercept"] + row["BETA"] * target_x)
    change = float(row["change_per_10pct_price"])
    ax.annotate(
        f"+10 п.п. цены  →  {format_number(change, 0)} л\nна водителя в месяц",
        xy=(target_x, target_y),
        xytext=(0.095, 121),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=21,
        fontweight="bold",
        color=COLORS["primary"],
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["primary"],
            "linewidth": 1.5,
        },
    )
    p_value = float(row["p_value"])
    ax.text(
        0.006,
        27,
        f"p < 0,001 · 80 наблюдений\n16 регионов × 5 месяцев",
        fontsize=15,
        color=COLORS["text"],
        va="center",
    )
    ax.text(
        0.006,
        8,
        "Каждая точка — регион × месяц",
        fontsize=14,
        color=COLORS["text"],
        alpha=0.80,
    )
    ax.set_xlabel("Рост средней цены к апрелю, %", fontsize=18, labelpad=12)
    ax.set_ylabel("Топливо на водителя за месяц, л", fontsize=18, labelpad=12)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.set_xlim(-0.005, max(0.155, float(plotted["price_shock"].max()) * 1.05))
    ax.set_ylim(0, max(145, float(plotted["liters_per_driver"].max()) * 1.08))
    format_axis(ax)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.16)
    paths = save_presentation_chart(fig, output_dir, "slide_05_chart", dpi)

    validation = {
        "what_viewer_should_see": "Чем сильнее рост цены, тем меньше наблюдаемый объём топлива на водителя.",
        "metric": "liters_per_driver = total_liters / n_drivers",
        "N": int(row["N"]),
        "period": [str(national.iloc[0]["month"]), str(national.iloc[-1]["month"])],
        "before": before,
        "after": after,
        "delta": delta,
        "delta_pct": delta_pct,
        "missing": int(plotted[["price_shock", "liters_per_driver"]].isna().any(axis=1).sum()),
        "filters": (
            "ALL: полная когорта 25 675 клиентов; 16 регионов × 5 месяцев; "
            "включены клиент-месяцы с нулевыми покупками; pooled OLS без весов"
        ),
        "regression": {
            "beta": float(row["BETA"]),
            "se_clustered_by_region": float(row["SE"]),
            "p_value": p_value,
            "ci95": [float(row["ci95_low"]), float(row["ci95_high"])],
            "r_squared": float(row["R_squared"]),
            "change_per_10pp_price": change,
        },
        "technical_match": True,
        "technical_reference": "archive_fines_per_liter_analysis/outputs/figures/01_price_vs_liters_per_driver.png",
        "caveat": (
            "Pooled OLS показывает связь, а не причинный эффект; litres_per_driver "
            "является прокси объёма покупок топлива, а не пробега."
        ),
    }
    return paths, validation


def _profile_matrix(clients: pd.DataFrame, cluster_order: list[int]) -> pd.DataFrame:
    columns = list(PROFILE_COLUMNS)
    values = clients[columns]
    return (
        (clients.groupby("cluster_id")[columns].mean() - values.mean())
        .div(values.std().replace(0, np.nan))
        .reindex(cluster_order)
        .fillna(0)
    )


def build_slide_06(root: Path, output_dir: Path, dpi: int) -> tuple[list[Path], dict]:
    """What to see in 3 seconds: groups differ strongly before the crisis."""
    base = root / "outputs/behavior_clustering/tables"
    clients = _read_csv(
        base / "client_behavior_clusters.csv",
        {"client_id", "cluster_id", "cluster_label", *PROFILE_COLUMNS},
    )
    summary = _read_csv(
        base / "behavior_cluster_summary.csv",
        {"cluster_id", "cluster_label", "clients"},
    ).sort_values("cluster_id")
    if clients["client_id"].duplicated().any():
        raise ValueError("Слайд 6: client_id должен быть уникальным")
    counts = clients.groupby("cluster_id")["client_id"].nunique().reindex(summary["cluster_id"])
    if not np.array_equal(counts.to_numpy(), summary["clients"].to_numpy()):
        raise ValueError("Слайд 6: размеры групп не совпадают с технической сводкой")

    cluster_order = summary["cluster_id"].astype(int).tolist()
    matrix = _profile_matrix(clients, cluster_order)
    cmap = LinearSegmentedColormap.from_list(
        "presentation_profile",
        [COLORS["secondary"], COLORS["muted"], COLORS["primary"]],
    )

    fig, ax = new_figure()
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-3, vmax=3)
    ax.set_xticks(
        np.arange(len(PROFILE_COLUMNS)),
        list(PROFILE_COLUMNS.values()),
        fontsize=15,
    )
    row_labels = [GROUP_LABELS.get(group, str(group)) for group in cluster_order]
    ax.set_yticks(np.arange(len(cluster_order)), row_labels, fontsize=15)
    ax.tick_params(axis="x", pad=12, length=0)
    ax.tick_params(axis="y", pad=10, length=0)
    for row_index in range(len(matrix)):
        for column_index in range(len(matrix.columns)):
            value = float(matrix.iat[row_index, column_index])
            color = COLORS["accent"] if abs(value) >= 1.35 else COLORS["text"]
            ax.text(
                column_index,
                row_index,
                f"{value:+.1f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold" if abs(value) >= 1 else "normal",
                color=color,
            )
    for boundary in np.arange(-0.5, len(cluster_order), 1):
        ax.axhline(boundary, color=COLORS["background"], linewidth=3)
    for boundary in np.arange(-0.5, len(PROFILE_COLUMNS), 1):
        ax.axvline(boundary, color=COLORS["background"], linewidth=3)

    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.13)
    colorbar.set_ticks([-3, 0, 3])
    colorbar.set_ticklabels(["ниже среднего", "среднее", "выше среднего"])
    colorbar.ax.tick_params(labelsize=14, length=0, pad=7)
    colorbar.outline.set_visible(False)
    ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=0.275, right=0.98, top=0.97, bottom=0.22)
    paths = save_presentation_chart(fig, output_dir, "slide_06_chart", dpi)

    validation = {
        "what_viewer_should_see": "Группы уже до кризиса имеют разные профили нарушений.",
        "metric": "Среднее по группе, стандартизованное относительно всех клиентов (z-score)",
        "N": int(clients["client_id"].nunique()),
        "period": {
            "configured_pre_window": ["2026-03-20", "2026-05-31"],
            "fines_available_in_processed_source": ["2026-04-01", "2026-05-31"],
        },
        "before": None,
        "after": None,
        "delta": None,
        "missing": int(clients[list(PROFILE_COLUMNS)].isna().sum().sum()),
        "filters": (
            "Те же 7 правиловых групп; средние по fines_pre, fines_last_12m_total, "
            "speeding_share_pre, dangerous_events_pre, offence_types_pre; vmin/vmax −3/+3"
        ),
        "group_sizes": {
            str(int(row.cluster_id)): int(row.clients) for row in summary.itertuples()
        },
        "matrix": {
            str(group): {
                column: float(matrix.loc[group, column]) for column in matrix.columns
            }
            for group in cluster_order
        },
        "technical_match": True,
        "technical_reference": "outputs/behavior_clustering/figures/behavior_cluster_profiles.png",
        "caveat": (
            "Окно в коде начинается 20 марта, но fines_clean.csv фактически начинается "
            "1 апреля; мартовские штрафы отсутствуют. Матрица воспроизводит текущий "
            "технический результат без изменения методологии."
        ),
    }
    return paths, validation


def _top_offence_categories(data: pd.DataFrame, top_n: int = 3) -> list[str]:
    count_columns = ["fine_count_pre_crisis_2026", "fine_count_post_crisis_2026"]
    ranking = data.groupby("offence_short_statement")[count_columns].sum().sum(axis=1)
    return ranking.nlargest(top_n).index.tolist()


def _validate_offence_source(root: Path, technical: pd.DataFrame) -> dict:
    fines = _read_csv(
        root / "data/processed/fines_clean.csv",
        {"client_id", "bill_id", "bill_offence_date", "offence_short_statement"},
    )
    mapping = _read_csv(
        root / "outputs/behavior_clustering/tables/client_clusters.csv",
        {"client_id", "cluster_id"},
    )
    if fines["bill_id"].duplicated().any() or mapping["client_id"].duplicated().any():
        raise ValueError("Слайд 7: дубли bill_id или client_id")
    fines["bill_offence_date"] = pd.to_datetime(fines["bill_offence_date"], errors="raise")
    joined = fines.merge(mapping, on="client_id", how="left", validate="many_to_one")
    if joined["cluster_id"].isna().any():
        raise ValueError("Слайд 7: есть штрафы без группы")
    joined["period"] = np.where(joined["bill_offence_date"].lt(CRISIS_START), "pre", "post")
    rebuilt = (
        joined.groupby(["cluster_id", "offence_short_statement", "period"])["bill_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    checked = technical.merge(
        rebuilt,
        on=["cluster_id", "offence_short_statement"],
        how="left",
        validate="one_to_one",
    ).fillna({"pre": 0, "post": 0})
    pre_match = np.array_equal(
        checked["fine_count_pre_crisis_2026"].to_numpy(),
        checked["pre"].astype(int).to_numpy(),
    )
    post_match = np.array_equal(
        checked["fine_count_post_crisis_2026"].to_numpy(),
        checked["post"].astype(int).to_numpy(),
    )
    if not (pre_match and post_match):
        raise ValueError("Слайд 7: техническая таблица не совпадает с fines_clean.csv")
    return {
        "unique_bills": int(fines["bill_id"].nunique()),
        "unmapped_bills": 0,
        "mismatched_cells": 0,
        "pre_total_all_groups": int((joined["period"] == "pre").sum()),
        "post_total_all_groups": int((joined["period"] == "post").sum()),
    }


def build_slide_07(root: Path, output_dir: Path, dpi: int) -> tuple[list[Path], dict]:
    """What to see in 3 seconds: leading offence types remain broadly similar."""
    source = root / "outputs/behavior_cluster_analysis/cluster_offence_types_pre_post_2026.csv"
    technical = _read_csv(
        source,
        {
            "cluster_id",
            "cluster_label",
            "cluster_clients",
            "offence_short_statement",
            "fine_count_pre_crisis_2026",
            "fine_count_post_crisis_2026",
        },
    )
    audit = _validate_offence_source(root, technical)
    included = technical.loc[technical["cluster_id"].ge(2)].copy()
    categories = _top_offence_categories(included, top_n=3)
    plotted = included.loc[included["offence_short_statement"].isin(categories)].copy()
    plotted["category_order"] = plotted["offence_short_statement"].map(
        {category: index for index, category in enumerate(categories)}
    )
    plotted = plotted.sort_values(["cluster_id", "category_order"])

    fig, ax = new_figure()
    positions: list[float] = []
    labels: list[str] = []
    group_headings: dict[int, float] = {}
    cursor = 0.0
    for group in sorted(plotted["cluster_id"].unique()):
        part = plotted.loc[plotted["cluster_id"].eq(group)]
        group_headings[int(group)] = cursor
        cursor += 0.95
        for row in part.itertuples(index=False):
            before = float(row.fines_per_1000_clients_30d_pre)
            after = float(row.fines_per_1000_clients_30d_post)
            if before <= 0 or after <= 0:
                raise ValueError("Слайд 7: log-scale требует положительных частот")
            ax.plot(
                [before, after],
                [cursor, cursor],
                color=COLORS["muted"],
                linewidth=4,
                solid_capstyle="round",
                zorder=2,
            )
            ax.scatter(
                before,
                cursor,
                s=90,
                facecolor=COLORS["background"],
                edgecolor=COLORS["secondary"],
                linewidth=2.5,
                zorder=3,
            )
            ax.scatter(
                after,
                cursor,
                s=105,
                facecolor=COLORS["primary"],
                edgecolor=COLORS["primary"],
                linewidth=1.5,
                zorder=4,
            )
            positions.append(cursor)
            labels.append(OFFENCE_LABELS.get(row.offence_short_statement, row.offence_short_statement))
            cursor += 1.0
        cursor += 0.75

    ax.set_yticks(positions, labels, fontsize=14)
    for group, heading_y in group_headings.items():
        ax.text(
            -0.25,
            heading_y,
            SLIDE_07_GROUP_LABELS[group],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax.set_ylim(cursor - 0.35, -0.55)
    ax.set_xscale("log")
    ax.set_xlim(5, 3500)
    ax.set_xticks([10, 100, 1000])
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " "))
    )
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Штрафов на 1 000 клиентов за 30 дней · логарифмическая шкала", fontsize=17, labelpad=12)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["background"],
                markeredgecolor=COLORS["secondary"], markeredgewidth=2.2, markersize=10,
                label="до 1 июня",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["primary"],
                markeredgecolor=COLORS["primary"], markersize=10, label="после 1 июня",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        frameon=False,
        fontsize=15,
        columnspacing=2.5,
    )
    fig.text(
        0.985,
        0.025,
        "Группы 0–1: докризисный ноль задан правилом группы",
        ha="right",
        va="bottom",
        fontsize=13,
        color=COLORS["text"],
        alpha=0.82,
    )
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_alpha(0.45)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.22)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.31, right=0.98, top=0.92, bottom=0.20)
    paths = save_presentation_chart(fig, output_dir, "slide_07_chart", dpi)

    pre_total = int(plotted["fine_count_pre_crisis_2026"].sum())
    post_total = int(plotted["fine_count_post_crisis_2026"].sum())
    pre_rates = plotted["fines_per_1000_clients_30d_pre"]
    post_rates = plotted["fines_per_1000_clients_30d_post"]
    validation = {
        "what_viewer_should_see": "Лидирующий тип сохраняется, но величина изменений заметно различается между группами.",
        "metric": "Уникальных штрафов на 1 000 клиентов за 30 дней",
        "N": pre_total + post_total,
        "period": {
            "before": ["2026-04-01", "2026-05-31"],
            "after": ["2026-06-01", "2026-08-31"],
        },
        "before": {"min_rate": float(pre_rates.min()), "max_rate": float(pre_rates.max())},
        "after": {"min_rate": float(post_rates.min()), "max_rate": float(post_rates.max())},
        "delta": None,
        "missing": 0,
        "filters": (
            "Группы 2–6; группы 0–1 исключены из-за нуля, заданного правилом; "
            "топ-3 типа по сумме событий в показанных группах"
        ),
        "denominator": "Число клиентов группы и длина периода; нормировка к 30 дням",
        "event_counts_for_plotted_cells": {"before": pre_total, "after": post_total},
        "top_categories": categories,
        "source_audit": audit,
        "technical_match": True,
        "technical_reference": "outputs/behavior_cluster_analysis/cluster_offence_types_pre_post_2026.csv",
        "caveat": (
            "Частоты нормированы на 30 дней, но группы заданы по докризисным штрафам; "
            "это описательное сравнение, а не причинный эффект."
        ),
    }
    return paths, validation


def _validation_markdown(validation: dict[str, dict]) -> str:
    lines = ["# Validation: presentation charts", ""]
    for graph, values in validation.items():
        lines.extend(
            [
                f"## {graph.upper()}",
                "",
                f"- Что увидеть за 3 секунды: {values['what_viewer_should_see']}",
                f"- metric: `{values['metric']}`",
                f"- N: `{values['N']}`",
                f"- period: `{json.dumps(values['period'], ensure_ascii=False)}`",
                f"- before: `{values['before']}`",
                f"- after: `{values['after']}`",
                f"- delta: `{values['delta']}`",
                f"- missing: `{values['missing']}`",
                f"- filters: {values['filters']}",
                f"- technical_match: `{values['technical_match']}`",
                f"- technical_reference: `{values['technical_reference']}`",
                "",
            ]
        )
        if values.get("caveat"):
            lines.extend([f"- caveat: {values['caveat']}", ""])
    return "\n".join(lines)


def build_all(root: Path, output_dir: Path, dpi: int = DPI) -> dict[str, list[Path]]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font = setup_presentation_style()

    paths_05, validation_05 = build_slide_05(root, output_dir, dpi)
    paths_06, validation_06 = build_slide_06(root, output_dir, dpi)
    paths_07, validation_07 = build_slide_07(root, output_dir, dpi)
    validation = {
        "graph_1": validation_05,
        "graph_2": validation_06,
        "graph_3": validation_07,
    }
    payload = {
        "font": font,
        "palette": COLORS,
        "figsize": FIGSIZE,
        "dpi": dpi,
        "validation": validation,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation.md").write_text(
        _validation_markdown(validation) + "\n", encoding="utf-8"
    )
    return {
        "slide_05": paths_05,
        "slide_06": paths_06,
        "slide_07": paths_07,
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = (args.output_dir or root / "presentation_charts").resolve()
    generated = build_all(root, output_dir, dpi=args.dpi)
    print(f"Presentation charts: {output_dir}")
    for name, paths in generated.items():
        print(f"- {name}: " + " / ".join(path.name for path in paths))
    print("- validation: validation.json / validation.md")


if __name__ == "__main__":
    main()
