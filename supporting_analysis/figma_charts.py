"""Build the three presentation charts prepared for the Figma Slides deck.

The exports use the deck palette and are written both as PNG and SVG.  SVG is
the preferred format for Figma because labels and shapes stay crisp at any
scale; PNG is useful for a quick preview.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline.plotting import plt
from matplotlib import font_manager, ticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Colors sampled from the Figma Slides template.
INK = "#1E0833"
SLIDE = "#BF8ED5"
PANEL = "#D1ABE3"
ACCENT = "#AB5998"
GRID = "#B58AC8"
WHITE = "#FFFFFF"
CRISIS = "#7A315F"
CHART_BG = "#E7D6EE"

GROUP_COLORS = [
    "#5B2A72",
    "#704080",
    "#87508D",
    "#9D5793",
    "#AB5998",
    "#7D639E",
    "#493062",
]

CATEGORY_COLORS = [
    "#351447",
    "#653277",
    "#8A477F",
    "#AB5998",
    "#C47BB1",
    "#EEE0F4",
]

SHORT_GROUPS = {
    0: "Без недавних\nнарушений",
    1: "Без новых штрафов,\nс историей",
    2: "Эпизодические",
    3: "Любители\nскорости",
    4: "Хронические",
    5: "Разноплановые",
    6: "Опасные",
}

SHORT_OFFENCES = {
    "Превышение скорости на 20-40 км/ч": "Скорость +20–40",
    "Не пристегнут ремень безопасности": "Ремень",
    "Нарушение разметки": "Разметка",
    "Превышение скорости на 40-60 км/ч": "Скорость +40–60",
    "Использование телефона за рулем": "Телефон",
}

MONTH_NAMES = {
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Корень проекта с data/ и outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка экспорта; по умолчанию outputs/figma_charts.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _configure_style() -> str:
    installed = {item.name for item in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in installed else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.titlepad": 15,
            "svg.fonttype": "none",
        }
    )
    return family


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"В {source} отсутствуют столбцы: {', '.join(missing)}")


def _read_csv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Не найден источник {path}. Сначала запустите: uv run python main.py"
        )
    frame = pd.read_csv(path)
    _require_columns(frame, columns, path)
    return frame


def _new_figure(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    return fig, ax


def _clean_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(INK)
    ax.spines[["bottom", "left"]].set_alpha(0.35)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)


def _save(
    fig, output_dir: Path, stem: str, dpi: int, *, transparent: bool = False
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    fig.savefig(
        paths[0],
        dpi=dpi,
        bbox_inches="tight",
        facecolor="none" if transparent else fig.get_facecolor(),
        transparent=transparent,
    )
    fig.savefig(
        paths[1],
        bbox_inches="tight",
        facecolor="none" if transparent else fig.get_facecolor(),
        transparent=transparent,
    )
    plt.close(fig)
    return paths


def _fuel_monthly(source: Path) -> pd.DataFrame:
    columns = {
        "order_datetime",
        "physical_fuel_transaction_main",
        "physical_fuel_volume_main",
        "fuel_cost_rub",
    }
    data = _read_csv(source, columns).copy()
    data["order_datetime"] = pd.to_datetime(data["order_datetime"], errors="coerce")
    start = pd.Timestamp("2026-04-01")
    end = pd.Timestamp("2026-09-01")
    data = data.loc[data["order_datetime"].ge(start) & data["order_datetime"].lt(end)]
    data["sale_month"] = data["order_datetime"].dt.to_period("M").dt.start_time
    monthly = (
        data.groupby("sale_month", as_index=False)
        .agg(
            sale_count=("physical_fuel_transaction_main", "sum"),
            sold_volume_liters=("physical_fuel_volume_main", "sum"),
            sales_revenue_rub=("fuel_cost_rub", "sum"),
        )
        .sort_values("sale_month")
    )
    monthly["weighted_price_rub_per_liter"] = monthly["sales_revenue_rub"].div(
        monthly["sold_volume_liters"].where(monthly["sold_volume_liters"].gt(0))
    )
    monthly["month_name"] = monthly["sale_month"].dt.month.map(MONTH_NAMES)
    return monthly


def build_price_and_volume(source: Path, output_dir: Path, dpi: int) -> list[Path]:
    """Chart 1: fuel volume bars plus the observed average-price line."""
    columns = {"month", "n_drivers", "total_liters", "avg_fuel_price"}
    data = _read_csv(source, columns).copy()
    data["month"] = pd.to_datetime(data["month"], errors="coerce")
    data = data.loc[
        data["month"].ge("2026-04-01") & data["month"].lt("2026-09-01")
    ].sort_values("month")
    if len(data) != 5:
        raise ValueError(f"В {source} ожидаются ровно апрель–август 2026")
    if data["n_drivers"].nunique() != 1:
        raise ValueError(f"В {source} должен быть один и тот же состав когорты")

    data["volume_mln_liters"] = data["total_liters"] / 1_000_000
    data["month_name"] = data["month"].dt.month.map(MONTH_NAMES)

    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    x = np.arange(len(data), dtype=float)

    # The post-crisis interval starts on 1 June, between May and June.
    ax.axvspan(1.5, 4.5, color="#F7DFF0", alpha=0.62, zorder=0)
    for level in np.linspace(0, 3.2, 4):
        ax.axhline(level, color=GRID, linewidth=1.0, alpha=0.34, zorder=1)

    bars = ax.bar(
        x,
        data["volume_mln_liters"],
        width=0.56,
        color=ACCENT,
        alpha=0.90,
        edgecolor="none",
        zorder=2,
    )
    # Rounded corners are applied to the rectangle path by a thick round join.
    for bar in bars:
        bar.set_joinstyle("round")

    for xi, value in zip(x, data["volume_mln_liters"], strict=True):
        ax.text(
            xi,
            value + 0.07,
            f"{value:.2f}".replace(".", ","),
            ha="center",
            va="bottom",
            color=INK,
            fontsize=15,
            fontweight="bold",
            zorder=6,
        )

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    prices = data["avg_fuel_price"].to_numpy(dtype=float)
    ax2.set_ylim(67.7, 72.25)
    ax2.plot(
        x,
        prices,
        color="#4D197A",
        linewidth=4.0,
        marker="o",
        markersize=11,
        markerfacecolor="#4D197A",
        markeredgewidth=0,
        zorder=5,
    )
    price_offsets = [(0, 16), (0, 17), (0, 17), (0, 17), (0, 18)]
    for xi, price, offset in zip(x, prices, price_offsets, strict=True):
        ax2.annotate(
            f"{price:.2f} ₽".replace(".", ","),
            (xi, price),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#4D197A",
            fontsize=13,
            fontweight="bold",
            zorder=7,
        )

    ax.axvline(1.5, color=ACCENT, linewidth=1.4, alpha=0.9, zorder=3)
    ax.text(
        -0.48,
        3.53,
        "ОБЪЁМ · млн литров",
        color=ACCENT,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
    )
    ax.text(
        0.95,
        3.53,
        "ЦЕНА · ₽/литр",
        color="#4D197A",
        fontsize=12.5,
        fontweight="bold",
        ha="left",
    )

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 3.72)
    ax.set_xticks(x, [name[:3] for name in data["month_name"]])
    ax.tick_params(axis="x", length=0, pad=14, labelsize=14, colors="#6B5A7C")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax2.tick_params(axis="y", right=False, labelright=False)
    for axis in (ax, ax2):
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.13)
    return _save(fig, output_dir, "01_price_and_fuel_volume", dpi)


def build_behavior_groups(source: Path, output_dir: Path, dpi: int) -> list[Path]:
    """Chart 2: a schematic, fact-checked map of the seven rule-based groups."""
    columns = {
        "cluster_id",
        "cluster_label",
        "clients",
        "share_pct",
        "fines_pre_mean",
        "fuel_liters_per_week_pre_median",
    }
    data = _read_csv(source, columns).sort_values("cluster_id").copy()
    if data["cluster_id"].duplicated().any():
        raise ValueError(f"В {source} cluster_id должен быть уникальным")

    if data["clients"].sum() != 25_675:
        raise ValueError("Сумма размеров семи групп должна быть равна 25 675")

    # Position encodes the group rules, not a fitted embedding. The horizontal
    # axis increases with recent offence frequency; the vertical axis increases
    # with offence diversity / danger. Groups 0 and 1 both have no recent fines,
    # so they remain at the left edge and are separated slightly for readability.
    positions = {
        0: (0.10, 0.22),
        1: (0.27, 0.29),
        2: (0.45, 0.39),
        3: (0.62, 0.55),
        4: (0.79, 0.46),
        5: (0.69, 0.78),
        6: (0.91, 0.87),
    }
    short_names = {
        0: "Без недавних",
        1: "С историей",
        2: "Эпизодические",
        3: "Скорость",
        4: "Хронические",
        5: "Разноплановые",
        6: "Опасные",
    }
    point_colors = {
        0: "#F2C6E8",
        1: "#C08BD5",
        2: "#AB5998",
        3: "#9E4F8D",
        4: "#AC599A",
        5: "#BD8AD1",
        6: "#F0C2E2",
    }

    def draw(show_points: bool):
        fig, ax = plt.subplots(figsize=(13.8, 6.4))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for value in np.linspace(0.12, 0.92, 5):
            ax.axvline(value, color=GRID, linewidth=1.0, alpha=0.58, zorder=0)
        for value in np.linspace(0.16, 0.88, 4):
            ax.axhline(value, color=GRID, linewidth=1.0, alpha=0.58, zorder=0)

        if show_points:
            for row in data.itertuples(index=False):
                group = int(row.cluster_id)
                px, py = positions[group]
                # Area is approximately proportional to group share, with a
                # floor so the two smallest groups remain legible on a slide.
                marker_area = 750 + 2800 * np.sqrt(float(row.share_pct) / 40.0)
                ax.scatter(
                    [px],
                    [py],
                    s=marker_area,
                    c=[point_colors[group]],
                    edgecolors="none",
                    zorder=3,
                )
                ax.text(
                    px,
                    py,
                    str(group),
                    ha="center",
                    va="center",
                    color=INK,
                    fontsize=22,
                    fontweight="bold",
                    zorder=4,
                )

        for row in data.itertuples(index=False):
            group = int(row.cluster_id)
            px, py = positions[group]
            label_y = py - (0.105 if group in {0, 1, 2, 3, 4} else 0.10)
            ax.text(
                px,
                label_y,
                (
                    f"{short_names[group]}\n{float(row.share_pct):.1f}%"
                    if show_points
                    else f"{group}. {short_names[group]}\n{float(row.share_pct):.1f}%"
                ).replace(".", ","),
                fontsize=10.6,
                ha="center",
                va="top",
                color=INK,
                fontweight="bold",
                linespacing=1.22,
                zorder=5,
            )

        ax.annotate(
            "частота недавних нарушений →",
            xy=(0.99, 0.02),
            ha="right",
            va="bottom",
            color="#4D197A",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0.015,
            0.985,
            "разнообразие / опасность ↑",
            ha="left",
            va="top",
            color="#4D197A",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0.015,
            0.02,
            "схематично · группы определены по докризисным правилам",
            ha="left",
            va="bottom",
            color="#6B5A7C",
            fontsize=8.7,
            alpha=0.88,
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(left=0.025, right=0.985, top=0.98, bottom=0.04)
        return fig

    return _save(
        draw(show_points=True),
        output_dir,
        "02_precrisis_behavior_groups_with_points",
        dpi,
    )


def _composition_table(data: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, list[str]]:
    value_columns = ["fine_count_pre_crisis_2026", "fine_count_post_crisis_2026"]
    totals = data.groupby("offence_short_statement")[value_columns].sum().sum(axis=1)
    top = totals.nlargest(top_n).index.tolist()
    rows: list[dict[str, float | int | str]] = []
    for cluster, block in data.groupby("cluster_id", sort=True):
        for period, column in [
            ("до", "fine_count_pre_crisis_2026"),
            ("после", "fine_count_post_crisis_2026"),
        ]:
            values = block.set_index("offence_short_statement")[column]
            total = float(values.sum())
            row: dict[str, float | int | str] = {
                "cluster_id": int(cluster),
                "period": period,
            }
            for offence in top:
                row[offence] = 100 * float(values.get(offence, 0)) / total if total else 0
            selected = sum(float(values.get(offence, 0)) for offence in top)
            row["Остальные"] = 100 * (total - selected) / total if total else 0
            rows.append(row)
    return pd.DataFrame(rows), top + ["Остальные"]


def build_offence_structure(source: Path, output_dir: Path, dpi: int) -> list[Path]:
    """Chart 3: 100% composition of offence types before and after the crisis."""
    columns = {
        "cluster_id",
        "cluster_label",
        "offence_short_statement",
        "fine_count_pre_crisis_2026",
        "fine_count_post_crisis_2026",
    }
    raw = _read_csv(source, columns).copy()
    # Groups 0 and 1 are defined by having no fines in the pre-crisis window,
    # so a before/after composition comparison would be structurally invalid.
    raw = raw.loc[raw["cluster_id"].ge(2)]
    composition, categories = _composition_table(raw)

    fig, ax = _new_figure((12.8, 4.65))
    groups = sorted(composition["cluster_id"].unique())
    x = np.arange(len(groups), dtype=float)
    width = 0.34
    bottoms = {"до": np.zeros(len(groups)), "после": np.zeros(len(groups))}

    for category, color in zip(categories, CATEGORY_COLORS, strict=True):
        for period, offset, alpha in [("до", -0.19, 0.68), ("после", 0.19, 1.0)]:
            part = composition.loc[composition["period"].eq(period)].set_index("cluster_id")
            values = part.reindex(groups)[category].fillna(0).to_numpy(dtype=float)
            ax.bar(
                x + offset,
                values,
                width,
                bottom=bottoms[period],
                color=color,
                alpha=alpha,
                edgecolor=PANEL,
                linewidth=0.35,
                label=SHORT_OFFENCES.get(category, category)
                if period == "после"
                else None,
            )
            bottoms[period] += values

    for center in x:
        ax.text(center - 0.19, 102.2, "до", ha="center", va="bottom", fontsize=9)
        ax.text(center + 0.19, 102.2, "после", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, [SHORT_GROUPS.get(group, str(group)) for group in groups])
    ax.set_ylim(0, 109)
    ax.set_ylabel("Доля типа нарушения")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.4,
        handlelength=1.6,
    )
    fig.text(
        0.985,
        0.025,
        "Группы 0–1 не показаны: их предкризисный ноль задан правилом сегментации",
        ha="right",
        va="bottom",
        fontsize=8.8,
        color=INK,
        alpha=0.78,
    )
    ax.margins(x=0.045)
    _clean_axes(ax)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.20)
    return _save(fig, output_dir, "03_offence_structure_pre_post", dpi)


def build_all(project_root: Path, output_dir: Path, dpi: int = 220) -> dict[str, list[Path]]:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    font = _configure_style()
    generated = {
        "price_and_volume": build_price_and_volume(
            project_root
            / "archive_fines_per_liter_analysis/outputs/tables/national_monthly.csv",
            output_dir,
            dpi,
        ),
        "behavior_groups": build_behavior_groups(
            project_root
            / "outputs/behavior_clustering/tables/behavior_cluster_summary.csv",
            output_dir,
            dpi,
        ),
        "offence_structure": build_offence_structure(
            project_root
            / "outputs/behavior_cluster_analysis/cluster_offence_types_pre_post_2026.csv",
            output_dir,
            dpi,
        ),
    }
    manifest = {
        "font": font,
        "palette": {
            "ink": INK,
            "slide": SLIDE,
            "panel": PANEL,
            "accent": ACCENT,
        },
        "charts": {
            name: [str(path.relative_to(project_root)) for path in paths]
            for name, paths in generated.items()
        },
        "figma_note": textwrap.dedent(
            """
            Use SVG for the deck. PNG files are previews. Groups 0 and 1 are
            intentionally excluded from the offence-composition comparison
            because their zero pre-crisis count is part of the group rule.
            """
        ).strip(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return generated


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "outputs/figma_charts").resolve()
    generated = build_all(root, output, dpi=args.dpi)
    print(f"Графики для Figma созданы в {output}")
    for paths in generated.values():
        for index in range(0, len(paths), 2):
            pair = paths[index : index + 2]
            print("- " + " / ".join(path.name for path in pair))


if __name__ == "__main__":
    main()
