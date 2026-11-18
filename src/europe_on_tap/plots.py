"""Figures for the README. Every function writes one PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from . import geo  # noqa: E402
from .drinking import ternary_xy  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)
INK = "#1f2937"
GREY = "#9ca3af"
BEER = "#d99a1e"
WINE = "#8c1c3a"
SPIRITS = "#4c78a8"
DRINK_COLORS = {"beer": BEER, "wine": WINE, "spirits": SPIRITS}
BEER_CMAP = LinearSegmentedColormap.from_list("beer", ["#fdf6e3", "#f6d27a", "#d99a1e", "#8a4b08", "#3d1f02"])


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Part 1: drinking cultures ------------------------------------------------

def beer_share_maps(who: pd.DataFrame, shapes, years: tuple[int, int], path: Path) -> Path:
    norm = Normalize(0, 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), gridspec_kw={"wspace": 0.02})
    for ax, year in zip(axes, years):
        values = who[who["year"] == year].set_index("iso3")["beer_share"].to_dict()
        geo.draw_countries(ax, shapes, values, BEER_CMAP, norm)
        geo.set_europe_view(ax)
        ax.set_title(str(year), fontsize=15, fontweight="bold", color=INK)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=BEER_CMAP)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
    cb.set_label("Beer's share of alcohol consumed (beer + wine + spirits)")
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    cb.ax.set_title("Grey: no WHO data for that year. Source: WHO Global Health Observatory.",
                    color=GREY, fontsize=8, y=-4.2)
    return _save(fig, path)


def convergence(who: pd.DataFrame, panel: pd.DataFrame, sigma: pd.DataFrame, highlight: dict, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    for _, g in panel.groupby("iso3"):
        ax1.plot(g["year"], g["beer_share"], color="#e5e7eb", lw=1, zorder=1)
    ends = {}
    for country, color in highlight.items():
        g = who[who["country"] == country]
        ax1.plot(g["year"], g["beer_share"], color=color, lw=2.2, zorder=2)
        ends[country] = (g["year"].iloc[-1], g["beer_share"].iloc[-1])
    # Nudge end labels apart so lines that finish close together stay readable.
    placed = []
    for country, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        label_y = max(y, placed[-1] + 0.03) if placed else y
        placed.append(label_y)
        ax1.annotate(country, (x, y), xytext=(x + 1.5, label_y), color=highlight[country],
                     fontsize=9, va="center")
    ax1.set_ylabel("Beer's share of alcohol")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_xlim(panel["year"].min(), panel["year"].max() + 9)
    ax1.set_title("Beer's share fell in beer countries and rose in wine countries")

    for col, color in DRINK_COLORS.items():
        ax2.plot(sigma.index, sigma[f"{col}_share"], color=color, lw=2.2, label=col.capitalize())
    ax2.set_ylabel("Std. dev. of share across countries")
    ax2.set_ylim(0, None)
    n = panel["iso3"].nunique()
    ax2.set_title(f"…so countries look ever more alike\n(σ-convergence, same {n} countries every year)")
    ax2.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def ternary_moves(clusters: pd.DataFrame, years: tuple[int, int], path: Path, label: set[str]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 8.2))
    h = np.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color=INK, lw=1)
    for t in (0.2, 0.4, 0.6, 0.8):  # light gridlines: constant share of each drink
        ax.plot([t / 2, 1 - t / 2], [t * h, t * h], color="#e5e7eb", lw=0.8, zorder=0)
        ax.plot([t, t / 2], [0, t * h], color="#e5e7eb", lw=0.8, zorder=0)
        ax.plot([t, (1 + t) / 2], [0, (1 - t) * h], color="#e5e7eb", lw=0.8, zorder=0)
    ax.text(-0.03, -0.04, "100% beer", ha="center", color=BEER, fontweight="bold")
    ax.text(1.03, -0.04, "100% wine", ha="center", color=WINE, fontweight="bold")
    ax.text(0.5, h + 0.03, "100% spirits", ha="center", color=SPIRITS, fontweight="bold")

    culture_colors = {"Beer culture": BEER, "Wine culture": WINE, "Spirits culture": SPIRITS}
    start, end = (clusters[clusters["year"] == y].set_index("country") for y in years)
    centre = ternary_xy(1, 1, 1)
    inward = 0
    for country in start.index:
        x0, y0 = ternary_xy(*start.loc[country, ["beer_share", "wine_share", "spirits_share"]])
        x1, y1 = ternary_xy(*end.loc[country, ["beer_share", "wine_share", "spirits_share"]])
        inward += np.hypot(x1 - centre[0], y1 - centre[1]) < np.hypot(x0 - centre[0], y0 - centre[1])
        color = culture_colors.get(start.loc[country, "cluster"], GREY)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.3, "alpha": 0.85, "mutation_scale": 11})
        ax.plot(x0, y0, "o", color=color, ms=4, alpha=0.6)
        if country in label:
            ax.annotate(country, (x1, y1), xytext=(5, 2), textcoords="offset points", fontsize=8, color=INK)
    ax.plot(*centre, "+", color=INK, ms=12, mew=1.5, zorder=5)
    ax.annotate("equal mix", centre, xytext=(8, 6), textcoords="offset points", fontsize=8, color=GREY,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85}, zorder=5)
    for name, color in culture_colors.items():
        ax.plot([], [], color=color, lw=2, label=f"{name} in {years[0]}")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Every country's drink mix, {years[0]} (dot) → {years[1]} (arrow tip)\n"
                 f"{inward} of {len(start)} countries moved closer to an equal mix")
    ax.set_aspect("equal")
    ax.axis("off")
    return _save(fig, path)


# --- Part 2: brewing heritage --------------------------------------------------

ERAS = [
    (None, 1500, "Before 1500", "#7f1d1d"),
    (1500, 1800, "1500–1799", "#c2410c"),
    (1800, 1950, "1800–1949", "#f59e0b"),
    (1950, 2000, "1950–1999", "#a3e635"),
    (2000, None, "2000 onwards", "#22d3ee"),
]


def brewery_map(breweries: pd.DataFrame, shapes, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    geo.draw_countries(ax, shapes, base_color="#1f2937", edge="#374151", linewidth=0.5)
    undated = breweries[breweries["founded"].isna()]
    x, y = geo.laea(undated["lon"], undated["lat"])
    ax.scatter(x, y, s=5, color="#6b7280", alpha=0.6, lw=0, label=f"Founding year unknown ({len(undated)})")
    for lo, hi, label, color in ERAS:
        sel = breweries["founded"].notna()
        if lo is not None:
            sel &= breweries["founded"] >= lo
        if hi is not None:
            sel &= breweries["founded"] < hi
        sub = breweries[sel]
        x, y = geo.laea(sub["lon"], sub["lat"])
        ax.scatter(x, y, s=14, color=color, alpha=0.9, lw=0, label=f"{label} ({len(sub)})", zorder=3)
    geo.set_europe_view(ax)
    leg = ax.legend(loc="upper left", frameon=False, title="Founded", fontsize=9, markerscale=1.8)
    plt.setp(leg.get_texts(), color="white")
    leg.get_title().set_color("white")
    ax.set_title(f"{len(breweries):,} European breweries on Wikidata", color="white", fontsize=14)
    return _save(fig, path)


def founding_years(breweries: pd.DataFrame, path: Path) -> Path:
    years = breweries["founded"].dropna()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bins = np.arange(1000, 2031, 10)
    counts, edges = np.histogram(years, bins)
    colors = []
    for left in edges[:-1]:
        colors.append(next(c for lo, hi, _, c in ERAS if (lo is None or left >= lo) and (hi is None or left < hi)))
    ax.bar(edges[:-1], counts, width=10, align="edge", color=colors, edgecolor="white", lw=0.3)
    ax.set_xlim(1000, 2030)
    ax.set_ylabel("Breweries founded per decade")
    ax.set_title(f"A thousand years of brewing ({len(years)} breweries with a known founding year)")
    oldest = breweries.dropna(subset=["founded"]).nsmallest(1, "founded").iloc[0]
    ax.annotate(f"{oldest['name']}\n{int(oldest['founded'])}", (oldest["founded"] + 5, 0.5),
                xytext=(oldest["founded"] + 40, counts.max() * 0.45), fontsize=8,
                arrowprops={"arrowstyle": "-", "color": GREY})
    return _save(fig, path)


# --- Part 3: on the shelf -------------------------------------------------------

def shelf_by_country(long: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    order = summary.sort_values("median_abv").index.tolist()
    alcoholic = long[~long["non_alcoholic"] & long["abv"].notna()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.42 * len(order) + 1.6), sharey=True,
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    data = [alcoholic.loc[alcoholic["country"] == c, "abv"].to_numpy() for c in order]
    bp = ax1.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                     medianprops={"color": INK, "lw": 1.5})
    for box in bp["boxes"]:
        box.set(facecolor="#f6d27a", edgecolor=BEER)
    ax1.set_yticks(range(1, len(order) + 1), [f"{c}  (n={summary.loc[c, 'products']:,})" for c in order])
    ax1.axvline(5, color=GREY, ls=":", lw=1)
    ax1.set_xlabel("ABV % (alcoholic beers; box = middle 50%)")
    ax1.set_title("How strong is the beer on the shelf?")

    ax2.barh(range(1, len(order) + 1), summary.loc[order, "share_non_alcoholic"], color=SPIRITS, height=0.6)
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Share of beers that are alcohol-free")
    ax2.set_title("Alcohol-free share")
    fig.text(0.01, -0.01, "Country = where the product is sold (Open Food Facts, community data).",
             color=GREY, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def style_model_report(scores: dict, y_true, y_pred, labels: list[str], path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.35]})
    names = list(scores)
    f1 = [scores[n]["macro_f1"] for n in names]
    colors = [GREY] * (len(names) - 1) + [BEER]
    ax1.barh(names, f1, color=colors)
    for i, v in enumerate(f1):
        ax1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    ax1.set_xlabel("Macro-F1 (5-fold CV, brands never shared between train and test)")
    ax1.set_title("Predicting beer style from the label")

    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred")).reindex(
        index=labels, columns=labels, fill_value=0)
    norm = cm.div(cm.sum(axis=1), axis=0)
    ax2.imshow(norm.to_numpy(), cmap=BEER_CMAP, vmin=0, vmax=1)
    ax2.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax2.set_yticks(range(len(labels)), [f"{lab} ({cm.loc[lab].sum()})" for lab in labels])
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = norm.iat[i, j]
            if v >= 0.05:
                ax2.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                         color="white" if v > 0.5 else INK)
    ax2.grid(False)
    ax2.set_xlabel("Predicted style")
    ax2.set_title("Where the best model gets confused (row = true style)")
    fig.tight_layout()
    return _save(fig, path)
