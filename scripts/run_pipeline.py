"""End-to-end pipeline: download -> analysis -> figures + reports/metrics.json.

Usage:
    python scripts/run_pipeline.py               # all three parts
    python scripts/run_pipeline.py --skip-shelf  # parts 1-2 only, no big download

The first run downloads ~1.5 GB (streamed, only beer rows are kept) from Open Food
Facts, which takes a while; later runs use the cache in data/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from europe_on_tap import drinking, geo, plots, shelf  # noqa: E402
from europe_on_tap.data import load_borders, load_breweries, load_off_beers, load_who  # noqa: E402

DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
START, END = 1970, 2024
HIGHLIGHT = {
    "United Kingdom": "#1d4ed8", "Ireland": "#15803d", "Denmark": "#b91c1c",
    "Spain": "#ea580c", "Italy": "#7c3aed", "Portugal": "#be185d",
}
TERNARY_LABELS = {
    "United Kingdom", "Ireland", "Italy", "Portugal", "Spain", "France", "Denmark",
    "Russia", "Poland", "Czechia", "Sweden", "Hungary",
}


def r(x, nd=3):
    return round(float(x), nd)


def part1(who, shapes) -> dict:
    print("Part 1: drinking cultures (WHO)")
    plots.beer_share_maps(who, shapes, (START, END), FIG / "map_beer_share.png")
    panel = drinking.balanced_panel(who, 1965, END)
    sigma = drinking.sigma_convergence(panel)
    plots.convergence(who, panel, sigma, HIGHLIGHT, FIG / "convergence.png")
    clusters = drinking.cluster_cultures(who, (START, END))
    plots.ternary_moves(clusters, (START, END), FIG / "ternary.png", TERNARY_LABELS)

    moves = clusters.pivot(index="country", columns="year", values="cluster")
    changed = moves[moves[START] != moves[END]]
    beta = {col: drinking.beta_convergence(who, START, END, col) for col in drinking.SHARES}
    share = who.pivot(index="country", columns="year", values="beer_share")
    return {
        "countries": int(who["iso3"].nunique()),
        "years": [int(who["year"].min()), int(who["year"].max())],
        "sigma_panel_countries": int(panel["iso3"].nunique()),
        "sigma_beer_share": {"1965": r(sigma.loc[1965, "beer_share"]), str(END): r(sigma.loc[END, "beer_share"])},
        "sigma_wine_share": {"1965": r(sigma.loc[1965, "wine_share"]), str(END): r(sigma.loc[END, "wine_share"])},
        "beta_convergence": {
            col: {"slope": r(b["slope"]), "r2": r(b["r2"]), "p_value": float(f"{b['p_value']:.2g}"), "n": b["n"]}
            for col, b in beta.items()
        },
        "beer_share_change": {
            c: [r(share.loc[c, START], 2), r(share.loc[c, END], 2)] for c in HIGHLIGHT
        },
        "dominant_drink_counts": {
            str(y): drinking.dominant_drink(who, y).value_counts().to_dict() for y in (START, END)
        },
        "cluster_switches": {c: [row[START], row[END]] for c, row in changed.iterrows()},
    }


def part2(breweries, shapes) -> dict:
    print("Part 2: brewing heritage (Wikidata)")
    plots.brewery_map(breweries, shapes, FIG / "brewery_map.png")
    plots.founding_years(breweries, FIG / "brewery_foundings.png")
    dated = breweries.dropna(subset=["founded"])
    oldest = dated.nsmallest(10, "founded")
    return {
        "breweries": int(len(breweries)),
        "with_founding_year": int(len(dated)),
        "top_countries": breweries["country"].value_counts().head(5).to_dict(),
        "founded_1850_1899": int(dated["founded"].between(1850, 1899).sum()),
        "founded_2000_on": int((dated["founded"] >= 2000).sum()),
        "oldest": [
            {"name": row["name"], "country": row["country"], "founded": int(row["founded"])}
            for _, row in oldest.iterrows()
        ],
    }


def part3(off) -> dict:
    print("Part 3: on the shelf (Open Food Facts)")
    df = shelf.prepare(off)
    long = shelf.by_country(df)
    summary = shelf.country_summary(long)
    plots.shelf_by_country(long, summary, FIG / "shelf_by_country.png")

    labelled = df[df["style"].notna() & (df["product_name"] != "")].reset_index(drop=True)
    scores, oof = shelf.cross_validate(labelled)
    best = "Name + ABV (TF-IDF + logistic)"
    labels = labelled["style"].value_counts().index.tolist()
    plots.style_model_report(scores, labelled["style"], oof[best], labels, FIG / "style_model.png")
    full = shelf.TextAbvClassifier().fit(labelled[["product_name", "abv"]], labelled["style"])
    return {
        "products": int(len(df)),
        "with_abv": int(df["abv"].notna().sum()),
        "countries_shown": summary.index.tolist(),
        "country_summary": {
            c: {k: r(v) for k, v in row.items()} for c, row in summary.iterrows()
        },
        "style_counts": labelled["style"].value_counts().to_dict(),
        "brands_in_cv": int(labelled["brands"].replace("", None).nunique()),
        "style_model_scores": {m: {k: r(v) for k, v in s.items()} for m, s in scores.items()},
        "style_top_words": shelf.top_terms(full),
    }


def main() -> None:
    DATA.mkdir(exist_ok=True)
    who = load_who(DATA)
    shapes = geo.country_shapes(load_borders(DATA))
    metrics = {
        "drinking_cultures": part1(who, shapes),
        "breweries": part2(load_breweries(DATA), shapes),
    }
    out = ROOT / "reports" / "metrics.json"
    if "--skip-shelf" not in sys.argv:
        metrics["shelf"] = part3(load_off_beers(DATA))
    elif out.exists():  # keep the shelf results from the last full run
        previous = json.loads(out.read_text(encoding="utf-8"))
        if "shelf" in previous:
            metrics["shelf"] = previous["shelf"]
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
