"""Part 1: how Europe's beverage mix changed, 1961-2024 (WHO data).

Shares are of beer + wine + spirits (the small "other" category is left out), so
they describe a country's drinking *culture* independently of how much it drinks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

SHARES = ["beer_share", "wine_share", "spirits_share"]


def balanced_panel(who: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Countries observed in every year from start to end, so the mix of countries
    does not change over time (otherwise newly reporting countries fake a trend)."""
    window = who[who["year"].between(start, end)]
    years = end - start + 1
    keep = window.groupby("iso3")["year"].nunique()
    return window[window["iso3"].isin(keep[keep == years].index)]


def sigma_convergence(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-country standard deviation of each share, per year (falling = converging)."""
    return panel.groupby("year")[SHARES].std()


def beta_convergence(who: pd.DataFrame, start: int, end: int, col: str = "beer_share") -> dict:
    """Regress the change in a share on its starting level.

    A negative slope means the countries that started low caught up, and the
    countries that started high fell back: beta-convergence.
    """
    a = who[who["year"] == start].set_index("iso3")[col]
    b = who[who["year"] == end].set_index("iso3")[col]
    both = pd.concat([a.rename("start"), b.rename("end")], axis=1).dropna()
    fit = stats.linregress(both["start"], both["end"] - both["start"])
    return {
        "data": both,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r2": fit.rvalue**2,
        "p_value": fit.pvalue,
        "n": len(both),
    }


def dominant_drink(who: pd.DataFrame, year: int) -> pd.Series:
    row = who[who["year"] == year].set_index("country")[SHARES]
    return row.idxmax(axis=1).str.replace("_share", "")


def cluster_cultures(who: pd.DataFrame, years: tuple[int, int], k: int = 3, seed: int = 0) -> pd.DataFrame:
    """K-means on beverage shares, fitted on both years pooled so clusters are comparable.

    Each cluster is named after the drink with the highest share in its centre.
    """
    panel = who[who["year"].isin(years)].dropna(subset=SHARES)
    counts = panel.groupby("iso3")["year"].nunique()
    panel = panel[panel["iso3"].isin(counts[counts == 2].index)].copy()

    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(panel[SHARES])
    centres = pd.DataFrame(km.cluster_centers_, columns=SHARES)
    names = {}
    for i, c in centres.iterrows():
        name = f"{c.idxmax().replace('_share', '').capitalize()} culture"
        names[i] = name if name not in names.values() else f"{name} ({i})"
    panel["cluster"] = [names[c] for c in km.labels_]
    return panel[["iso3", "country", "year", *SHARES, "cluster"]]


def ternary_xy(beer, wine, spirits):
    """Project shares onto an equilateral triangle: beer bottom-left,
    wine bottom-right, spirits on top."""
    beer, wine, spirits = map(np.asarray, (beer, wine, spirits))
    total = beer + wine + spirits
    x = (wine + spirits / 2) / total
    y = (np.sqrt(3) / 2) * spirits / total
    return x, y
