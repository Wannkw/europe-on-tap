"""Download and cache the four public data sources.

* WHO Global Health Observatory: recorded alcohol per capita by beverage type.
* Wikidata: breweries in Europe with coordinates and founding year.
* Open Food Facts: beer products with ABV, country and style categories.
* Natural Earth: country polygons for the maps.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = "europe-on-tap/0.1 (data science portfolio project)"

WHO_URL = "https://ghoapi.azureedge.net/api/SA_0000001400"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
OFF_EXPORT_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)

# Geographic Europe within the WHO European Region (drops the Caucasus,
# Central Asia, Israel and Turkey).
EUROPE = {
    "ALB": "Albania", "AND": "Andorra", "AUT": "Austria", "BEL": "Belgium",
    "BGR": "Bulgaria", "BIH": "Bosnia and Herzegovina", "BLR": "Belarus",
    "CHE": "Switzerland", "CYP": "Cyprus", "CZE": "Czechia", "DEU": "Germany",
    "DNK": "Denmark", "ESP": "Spain", "EST": "Estonia", "FIN": "Finland",
    "FRA": "France", "GBR": "United Kingdom", "GRC": "Greece", "HRV": "Croatia",
    "HUN": "Hungary", "IRL": "Ireland", "ISL": "Iceland", "ITA": "Italy",
    "LTU": "Lithuania", "LUX": "Luxembourg", "LVA": "Latvia", "MDA": "Moldova",
    "MKD": "North Macedonia", "MLT": "Malta", "MNE": "Montenegro",
    "NLD": "Netherlands", "NOR": "Norway", "POL": "Poland", "PRT": "Portugal",
    "ROU": "Romania", "RUS": "Russia", "SRB": "Serbia", "SVK": "Slovakia",
    "SVN": "Slovenia", "SWE": "Sweden", "UKR": "Ukraine",
}
BEVERAGES = {
    "ALCOHOLTYPE_SA_BEER": "beer",
    "ALCOHOLTYPE_SA_WINE": "wine",
    "ALCOHOLTYPE_SA_SPIRITS": "spirits",
    "ALCOHOLTYPE_SA_OTHER_ALCOHOL": "other",
    "ALCOHOLTYPE_SA_TOTAL": "total",
}


def _get(url: str, retries: int = 6, **kwargs) -> requests.Response:
    """GET with exponential backoff; 429 and 5xx (overloaded server) are retried."""
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=120, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(15 * 2**attempt)
    raise RuntimeError("unreachable")


# --- WHO ---------------------------------------------------------------------

def load_who(cache_dir: Path) -> pd.DataFrame:
    """Litres of pure alcohol per adult (15+) per year: one row per country-year."""
    path = Path(cache_dir) / "who_alcohol_by_beverage.json"
    if not path.exists():
        resp = _get(WHO_URL, params={"$filter": "ParentLocationCode eq 'EUR'"})
        path.write_text(resp.text, encoding="utf-8")
    raw = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["value"])
    raw = raw[raw["SpatialDim"].isin(EUROPE) & raw["Dim1"].isin(BEVERAGES)]
    wide = (
        raw.assign(beverage=raw["Dim1"].map(BEVERAGES))
        .pivot_table(index=["SpatialDim", "TimeDim"], columns="beverage", values="NumericValue")
        .reset_index()
        .rename(columns={"SpatialDim": "iso3", "TimeDim": "year"})
    )
    wide.columns.name = None
    wide["country"] = wide["iso3"].map(EUROPE)
    wide = wide.dropna(subset=["beer", "wine", "spirits"])
    parts = wide[["beer", "wine", "spirits"]].sum(axis=1)
    for col in ("beer", "wine", "spirits"):
        wide[f"{col}_share"] = wide[col] / parts
    return wide[parts > 0].sort_values(["iso3", "year"]).reset_index(drop=True)


# --- Wikidata ----------------------------------------------------------------

BREWERY_QUERY = """
SELECT ?b ?bLabel ?countryLabel ?inception ?coord WHERE {
  ?b wdt:P31/wdt:P279* wd:Q131734;   # instance of (a subclass of) brewery
     wdt:P17 ?country;
     wdt:P625 ?coord.
  ?country wdt:P30 wd:Q46.           # country is on the continent of Europe
  OPTIONAL { ?b wdt:P571 ?inception }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def load_breweries(cache_dir: Path) -> pd.DataFrame:
    path = Path(cache_dir) / "wikidata_breweries.csv"
    if not path.exists():
        resp = _get(WIKIDATA_URL, params={"query": BREWERY_QUERY}, headers={"Accept": "text/csv"})
        path.write_text(resp.text, encoding="utf-8")
    df = pd.read_csv(path)
    coords = df["coord"].str.extract(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)").astype(float)
    df["lon"], df["lat"] = coords[0], coords[1]
    # Inception is an ISO timestamp; a year before 1000 comes zero-padded, e.g. "0950-..".
    df["founded"] = pd.to_numeric(df["inception"].str.extract(r"^(-?\d{1,4})-")[0], errors="coerce")
    # A few breweries also list historical states (East Germany, Austrian Empire);
    # prefer the most common label, which is the present-day country.
    df["country_n"] = df["countryLabel"].map(df["countryLabel"].value_counts())
    df = (
        # several inception values: keep the earliest; then modern country, then name
        df.sort_values(["founded", "country_n", "bLabel"], ascending=[True, False, True])
        .drop_duplicates("b")
        .rename(columns={"bLabel": "name", "countryLabel": "country"})
    )
    df = df[df["lon"].between(-25, 45) & df["lat"].between(34, 72)]
    return df[["name", "country", "founded", "lat", "lon"]].reset_index(drop=True)


# --- Open Food Facts ---------------------------------------------------------

OFF_COLUMNS = ["code", "product_name", "brands", "categories_tags", "countries_tags", "alcohol_100g"]


def _download_resumable(url: str, dest: Path, max_failures: int = 50) -> None:
    """Download a large file, resuming with HTTP Range requests after a dropped connection."""
    part = dest.with_suffix(dest.suffix + ".part")
    failures = 0
    while True:
        done = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={done}-"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code == 416:  # nothing left to fetch
                    break
                resp.raise_for_status()
                if done and resp.status_code != 206:
                    raise RuntimeError("server ignored the Range header; cannot resume")
                total = done + int(resp.headers.get("Content-Length", 0))
                with open(part, "ab") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if part.stat().st_size >= total:
                break
        except requests.RequestException as err:
            failures += 1
            if failures > max_failures:
                raise
            print(f"  connection dropped at {part.stat().st_size / 1e6:,.0f} MB ({err.__class__.__name__}); resuming")
            time.sleep(10)
    part.replace(dest)


def _stream_off_beers(dest: Path) -> None:
    """Extract beers from the full Open Food Facts export (~1.5 GB gzip).

    The search API is rate limited to a few requests a minute, so the bulk export
    is the route Open Food Facts recommends for anything beyond a few pages.
    """
    export = dest.parent / "off_export.csv.gz"
    if not export.exists():
        _download_resumable(OFF_EXPORT_URL, export)
    tmp = dest.with_suffix(".partial")
    with gzip.open(export, "rt", encoding="utf-8", errors="replace") as lines, \
            open(tmp, "w", encoding="utf-8", newline="") as out:
        header = next(lines).rstrip("\n").split("\t")
        idx = [header.index(c) for c in OFF_COLUMNS]
        out.write("\t".join(OFF_COLUMNS) + "\n")
        for line in lines:
            if "en:beers" not in line:  # cheap pre-filter before splitting
                continue
            cells = line.rstrip("\n").split("\t")
            if len(cells) != len(header) or "en:beers" not in cells[idx[3]].split(","):
                continue
            out.write("\t".join(cells[i] for i in idx) + "\n")
    tmp.replace(dest)


def load_off_beers(cache_dir: Path) -> pd.DataFrame:
    """All products in the Open Food Facts 'beers' category."""
    path = Path(cache_dir) / "off_beers.tsv"
    if not path.exists():
        _stream_off_beers(path)
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
    df = df.drop_duplicates("code")
    df["abv"] = pd.to_numeric(df["alcohol_100g"], errors="coerce")
    df["product_name"] = df["product_name"].str.strip()
    df["brands"] = df["brands"].str.split(",").str[0].str.strip().str.lower()
    for col in ("countries_tags", "categories_tags"):
        df[col] = df[col].apply(lambda v: [t for t in v.split(",") if t])
    return df.drop(columns="alcohol_100g").reset_index(drop=True)


# --- Natural Earth -------------------------------------------------------------

def load_borders(cache_dir: Path) -> dict:
    path = Path(cache_dir) / "ne_50m_admin_0_countries.geojson"
    if not path.exists():
        path.write_bytes(_get(NE_URL).content)
    return json.loads(path.read_text(encoding="utf-8"))
