# Europe on Tap

How Europe drinks, where it brews, and what is on the shelf. A data science project built on public data: WHO alcohol statistics, Wikidata and Open Food Facts.

![Beer's share of alcohol, 1970 vs 2024](reports/figures/map_beer_share.png)

## Questions

1. **Drinking cultures.** Europe used to split cleanly into beer, wine and spirits countries. Is that still true?
2. **Brewing heritage.** Where are Europe's breweries, and when were they founded?
3. **On the shelf** (in progress). How strong is the beer sold in each country, and can a model tell a beer's style from its name?

## 1. Drinking cultures are converging

Data: WHO Global Health Observatory, recorded alcohol per adult (15+) in litres of pure alcohol, split by beer, wine and spirits. It covers 41 European countries from 1960 to 2024; 28 of them already have data for 1970. Each drink's *share* is computed over beer + wine + spirits.

![Convergence](reports/figures/convergence.png)

The traditional beer countries now get a much smaller part of their alcohol from beer, and the wine countries a much larger one:

| Country | Beer share 1970 | Beer share 2024 |
|---|---|---|
| United Kingdom | 80% | 39% |
| Ireland | 73% | 47% |
| Denmark | 74% | 38% |
| Spain | 18% | 42% |
| Italy | 4% | 27% |
| Portugal | 5% | 27% |

Two standard convergence tests confirm the pattern:

- **Sigma-convergence:** over the same 26 countries every year, the standard deviation of beer's share fell from 0.234 (1965) to 0.086 (2024), and wine's from 0.274 to 0.140.
- **Beta-convergence:** regressing the 1970→2024 change in share on the 1970 level (n = 28) gives a strongly negative slope for every drink. Countries that started high fell the most:

| Share | Slope | R² | p-value |
|---|---|---|---|
| Beer | −0.861 | 0.848 | 4e-12 |
| Wine | −0.686 | 0.677 | 7.6e-08 |
| Spirits | −0.743 | 0.611 | 9e-07 |

K-means (k = 3) on the three shares, fitted on 1970 and 2024 together, recovers a beer, a wine and a spirits cluster. The ternary plot shows every country's move between the two years:

![Ternary plot](reports/figures/ternary.png)

24 of the 28 countries moved closer to an equal mix of the three drinks. 11 countries changed cluster; for example, Spain moved from the wine cluster to the beer cluster, and Denmark from beer to wine.

Beer is still the most common main drink. It was the largest of the three in 13 of 28 countries in 1970 and 20 of 41 in 2024. The heaviest beer drinkers in 2024 were Czechia (6.5 litres of pure alcohol from beer per adult), Austria (6.1) and Croatia (5.7).

## 2. A thousand years of brewing

Data: every item on Wikidata that is a brewery (or a subclass of one), has coordinates and is in a European country. That gives 1,478 breweries. The top countries are Germany (540), Belgium (234), Czechia (143), the United Kingdom (120) and Poland (108).

![Brewery map](reports/figures/brewery_map.png)

610 breweries have a founding year. The oldest are Bavarian monastery breweries: Weihenstephan (1040), Tegernsee (1050) and Weltenburg (1050). Foundings peak twice: in the late 19th century (186 founded 1850–1899) and in the craft-beer wave since 2000 (131).

![Founding years](reports/figures/brewery_foundings.png)

## 3. On the shelf (in progress)

This part uses the Open Food Facts product database (about 1.5 GB compressed). Its download is still running, and the results will be added once it finishes. The code is already in [`src/europe_on_tap/shelf.py`](src/europe_on_tap/shelf.py):

- Median ABV, share of strong beers (≥ 7%) and share of alcohol-free beers per country.
- Beer style from the product name and ABV: TF-IDF features into a logistic regression. Cross-validation is grouped by brand, so the model is never scored on a brand it has already seen.

## Limitations

- WHO figures are *recorded* consumption (sales and production). Home brewing, cross-border shopping and tourism are not counted.
- Shares leave out the WHO "other" category (cider, fermented drinks and so on).
- Wikidata coverage is uneven. Germany and Belgium are documented far better than Southern and Eastern Europe, so the map shows where breweries are *recorded*, not where they all are. Most entries (868 of 1,478) have no founding year.
- Some early founding dates on Wikidata belong to the monastery that later housed the brewery rather than to the brewery itself.

## Run it

```bash
git clone https://github.com/Wannkw/europe-on-tap.git
cd europe-on-tap
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_pipeline.py --skip-shelf   # parts 1-2, small downloads
python scripts/run_pipeline.py                # all parts, downloads ~1.5 GB once
pytest -q
```

Downloads are cached in `data/`. Figures go to `reports/figures/` and the numbers quoted here to `reports/metrics.json`.

## Project structure

```
src/europe_on_tap/
  data.py       downloads and caching (WHO, Wikidata, Open Food Facts, Natural Earth)
  drinking.py   convergence tests, dominant drink, k-means drinking cultures
  geo.py        Lambert equal-area projection and country polygons (no GIS dependencies)
  shelf.py      beer style labels, per-country summary, style classifier
  plots.py      all figures
scripts/run_pipeline.py
tests/
reports/        figures and metrics.json
```

## Data sources

- WHO Global Health Observatory, indicator SA_0000001400: https://www.who.int/data/gho
- Wikidata (CC0): https://www.wikidata.org
- Open Food Facts (ODbL): https://world.openfoodfacts.org
- Natural Earth country boundaries (public domain): https://www.naturalearthdata.com
