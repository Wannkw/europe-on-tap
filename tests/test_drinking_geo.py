import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from europe_on_tap import drinking, geo  # noqa: E402


def test_ternary_corners():
    x, y = drinking.ternary_xy([1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert np.allclose(x, [0, 1, 0.5])
    assert np.allclose(y, [0, 0, np.sqrt(3) / 2])


def test_laea_centre_and_distance():
    x, y = geo.laea(10.0, 52.0)
    assert x == pytest.approx(0, abs=1e-9) and y == pytest.approx(0, abs=1e-9)
    # Equal-area projection: north-south distance from the centre is close to great-circle.
    _, y_north = geo.laea(10.0, 62.0)
    assert y_north == pytest.approx(6371 * np.radians(10), rel=0.01)


def _toy_who(rows):
    df = pd.DataFrame(rows, columns=["iso3", "country", "year", "beer_share", "wine_share", "spirits_share"])
    return df


def test_balanced_panel_drops_countries_with_gaps():
    who = _toy_who([
        ("AAA", "A", 2000, 0.5, 0.3, 0.2), ("AAA", "A", 2001, 0.5, 0.3, 0.2),
        ("BBB", "B", 2001, 0.2, 0.6, 0.2),
    ])
    assert drinking.balanced_panel(who, 2000, 2001)["iso3"].unique().tolist() == ["AAA"]


def test_beta_convergence_detects_catch_up():
    # Low starters gain, high starters lose: slope must be negative.
    who = _toy_who([
        ("AAA", "A", 1970, 0.8, 0.1, 0.1), ("AAA", "A", 2020, 0.5, 0.3, 0.2),
        ("BBB", "B", 1970, 0.1, 0.8, 0.1), ("BBB", "B", 2020, 0.4, 0.4, 0.2),
        ("CCC", "C", 1970, 0.4, 0.3, 0.3), ("CCC", "C", 2020, 0.45, 0.3, 0.25),
    ])
    result = drinking.beta_convergence(who, 1970, 2020)
    assert result["slope"] < 0 and result["n"] == 3


def test_clusters_are_named_after_dominant_drink():
    rng = np.random.default_rng(0)
    rows = []
    for i, centre in enumerate([(0.7, 0.15, 0.15), (0.15, 0.7, 0.15), (0.15, 0.15, 0.7)]):
        for j in range(5):
            for year in (1970, 2020):
                shares = np.clip(np.array(centre) + rng.normal(0, 0.02, 3), 0, None)
                shares /= shares.sum()
                rows.append((f"C{i}{j}", f"C{i}{j}", year, *shares))
    out = drinking.cluster_cultures(_toy_who(rows), (1970, 2020))
    assert set(out["cluster"]) == {"Beer culture", "Wine culture", "Spirits culture"}
    assert (out[out["iso3"].str.startswith("C0")]["cluster"] == "Beer culture").all()
