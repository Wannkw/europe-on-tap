import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from europe_on_tap import shelf  # noqa: E402


def test_style_priority_specific_before_lager():
    assert shelf.label_style(["en:beers", "en:lagers", "en:radler-beers"]) == "Fruit & flavoured"
    assert shelf.label_style(["en:beers", "en:lagers", "en:wheat-beers"]) == "Wheat"
    assert shelf.label_style(["en:beers", "en:pilsners"]) == "Lager & pilsner"
    assert shelf.label_style(["en:beers", "en:blonde-ales"]) is None


def _products(rows):
    return pd.DataFrame(rows, columns=["code", "product_name", "brands", "abv", "categories_tags", "countries_tags"])


def test_prepare_flags_alcohol_free_and_drops_bad_abv():
    df = shelf.prepare(_products([
        ("1", "Pils", "a", 4.9, ["en:beers", "en:lagers"], ["en:germany"]),
        ("2", "Pils 0.0", "a", 0.0, ["en:beers", "en:lagers"], ["en:germany"]),
        ("3", "Typo", "b", 49.0, ["en:beers", "en:lagers"], ["en:france"]),
    ]))
    assert df["non_alcoholic"].tolist() == [False, True, False]
    assert df.loc[0, "style"] == "Lager & pilsner"
    assert pd.isna(df.loc[1, "style"])
    assert np.isnan(df.loc[2, "abv"])


def test_classifier_learns_obvious_words():
    rng = np.random.default_rng(0)
    rows = []
    vocab = {"Wheat": "weizen hefe", "Stout & porter": "stout dark", "Lager & pilsner": "pils lager"}
    for i in range(90):
        style = list(vocab)[i % 3]
        word = vocab[style].split()[i % 2]
        rows.append({"product_name": f"brand{i} {word}", "abv": rng.normal(5, 0.5), "style": style})
    data = pd.DataFrame(rows)
    model = shelf.TextAbvClassifier().fit(data[["product_name", "abv"]], data["style"])
    test = pd.DataFrame({"product_name": ["new weizen", "big stout", "cold pils"], "abv": [5.2, np.nan, 4.8]})
    assert model.predict(test).tolist() == ["Wheat", "Stout & porter", "Lager & pilsner"]


def test_cv_never_shares_brands_between_folds():
    from sklearn.model_selection import StratifiedGroupKFold

    styles = ["A", "B"] * 20
    brands = [f"b{i // 4}" for i in range(40)]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for train, test in cv.split(np.zeros(40), styles, brands):
        assert not set(np.array(brands)[train]) & set(np.array(brands)[test])
