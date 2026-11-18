"""Part 3: beer on European shelves (Open Food Facts).

Countries here are where a product is *sold* (``countries_tags``), not where it
was brewed. Styles come from the community-maintained category tags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# First matching rule wins. Specific styles come before the broad "lager" tag,
# because e.g. a fruit radler is often also tagged as a lager.
STYLE_RULES: list[tuple[str, set[str]]] = [
    ("Fruit & flavoured", {"en:beers-with-fruits", "en:flavored-beers", "en:radler-beers", "en:shandy",
                           "en:mixed-beer-drinks"}),
    ("Sour & lambic", {"en:sour-beers", "en:lambic-beers"}),
    ("Abbey & Trappist", {"en:abbey-ales", "en:trappist-beers", "en:triple-ales",
                          "en:belgian-style-dubbel-ales"}),
    ("Wheat", {"en:wheat-beers"}),
    ("Stout & porter", {"en:stouts", "en:porters"}),
    ("Pale ale & IPA", {"en:pale-ales"}),
    ("Amber & dark", {"en:amber-beers", "en:dark-ales"}),
    ("Lager & pilsner", {"en:lagers", "en:pilsners", "en:pale-lagers"}),
]
NON_ALCOHOLIC_TAG = "en:non-alcoholic-beers"

EUROPEAN_COUNTRY_TAGS = {
    "en:austria": "Austria", "en:belgium": "Belgium", "en:czech-republic": "Czechia",
    "en:denmark": "Denmark", "en:finland": "Finland", "en:france": "France",
    "en:germany": "Germany", "en:greece": "Greece", "en:hungary": "Hungary",
    "en:ireland": "Ireland", "en:italy": "Italy", "en:luxembourg": "Luxembourg",
    "en:netherlands": "Netherlands", "en:norway": "Norway", "en:poland": "Poland",
    "en:portugal": "Portugal", "en:romania": "Romania", "en:slovakia": "Slovakia",
    "en:spain": "Spain", "en:sweden": "Sweden", "en:switzerland": "Switzerland",
    "en:united-kingdom": "United Kingdom", "en:bulgaria": "Bulgaria", "en:croatia": "Croatia",
    "en:lithuania": "Lithuania", "en:latvia": "Latvia", "en:estonia": "Estonia",
    "en:russia": "Russia", "en:ukraine": "Ukraine", "en:slovenia": "Slovenia", "en:serbia": "Serbia",
}


def label_style(tags: list[str]) -> str | None:
    tags = set(tags)
    for style, rule in STYLE_RULES:
        if tags & rule:
            return style
    return None


def prepare(off: pd.DataFrame) -> pd.DataFrame:
    """Add style / non-alcoholic labels and drop implausible ABV values."""
    df = off.copy()
    df.loc[~df["abv"].between(0, 20), "abv"] = np.nan  # e.g. 50 = typo, or a spirit miscategorised
    df["non_alcoholic"] = df["categories_tags"].apply(lambda t: NON_ALCOHOLIC_TAG in t) | (df["abv"] <= 0.5)
    df["style"] = df["categories_tags"].apply(label_style)
    df.loc[df["non_alcoholic"], "style"] = None  # alcohol-free versions span every style
    return df


def by_country(df: pd.DataFrame, min_products: int = 100) -> pd.DataFrame:
    """One row per (product, European country it is sold in)."""
    long = df.explode("countries_tags")
    long = long[long["countries_tags"].isin(EUROPEAN_COUNTRY_TAGS)]
    long["country"] = long["countries_tags"].map(EUROPEAN_COUNTRY_TAGS)
    counts = long["country"].value_counts()
    return long[long["country"].isin(counts[counts >= min_products].index)]


def country_summary(long: pd.DataFrame) -> pd.DataFrame:
    alcoholic = long[~long["non_alcoholic"] & long["abv"].notna()]
    return pd.DataFrame({
        "products": long.groupby("country").size(),
        "median_abv": alcoholic.groupby("country")["abv"].median(),
        "share_strong": alcoholic.groupby("country")["abv"].apply(lambda s: (s >= 7).mean()),
        "share_non_alcoholic": long.groupby("country")["non_alcoholic"].mean(),
    }).sort_values("median_abv")


# --- Style classifier -------------------------------------------------------------

class TextAbvClassifier(BaseEstimator, ClassifierMixin):
    """TF-IDF on the product name (words + character n-grams) plus ABV, into a
    multinomial logistic regression. Missing ABV is imputed and flagged."""

    def __init__(self, use_text: bool = True, use_abv: bool = True, C: float = 5.0):
        self.use_text = use_text
        self.use_abv = use_abv
        self.C = C

    def _abv_block(self, X: pd.DataFrame):
        abv = X["abv"].to_numpy(dtype=float)
        missing = np.isnan(abv)
        filled = np.where(missing, self.abv_median_, abv)
        # Scaled ABV and its square let a linear model express "middle strength".
        z = (filled - self.abv_median_) / 2.0
        return csr_matrix(np.column_stack([z, z**2, missing.astype(float)]))

    def _features(self, X: pd.DataFrame, fit: bool):
        blocks = []
        if self.use_text:
            text = X["product_name"].fillna("")
            if fit:
                blocks.append(self.words_.fit_transform(text))
                blocks.append(self.chars_.fit_transform(text))
            else:
                blocks.append(self.words_.transform(text))
                blocks.append(self.chars_.transform(text))
        if self.use_abv:
            blocks.append(self._abv_block(X))
        return hstack(blocks).tocsr()

    def fit(self, X: pd.DataFrame, y):
        self.abv_median_ = float(np.nanmedian(X["abv"])) if X["abv"].notna().any() else 5.0
        self.words_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
        self.chars_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                      strip_accents="unicode")
        self.model_ = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.model_.fit(self._features(X, fit=True), y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(self._features(X, fit=False))


class AbvOnly(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.model_ = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)
        self.model_.fit(X[["abv"]], y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X[["abv"]])


def candidate_models() -> dict:
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "ABV only (gradient boosting)": AbvOnly(),
        "Name only (TF-IDF + logistic)": TextAbvClassifier(use_abv=False),
        "Name + ABV (TF-IDF + logistic)": TextAbvClassifier(),
    }


def cross_validate(data: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Brand-grouped CV: a brand never appears in both train and test, so the model
    must generalise from words like 'weizen' rather than memorise 'Paulaner'."""
    X, y = data[["product_name", "abv"]], data["style"].to_numpy()
    groups = data["brands"].replace("", np.nan).fillna(data["code"].astype(str)).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups))

    scores, oof = {}, {}
    for name, model in candidate_models().items():
        pred = np.empty(len(y), dtype=object)
        for train, test in folds:
            fitted = clone(model).fit(X.iloc[train], y[train])
            pred[test] = fitted.predict(X.iloc[test])
        scores[name] = {
            "macro_f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
        }
        oof[name] = pred
    return scores, oof


def top_terms(model: TextAbvClassifier, n: int = 6) -> dict[str, list[str]]:
    """Highest-weighted whole words per class: what the model has learned."""
    vocab = np.array(model.words_.get_feature_names_out())
    weights = model.model_.coef_[:, : len(vocab)]
    return {cls: vocab[np.argsort(-w)[:n]].tolist() for cls, w in zip(model.classes_, weights)}
