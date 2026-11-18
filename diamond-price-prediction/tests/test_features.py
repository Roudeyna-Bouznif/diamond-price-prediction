import pandas as pd
import pytest

from diamonds.data import make_synthetic
from diamonds.features import FEATURES, clean, make_features


def test_clean_removes_impossible_rows():
    df = make_synthetic(200)
    bad = df.iloc[[0]].copy()
    bad[["x", "y", "z"]] = 0
    out = clean(pd.concat([df, bad], ignore_index=True))
    assert (out[["x", "y", "z"]] > 0).all().all()


def test_make_features_columns_and_codes():
    X = make_features(make_synthetic(50))
    assert list(X.columns) == FEATURES
    assert X["cut"].between(0, 4).all() and not X.isna().any().any()


def test_unknown_category_raises():
    df = make_synthetic(5)
    df.loc[0, "cut"] = "Shiny"
    with pytest.raises(ValueError, match="Unknown value"):
        make_features(df)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Missing columns"):
        make_features(make_synthetic(5).drop(columns=["carat"]))
