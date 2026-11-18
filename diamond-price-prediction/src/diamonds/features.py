"""Cleaning and feature engineering (shared by training and serving)."""

from __future__ import annotations

import pandas as pd

# Ordered from worst to best quality, so the integer code is meaningful.
CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

ORDINALS = {"cut": CUT, "color": COLOR, "clarity": CLARITY}
NUMERIC = ["carat", "depth", "table", "x", "y", "z"]
FEATURES = NUMERIC + list(ORDINALS) + ["volume"]
RAW_COLUMNS = NUMERIC + list(ORDINALS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and physically impossible / extreme rows (training only)."""
    out = df.drop_duplicates().copy()
    mask = (
        (out["price"] > 0)
        & (out[["x", "y", "z"]] > 0).all(axis=1)
        & (out["y"] < 20)
        & (out["z"] < 20)
        & out["depth"].between(40, 80)
        & out["table"].between(40, 100)
    )
    return out[mask].reset_index(drop=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model matrix. Raises ValueError on unknown categories."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[NUMERIC].astype(float).copy()
    for col, order in ORDINALS.items():
        codes = df[col].map({v: i for i, v in enumerate(order)})
        if codes.isna().any():
            bad = sorted(set(df.loc[codes.isna(), col].astype(str)))
            raise ValueError(f"Unknown value(s) for '{col}': {bad}. Allowed: {order}")
        out[col] = codes.astype(float)
    out["volume"] = df["x"].astype(float) * df["y"].astype(float) * df["z"].astype(float)
    return out[FEATURES]
