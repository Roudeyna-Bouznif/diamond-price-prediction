"""Data loading. The real dataset is downloaded on demand; a synthetic generator
allows tests and CI to run offline."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CLARITY, COLOR, CUT

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
DEFAULT_PATH = Path("data/diamonds.csv")


def load_diamonds(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(URL, path)
    return pd.read_csv(path)


def make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Diamond-like data with a known log-linear price law (for tests/CI only)."""
    rng = np.random.default_rng(seed)
    carat = np.clip(rng.lognormal(-0.3, 0.55, n), 0.2, 4.0).round(2)
    cut = rng.choice(CUT, n)
    color = rng.choice(COLOR, n)
    clarity = rng.choice(CLARITY, n)
    x = 6.5 * carat ** (1 / 3) * rng.normal(1, 0.01, n)
    y = x * rng.normal(1, 0.005, n)
    z = 0.61 * x * rng.normal(1, 0.01, n)
    depth = z / ((x + y) / 2) * 100
    table = rng.normal(57, 2, n)
    log_price = (
        8.3
        + 1.8 * np.log(carat)
        + 0.04 * np.array([CUT.index(v) for v in cut])
        + 0.06 * np.array([COLOR.index(v) for v in color])
        + 0.09 * np.array([CLARITY.index(v) for v in clarity])
        + rng.normal(0, 0.12, n)
    )
    return pd.DataFrame(
        {
            "carat": carat, "cut": cut, "color": color, "clarity": clarity,
            "depth": depth.round(1), "table": table.round(1),
            "price": np.exp(log_price).round().astype(int),
            "x": x.round(2), "y": y.round(2), "z": z.round(2),
        }
    )
