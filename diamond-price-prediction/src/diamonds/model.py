"""Model bundle: point model + quantile models + conformal correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import make_features


@dataclass
class Bundle:
    point: Any          # estimator predicting log(price)
    lower: Any          # quantile model, alpha/2
    upper: Any          # quantile model, 1 - alpha/2
    qhat: float         # conformal correction (log scale)
    alpha: float        # 0.10 -> 90 % intervals
    metadata: dict = field(default_factory=dict)

    def predict_log(self, X: pd.DataFrame, conformal: bool = True):
        mid = self.point.predict(X)
        shift = self.qhat if conformal else 0.0
        lo = np.minimum(self.lower.predict(X) - shift, mid)
        hi = np.maximum(self.upper.predict(X) + shift, mid)
        return lo, mid, hi

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw diamond attributes -> price and interval in USD."""
        lo, mid, hi = self.predict_log(make_features(raw))
        return pd.DataFrame(
            {"price": np.exp(mid), "lower": np.exp(lo), "upper": np.exp(hi)}
        )
