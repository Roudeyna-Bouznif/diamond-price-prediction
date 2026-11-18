import joblib
import numpy as np
import pandas as pd

from diamonds.data import make_synthetic
from diamonds.train import conformal_qhat


def test_point_accuracy_on_synthetic_law(trained):
    _, m = trained
    assert m["point_metrics_test"]["r2"] > 0.9


def test_conformal_coverage_close_to_nominal(trained):
    _, m = trained
    assert 0.86 <= m["interval_conformal"]["coverage"] <= 0.95


def test_intervals_are_ordered_and_contain_prediction(trained):
    out, _ = trained
    bundle = joblib.load(out / "models" / "model.joblib")
    res = bundle.predict(make_synthetic(300, seed=5))
    assert (res["lower"] <= res["price"]).all() and (res["price"] <= res["upper"]).all()


def test_more_carat_costs_more(trained):
    out, _ = trained
    bundle = joblib.load(out / "models" / "model.joblib")
    base = dict(cut="Ideal", color="G", clarity="VS2", depth=61.8, table=57.0)
    small = pd.DataFrame([{**base, "carat": 0.5, "x": 5.1, "y": 5.1, "z": 3.1}])
    big = pd.DataFrame([{**base, "carat": 1.5, "x": 7.4, "y": 7.4, "z": 4.6}])
    assert bundle.predict(big)["price"][0] > bundle.predict(small)["price"][0]


def test_conformal_qhat_is_zero_when_already_covered():
    y = np.zeros(100)
    assert conformal_qhat(y - 1, y + 1, y, 0.1) <= 0
