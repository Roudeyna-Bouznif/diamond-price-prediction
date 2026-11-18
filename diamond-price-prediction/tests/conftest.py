import pytest

from diamonds.data import make_synthetic
from diamonds.train import run_training


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    metrics = run_training(make_synthetic(6000, seed=1), out, cv=False, figures=False)
    return out, metrics
