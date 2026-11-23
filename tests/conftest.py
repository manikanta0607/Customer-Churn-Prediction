import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn_prediction import config, data, modeling, predict  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return data.load_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return data.clean(raw_df, require_target=True)


@pytest.fixture(scope="session")
def small_artifact(clean_df):
    """A quickly trained model so tests do not depend on the committed model file."""
    sample = clean_df.sample(1500, random_state=config.RANDOM_STATE)
    x, y = data.split_features_target(sample)
    from sklearn.linear_model import LogisticRegression

    pipe = modeling.build_pipeline(LogisticRegression(max_iter=1000)).fit(x, y)
    return {
        "pipeline": pipe,
        "threshold": 0.35,
        "reference": predict.build_reference(x),
        "metadata": {"model": "test", "scikit_learn_version": None},
    }
