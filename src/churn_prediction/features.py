"""Feature engineering and the scikit-learn preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

ENGINEERED_NUMERIC = ["avg_monthly_spend", "n_addon_services"]
TENURE_BINS = [-1, 12, 24, 48, 10_000]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49m+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Pure function so it behaves the same at train and predict time."""
    out = df.copy()
    out["avg_monthly_spend"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["n_addon_services"] = (out[config.ADD_ON_SERVICES] == "Yes").sum(axis=1)
    out["tenure_group"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype("string")
    return out


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical ones."""
    numeric = config.NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    categorical = config.CATEGORICAL_COLUMNS + ["tenure_group"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("bin", "passthrough", config.BINARY_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )


def feature_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)
