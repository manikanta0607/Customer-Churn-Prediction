"""Inference helpers shared by the training script, tests and the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, data


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file is missing."""


def build_reference(x_train: pd.DataFrame) -> dict:
    """Typical value per feature (median / most common) used for explanations."""
    ref: dict = {}
    for col in config.FEATURE_COLUMNS:
        if col in config.NUMERIC_COLUMNS:
            ref[col] = float(x_train[col].median())
        else:
            ref[col] = x_train[col].mode().iloc[0]
            if isinstance(ref[col], (np.integer,)):
                ref[col] = int(ref[col])
    return ref


def load_artifact(path: Path | str = config.MODEL_PATH) -> dict:
    """Load the saved artifact: pipeline, threshold, reference values and metadata."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"Model not found at {path}. Train it first: `python scripts/train.py`.")
    return joblib.load(path)


def risk_band(prob: float, threshold: float) -> str:
    """Human-friendly band relative to the decision threshold."""
    if prob >= threshold:
        return "High"
    if prob >= threshold / 2:
        return "Medium"
    return "Low"


def predict_frame(artifact: dict, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Score a raw customer table. Returns cleaned rows plus probability, flag and band."""
    thr = artifact["threshold"] if threshold is None else float(threshold)
    cleaned = data.clean(df, require_target=False)
    proba = artifact["pipeline"].predict_proba(cleaned[config.FEATURE_COLUMNS])[:, 1]
    out = cleaned.copy()
    out["churn_probability"] = np.round(proba, 4)
    out["predicted_churn"] = (proba >= thr).astype(int)
    out["risk_band"] = [risk_band(p, thr) for p in proba]
    return out


# Related features are swapped together: changing e.g. tenure without total charges
# would create unrealistic customers and produce misleading offsetting effects.
FACTOR_GROUPS: dict[str, list[str]] = {
    "Tenure & total charges": ["tenure", "TotalCharges"],
    "Internet, add-ons & monthly price": ["InternetService", "MonthlyCharges", *config.ADD_ON_SERVICES],
    "Phone services": ["PhoneService", "MultipleLines"],
    "Contract": ["Contract"],
    "Payment & billing": ["PaymentMethod", "PaperlessBilling"],
    "Demographics": ["gender", "SeniorCitizen", "Partner", "Dependents"],
}


def _fmt(value) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def explain_customer(artifact: dict, row: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    """Approximate per-customer drivers by swapping groups of related features.

    Each factor group is replaced by typical values (median / most common in the
    training data) and the drop in churn probability is recorded. Positive
    ``effect_pp`` means the customer's actual values raise risk relative to a typical
    customer, in percentage points. This is a simple what-if approximation, not an
    exact attribution method, and effects do not add up to the total.
    """
    cleaned = data.clean(row, require_target=False)[config.FEATURE_COLUMNS]
    base_row = cleaned.iloc[[0]]
    pipe = artifact["pipeline"]
    ref = artifact["reference"]
    base = float(pipe.predict_proba(base_row)[:, 1][0])

    names, variants, details = [], [], []
    for factor, cols in FACTOR_GROUPS.items():
        differing = [c for c in cols if str(base_row.iloc[0][c]) != str(ref[c])]
        if not differing:
            continue
        v = base_row.copy()
        for c in cols:
            v[c] = ref[c]
        names.append(factor)
        variants.append(v)
        details.append("; ".join(f"{c}: {_fmt(base_row.iloc[0][c])} (typical {_fmt(ref[c])})" for c in differing))
    if not variants:
        return pd.DataFrame(columns=["factor", "effect_pp", "details"])
    probs = pipe.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    result = pd.DataFrame({"factor": names, "effect_pp": np.round((base - probs) * 100, 2), "details": details})
    result["abs"] = result["effect_pp"].abs()
    return result.sort_values("abs", ascending=False).head(top_k).drop(columns="abs").reset_index(drop=True)
