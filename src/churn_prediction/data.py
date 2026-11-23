"""Loading, validating and cleaning customer data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

_NO_SERVICE_COLUMNS = config.ADD_ON_SERVICES + ["MultipleLines"]


class DataValidationError(ValueError):
    """Raised when input data does not match the expected schema."""


def load_raw(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `python scripts/download_data.py`.")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    """Check that all required columns exist and the frame is not empty."""
    required = list(config.FEATURE_COLUMNS)
    if require_target:
        required.append(config.TARGET)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise DataValidationError("The data contains no rows.")


def clean(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Return a cleaned copy of ``df``.

    Steps: validate schema, trim text, convert ``TotalCharges`` to numeric (blank
    values belong to brand-new customers with tenure 0 and become 0), merge the
    redundant "No phone/internet service" labels into "No", drop duplicate customer
    IDs and encode the target (Yes=1, No=0) when present.
    """
    validate_schema(df, require_target=require_target)
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) in ("str", "string"):
            out[col] = out[col].astype("string").str.strip()

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    new_customers = out["tenure"] == 0
    out.loc[new_customers & out["TotalCharges"].isna(), "TotalCharges"] = 0.0

    still_missing = out[config.NUMERIC_COLUMNS].isna().any(axis=1)
    if still_missing.any():
        raise DataValidationError(f"{int(still_missing.sum())} row(s) have non-numeric tenure/charges values.")

    for col in _NO_SERVICE_COLUMNS:
        out[col] = out[col].replace({"No phone service": "No", "No internet service": "No"})

    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce").fillna(0)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)

    if config.ID_COLUMN in out.columns:
        out = out.drop_duplicates(subset=config.ID_COLUMN, keep="first")

    if config.TARGET in out.columns:
        mapped = out[config.TARGET].map({"Yes": 1, "No": 0})
        if mapped.isna().any():
            if require_target:
                raise DataValidationError("Churn column must contain only Yes/No.")
            out = out.drop(columns=[config.TARGET])
        else:
            out[config.TARGET] = mapped.astype(int)

    return out.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into model inputs and the churn label."""
    return df[config.FEATURE_COLUMNS].copy(), df[config.TARGET].copy()


def train_test(df: pd.DataFrame):
    """Stratified train/test split of a cleaned labelled frame."""
    x, y = split_features_target(df)
    return train_test_split(
        x,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
