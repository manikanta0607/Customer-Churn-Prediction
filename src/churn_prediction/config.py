"""Central configuration: paths, column groups and business assumptions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
IMAGES_DIR = ROOT / "docs" / "images"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ID_COLUMN = "customerID"
TARGET = "Churn"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLUMNS = ["SeniorCitizen"]
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
ADD_ON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
# Columns that must be present in any data used for prediction.
FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS

# --- Illustrative retention-campaign economics (assumptions, NOT dataset facts) ---
# Used only to choose a decision threshold. Change them to match a real business.
CUSTOMER_VALUE = 500.0  # revenue kept if a churner is retained
OFFER_COST = 50.0  # cost of contacting/offering an incentive to one customer
OFFER_SUCCESS_RATE = 0.30  # share of contacted true churners who stay
