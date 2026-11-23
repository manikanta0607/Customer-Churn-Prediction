"""Model pipelines, candidate comparison and cost-based threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor, feature_step


def build_pipeline(estimator) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> classifier."""
    return Pipeline(
        [
            ("features", feature_step()),
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    """Algorithms to compare, each with a small hyper-parameter grid."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": (
            build_pipeline(LogisticRegression(max_iter=2000)),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            build_pipeline(RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1)),
            {"model__max_depth": [8, None], "model__min_samples_leaf": [3, 10]},
        ),
        "Gradient Boosting": (
            build_pipeline(HistGradientBoostingClassifier(max_iter=150, random_state=rs)),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [8, 15],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    }


@dataclass
class SearchResult:
    name: str
    search: GridSearchCV
    summary: dict


def compare_models(x_train: pd.DataFrame, y_train: pd.Series) -> list[SearchResult]:
    """Tune every candidate with stratified CV and return results (best-first)."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1"}
    results: list[SearchResult] = []
    for name, (pipe, grid) in candidate_models().items():
        search = GridSearchCV(pipe, grid, scoring=scoring, refit="roc_auc", cv=cv, n_jobs=1)
        search.fit(x_train, y_train)
        i = search.best_index_
        r = search.cv_results_
        summary = {
            "model": name,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_roc_auc_mean": float(r["mean_test_roc_auc"][i]),
            "cv_roc_auc_std": float(r["std_test_roc_auc"][i]),
            "cv_average_precision_mean": float(r["mean_test_average_precision"][i]),
            "cv_f1_at_0.5_mean": float(r["mean_test_f1"][i]),
        }
        results.append(SearchResult(name, search, summary))
    return sorted(results, key=lambda r: r.summary["cv_roc_auc_mean"], reverse=True)


def expected_net_benefit(y_true, proba, threshold: float) -> float:
    """Average net benefit per customer of contacting everyone scored >= threshold.

    Contacting a true churner yields ``success_rate * customer_value - offer_cost``;
    contacting a loyal customer costs ``offer_cost``. Uncontacted customers add 0.
    The constants live in ``config`` and are illustrative assumptions.
    """
    y_true = np.asarray(y_true)
    contact = np.asarray(proba) >= threshold
    tp = np.sum(contact & (y_true == 1))
    fp = np.sum(contact & (y_true == 0))
    gain = tp * (config.OFFER_SUCCESS_RATE * config.CUSTOMER_VALUE - config.OFFER_COST)
    loss = fp * config.OFFER_COST
    return float((gain - loss) / len(y_true))


def choose_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    """Pick the threshold with the best expected net benefit on out-of-fold scores."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curve = pd.DataFrame({"threshold": grid, "net_benefit": [expected_net_benefit(y_true, proba, t) for t in grid]})
    best = curve.loc[curve["net_benefit"].idxmax(), "threshold"]
    return float(best), curve


def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Standard metrics at a chosen threshold (probability metrics are threshold-free)."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "contacted_share": float(pred.mean()),
        "net_benefit_per_customer": expected_net_benefit(y_true, proba, threshold),
    }


# Simplest (most interpretable) first. Used by the one-standard-deviation rule below.
SIMPLICITY_ORDER = ["Logistic Regression", "Gradient Boosting", "Random Forest"]


def select_model(results: list[SearchResult]) -> SearchResult:
    """One-standard-deviation rule.

    Cross-validated scores of different models are often statistically tied, so
    picking the numerically highest one mostly selects noise. Instead, keep every
    model whose mean CV ROC-AUC is within one standard deviation of the best and
    choose the simplest of them.
    """
    best = max(results, key=lambda r: r.summary["cv_roc_auc_mean"])
    cutoff = best.summary["cv_roc_auc_mean"] - best.summary["cv_roc_auc_std"]
    eligible = [r for r in results if r.summary["cv_roc_auc_mean"] >= cutoff]
    return min(eligible, key=lambda r: SIMPLICITY_ORDER.index(r.name))
