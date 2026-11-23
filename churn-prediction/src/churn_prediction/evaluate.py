"""Evaluation plots saved to docs/images."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

from . import config  # noqa: E402


def _save(fig, name: str) -> None:
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / name, dpi=130)
    plt.close(fig)


def plot_all(pipeline, x_test, y_test, proba, threshold, curve: pd.DataFrame) -> pd.DataFrame:
    """Create evaluation figures and return permutation importance per input feature."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color="#E4572E", label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.set_title("ROC curve (test set)")
    _save(fig, "eval_roc_curve.png")

    fig, ax = plt.subplots(figsize=(5, 4.2))
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, color="#4C78A8", label="Model")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.axhline(float(np.mean(y_test)), ls="--", color="grey", label="No-skill")
    ax.legend()
    ax.set_title("Precision-recall curve (test set)")
    _save(fig, "eval_pr_curve.png")

    pred = (np.asarray(proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    _save(fig, "eval_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(curve["threshold"], curve["net_benefit"], color="#2E7D32")
    best = curve.loc[curve["net_benefit"].idxmax()]
    ax.axvline(best["threshold"], ls="--", color="grey")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Net benefit per customer ($, assumed economics)")
    ax.set_title("Threshold selection (out-of-fold training scores)")
    _save(fig, "eval_threshold_curve.png")

    imp = permutation_importance(
        pipeline,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    importance = (
        pd.DataFrame({"feature": x_test.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#6A5ACD")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    _save(fig, "eval_feature_importance.png")
    return importance
