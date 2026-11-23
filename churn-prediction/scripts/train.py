"""Train, compare and save the churn model. Run: python scripts/train.py"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from churn_prediction import config, data, evaluate, modeling, predict  # noqa: E402


def main() -> None:
    df = data.clean(data.load_raw(), require_target=True)
    x_train, x_test, y_train, y_test = data.train_test(df)
    print(f"Train {len(x_train)} rows | Test {len(x_test)} rows | churn rate {y_train.mean():.3f}")

    results = modeling.compare_models(x_train, y_train)
    print("\nCross-validated comparison (train set only):")
    for r in results:
        s = r.summary
        print(f"  {s['model']:<20} AUC {s['cv_roc_auc_mean']:.4f} +/- {s['cv_roc_auc_std']:.4f}  {s['best_params']}")

    best = modeling.select_model(results)
    print(f"Selection (1-SD rule, simplest within one std of the top AUC): {best.name}")
    pipeline = best.search.best_estimator_

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(pipeline, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, curve = modeling.choose_threshold(y_train, oof)

    proba_test = pipeline.predict_proba(x_test)[:, 1]
    at_threshold = modeling.classification_metrics(y_test, proba_test, threshold)
    at_half = modeling.classification_metrics(y_test, proba_test, 0.5)
    importance = evaluate.plot_all(pipeline, x_test, y_test, proba_test, threshold, curve)

    metrics = {
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "library_versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "dataset": {
            "rows": int(len(df)),
            "churn_rate": float(df[config.TARGET].mean()),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "selected_model": best.summary["model"],
        "selected_params": best.summary["best_params"],
        "model_comparison": [r.summary for r in results],
        "threshold": threshold,
        "economic_assumptions": {
            "customer_value": config.CUSTOMER_VALUE,
            "offer_cost": config.OFFER_COST,
            "offer_success_rate": config.OFFER_SUCCESS_RATE,
            "note": "Illustrative assumptions used only to choose the threshold.",
        },
        "test_metrics_at_selected_threshold": at_threshold,
        "test_metrics_at_0.5": at_half,
        "baseline_majority_class_accuracy": float(1 - y_test.mean()),
        "top_features_permutation": importance.head(10).round(4).to_dict(orient="records"),
    }
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "reference": predict.build_reference(x_train),
        "metadata": {
            "model": best.summary["model"],
            "trained_at_utc": metrics["trained_at_utc"],
            "scikit_learn_version": sklearn.__version__,
        },
    }
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.MODEL_PATH, compress=3)

    print(f"\nSelected: {best.summary['model']} | threshold {threshold:.2f}")
    print("Test metrics @ selected threshold:", json.dumps({k: round(v, 4) for k, v in at_threshold.items()}))
    print("Test metrics @ 0.5:", json.dumps({k: round(v, 4) for k, v in at_half.items()}))
    print(f"Majority-class baseline accuracy: {metrics['baseline_majority_class_accuracy']:.4f}")
    print(f"Saved model ({config.MODEL_PATH.stat().st_size / 1024:.0f} KB) and metrics.")


if __name__ == "__main__":
    main()
