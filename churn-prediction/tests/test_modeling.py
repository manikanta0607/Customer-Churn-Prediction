import numpy as np
import pytest

from churn_prediction import config, modeling


def test_net_benefit_arithmetic():
    y = np.array([1, 0])
    p = np.array([0.9, 0.9])
    tp_gain = config.OFFER_SUCCESS_RATE * config.CUSTOMER_VALUE - config.OFFER_COST
    expected = (tp_gain - config.OFFER_COST) / 2
    assert modeling.expected_net_benefit(y, p, 0.5) == pytest.approx(expected)
    assert modeling.expected_net_benefit(y, p, 0.95) == 0.0


def test_choose_threshold_prefers_separating_value():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.1, 2000), 0, 1)
    thr, curve = modeling.choose_threshold(y, p)
    assert 0.05 <= thr <= 0.95
    assert curve["net_benefit"].max() == pytest.approx(modeling.expected_net_benefit(y, p, thr))
    assert thr > 0.3  # should not flag everyone when scores separate the classes


def test_classification_metrics_perfect_and_keys():
    y = np.array([0, 0, 1, 1])
    m = modeling.classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == m["roc_auc"] == 1.0
    assert {"brier_score", "average_precision", "contacted_share"} <= set(m)


def test_select_model_prefers_simplest_within_one_sd():
    class R:
        def __init__(self, name, mean, std):
            self.name = name
            self.summary = {"cv_roc_auc_mean": mean, "cv_roc_auc_std": std}

    tied = [R("Random Forest", 0.848, 0.010), R("Gradient Boosting", 0.847, 0.010), R("Logistic Regression", 0.846, 0.012)]
    assert modeling.select_model(tied).name == "Logistic Regression"
    clear_winner = [R("Random Forest", 0.90, 0.005), R("Logistic Regression", 0.80, 0.005)]
    assert modeling.select_model(clear_winner).name == "Random Forest"


def test_pipeline_trains_and_beats_chance(clean_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from churn_prediction import data

    x_tr, x_te, y_tr, y_te = data.train_test(clean_df)
    pipe = modeling.build_pipeline(LogisticRegression(max_iter=1000)).fit(x_tr, y_tr)
    proba = pipe.predict_proba(x_te)[:, 1]
    assert proba.shape == (len(x_te),) and proba.min() >= 0 and proba.max() <= 1
    assert roc_auc_score(y_te, proba) > 0.80


def test_candidates_cover_three_algorithms():
    assert set(modeling.candidate_models()) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
