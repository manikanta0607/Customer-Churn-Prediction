import joblib
import pandas as pd
import pytest

from churn_prediction import config, data, predict


def test_risk_band_rules():
    assert predict.risk_band(0.5, 0.35) == "High"
    assert predict.risk_band(0.35, 0.35) == "High"
    assert predict.risk_band(0.2, 0.35) == "Medium"
    assert predict.risk_band(0.05, 0.35) == "Low"


def test_predict_frame_columns_and_ranges(small_artifact, raw_df):
    out = predict.predict_frame(small_artifact, raw_df.head(50))
    assert len(out) == 50
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["predicted_churn"]) <= {0, 1}
    assert set(out["risk_band"]) <= {"High", "Medium", "Low"}


def test_threshold_changes_flags_monotonically(small_artifact, raw_df):
    low = predict.predict_frame(small_artifact, raw_df.head(200), threshold=0.1)["predicted_churn"].sum()
    high = predict.predict_frame(small_artifact, raw_df.head(200), threshold=0.9)["predicted_churn"].sum()
    assert low >= high


def test_predict_works_without_churn_column(small_artifact, raw_df):
    out = predict.predict_frame(small_artifact, raw_df.drop(columns=["Churn"]).head(5))
    assert len(out) == 5


def test_predict_rejects_bad_input(small_artifact, raw_df):
    with pytest.raises(data.DataValidationError):
        predict.predict_frame(small_artifact, raw_df.drop(columns=["tenure"]).head(3))


def test_unseen_category_does_not_crash(small_artifact, raw_df):
    row = raw_df.head(1).copy()
    row["PaymentMethod"] = "Crypto"
    assert 0 <= predict.predict_frame(small_artifact, row)["churn_probability"].iloc[0] <= 1


def test_explain_customer_shape_and_sign(small_artifact, raw_df):
    new_customer = raw_df.query("customerID == '7590-VHVEG'")
    drivers = predict.explain_customer(small_artifact, new_customer, top_k=3)
    assert list(drivers.columns) == ["factor", "effect_pp", "details"]
    assert 1 <= len(drivers) <= 3
    tenure_row = drivers[drivers["factor"] == "Tenure & total charges"]
    assert not tenure_row.empty and tenure_row["effect_pp"].iloc[0] > 0  # 1-month customer: tenure raises risk


def test_explain_typical_customer_is_empty(small_artifact, raw_df):
    row = raw_df.head(1).copy()
    for col, value in small_artifact["reference"].items():
        row[col] = value
    assert predict.explain_customer(small_artifact, row).empty


def test_missing_model_error(tmp_path):
    with pytest.raises(predict.ModelNotFoundError, match="train"):
        predict.load_artifact(tmp_path / "missing.joblib")


def test_artifact_roundtrip(small_artifact, raw_df, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(small_artifact, path)
    loaded = predict.load_artifact(path)
    a = predict.predict_frame(small_artifact, raw_df.head(20))["churn_probability"]
    b = predict.predict_frame(loaded, raw_df.head(20))["churn_probability"]
    pd.testing.assert_series_equal(a, b)


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="trained model not present")
def test_committed_model_is_usable(raw_df):
    artifact = predict.load_artifact()
    out = predict.predict_frame(artifact, raw_df.head(100))
    assert out["churn_probability"].between(0, 1).all()
    assert 0.05 <= artifact["threshold"] <= 0.95


def test_batch_csv_roundtrip(small_artifact, raw_df):
    """Mimics the batch page: CSV text in, scored CSV text out."""
    import io

    upload = io.StringIO(raw_df.drop(columns=["Churn"]).head(30).to_csv(index=False))
    scored = predict.predict_frame(small_artifact, pd.read_csv(upload))
    exported = pd.read_csv(io.StringIO(scored.to_csv(index=False)))
    assert len(exported) == 30
    assert {"churn_probability", "predicted_churn", "risk_band"} <= set(exported.columns)


def test_batch_csv_with_wrong_columns_is_rejected(small_artifact):
    import io

    bad = pd.read_csv(io.StringIO("a,b\n1,2\n"))
    with pytest.raises(data.DataValidationError, match="Missing required column"):
        predict.predict_frame(small_artifact, bad)
