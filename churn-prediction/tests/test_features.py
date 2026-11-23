import pandas as pd

from churn_prediction import config, features


def test_add_features_columns_and_values(clean_df):
    out = features.add_features(clean_df)
    for col in ["avg_monthly_spend", "n_addon_services", "tenure_group"]:
        assert col in out.columns
    assert out["avg_monthly_spend"].notna().all()
    assert out["n_addon_services"].between(0, 6).all()
    assert out["tenure_group"].notna().all()


def test_tenure_zero_does_not_divide_by_zero():
    row = pd.DataFrame([{**{c: "No" for c in config.ADD_ON_SERVICES}, "tenure": 0, "TotalCharges": 0.0}])
    out = features.add_features(row)
    assert out.loc[0, "avg_monthly_spend"] == 0
    assert out.loc[0, "tenure_group"] == "0-12m"


def test_addon_count_counts_yes_only():
    row = {c: "No" for c in config.ADD_ON_SERVICES}
    row["TechSupport"] = row["StreamingTV"] = "Yes"
    row.update(tenure=10, TotalCharges=100.0)
    assert features.add_features(pd.DataFrame([row])).loc[0, "n_addon_services"] == 2


def test_tenure_group_boundaries():
    base = {c: "No" for c in config.ADD_ON_SERVICES}
    frame = pd.DataFrame([{**base, "tenure": t, "TotalCharges": 1.0} for t in [12, 13, 24, 25, 48, 49]])
    assert features.add_features(frame)["tenure_group"].tolist() == ["0-12m", "13-24m", "13-24m", "25-48m", "25-48m", "49m+"]


def test_preprocessor_output_is_numeric_and_handles_unknown(clean_df):
    x = clean_df[config.FEATURE_COLUMNS]
    pre_in = features.add_features(x)
    pre = features.build_preprocessor().fit(pre_in)
    out = pre.transform(pre_in.head(5))
    assert out.shape[0] == 5 and out.dtype.kind == "f"
    odd = pre_in.head(1).copy()
    odd["Contract"] = "Brand new contract"
    assert pre.transform(odd).shape == (1, out.shape[1])
