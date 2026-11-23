import pandas as pd
import pytest

from churn_prediction import config, data


def test_load_raw_shape(raw_df):
    assert raw_df.shape == (7043, 21)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        data.load_raw(tmp_path / "nope.csv")


def test_validate_schema_reports_missing_columns(raw_df):
    with pytest.raises(data.DataValidationError, match="Contract"):
        data.validate_schema(raw_df.drop(columns=["Contract"]))


def test_validate_schema_rejects_empty(raw_df):
    with pytest.raises(data.DataValidationError, match="no rows"):
        data.validate_schema(raw_df.iloc[0:0])


def test_validate_requires_target_when_asked(raw_df):
    with pytest.raises(data.DataValidationError, match="Churn"):
        data.validate_schema(raw_df.drop(columns=["Churn"]), require_target=True)


def test_clean_fixes_total_charges(raw_df, clean_df):
    assert pd.api.types.is_float_dtype(clean_df["TotalCharges"])
    assert clean_df["TotalCharges"].isna().sum() == 0
    blanks = raw_df["TotalCharges"].astype("string").str.strip() == ""
    assert blanks.sum() == 11
    assert (clean_df.loc[blanks.values, "TotalCharges"] == 0).all()
    assert (clean_df.loc[blanks.values, "tenure"] == 0).all()


def test_clean_encodes_target_and_merges_labels(clean_df):
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert 0.25 < clean_df[config.TARGET].mean() < 0.28
    assert "No internet service" not in set(clean_df["OnlineSecurity"])
    assert "No phone service" not in set(clean_df["MultipleLines"])


def test_clean_drops_duplicate_ids(raw_df):
    doubled = pd.concat([raw_df.head(10), raw_df.head(10)])
    assert len(data.clean(doubled, require_target=True)) == 10


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    data.clean(raw_df, require_target=True)
    pd.testing.assert_frame_equal(before, raw_df)


def test_clean_without_target_column(raw_df):
    out = data.clean(raw_df.drop(columns=["Churn"]).head(20))
    assert config.TARGET not in out.columns


def test_clean_rejects_non_numeric_charges(raw_df):
    bad = raw_df.head(5).copy()
    bad["MonthlyCharges"] = ["abc", "1", "2", "3", "4"]
    with pytest.raises(data.DataValidationError, match="non-numeric"):
        data.clean(bad)


def test_split_is_stratified_and_disjoint(clean_df):
    x_tr, x_te, y_tr, y_te = data.train_test(clean_df)
    assert len(x_tr) + len(x_te) == len(clean_df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert set(x_tr.index).isdisjoint(x_te.index)
    assert list(x_tr.columns) == config.FEATURE_COLUMNS
