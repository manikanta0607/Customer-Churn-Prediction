# Low-Level Design

## `config.py`
- `FEATURE_COLUMNS` = numeric (`tenure`, `MonthlyCharges`, `TotalCharges`) + binary (`SeniorCitizen`) + 15 categorical columns. This is the required input schema for prediction.
- `CUSTOMER_VALUE = 500`, `OFFER_COST = 50`, `OFFER_SUCCESS_RATE = 0.30`: illustrative economics for threshold selection only.
- `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`, `CV_FOLDS = 5`.

## `data.py`
| Function | Behaviour |
|---|---|
| `load_raw(path)` | Reads the CSV; raises `FileNotFoundError` with a hint to run the download script |
| `validate_schema(df, require_target)` | Raises `DataValidationError` for missing columns or an empty frame |
| `clean(df, require_target)` | Copy, trim text, numeric conversion, blank `TotalCharges` where tenure is 0 set to 0, "No phone/internet service" merged into "No", `SeniorCitizen` to int, duplicate IDs dropped, `Churn` mapped Yes/No to 1/0. Raises on non-numeric charges |
| `split_features_target(df)` | Returns `FEATURE_COLUMNS` and the target |
| `train_test(df)` | Stratified split |

## `features.py`
`add_features` adds three columns:
- `avg_monthly_spend = TotalCharges / max(tenure, 1)`
- `n_addon_services`: count of "Yes" across six add-on services
- `tenure_group`: bins 0-12m, 13-24m, 25-48m, 49m+

`build_preprocessor` is a `ColumnTransformer`: `StandardScaler` on five numeric columns, `OneHotEncoder(handle_unknown="ignore")` on 15 categorical columns plus `tenure_group`, and pass-through for `SeniorCitizen`. Unseen categories at prediction time become all-zero rows instead of errors.

## `modeling.py`
- `build_pipeline(estimator)`: feature step, preprocessor, estimator.
- `candidate_models()`: Logistic Regression (grid over `C`), Random Forest (`max_depth`, `min_samples_leaf`), HistGradientBoosting (`learning_rate`, `max_leaf_nodes`, `l2_regularization`).
- `compare_models(x, y)`: `GridSearchCV` per candidate, 5-fold stratified, refit on ROC-AUC, returns summaries.
- `select_model(results)`: one-standard-deviation rule with simplicity order Logistic Regression, Gradient Boosting, Random Forest.
- `expected_net_benefit(y, p, t)`: per-customer average of `TP * (success * value - cost) - FP * cost`.
- `choose_threshold(y, p)`: scans 0.05 to 0.95 in steps of 0.01.
- `classification_metrics(y, p, t)`: accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score, contacted share, net benefit.

No class weighting is used; the decision threshold handles imbalance, which also keeps probabilities closer to calibrated.

## `predict.py`
- Artifact dict keys: `pipeline`, `threshold`, `reference` (median or mode per feature from training data), `metadata`.
- `predict_frame(artifact, df, threshold=None)` returns cleaned rows with `churn_probability`, `predicted_churn`, `risk_band`.
- `explain_customer(artifact, row, top_k)` swaps each factor group to typical values and reports the probability change in percentage points. Groups: tenure and total charges; internet service, add-ons and monthly price; phone services; contract; payment and billing; demographics. Groups where the customer already matches typical values are skipped.

## `reports/metrics.json`
Timestamp, library versions, dataset sizes, selected model and parameters, comparison table, threshold, economic assumptions, test metrics at the selected threshold and at 0.5, majority-class baseline accuracy, top permutation importances.

## Dashboard (`app/streamlit_app.py`)
- `st.cache_resource` for the model and `st.cache_data` for the dataset and metrics.
- Missing model: error message and `st.stop()`.
- Overview filter with no matches: warning instead of empty charts.
- Add-on selectors are disabled when "Internet service" is "No", and forced to "No" on submit.
- `TotalCharges` in the form is computed as tenure times monthly charges.
- Batch page: CSV parse errors and schema errors shown as messages; results sorted by probability and downloadable.
- Sidebar warns if the installed scikit-learn version differs from the one that trained the model.
