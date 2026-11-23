# Software Requirements Specification

## 1. Purpose
Defines what the churn prediction system must do and the quality attributes it must meet. Each requirement maps to an implementation and a verification method.

## 2. Functional requirements
| ID | Requirement | Implemented in | Verified by |
|---|---|---|---|
| FR-1 | Load the customer dataset and fail with a clear message if it is missing | `data.load_raw` | `test_missing_file_gives_helpful_error` |
| FR-2 | Validate that required columns exist and data is non-empty | `data.validate_schema` | `test_validate_*` |
| FR-3 | Clean data: numeric conversion, blank `TotalCharges`, label merging, duplicate removal, target encoding | `data.clean` | `test_clean_*` |
| FR-4 | Engineer features (average monthly spend, add-on count, tenure group) | `features.add_features` | `test_features.py` |
| FR-5 | Compare Logistic Regression, Random Forest and Gradient Boosting using stratified cross-validation | `modeling.compare_models` | `train.py` run, `test_candidates_*` |
| FR-6 | Select a model using a documented rule | `modeling.select_model` | `test_select_model_*` |
| FR-7 | Choose a decision threshold by expected net benefit | `modeling.choose_threshold` | `test_choose_threshold_*`, `test_net_benefit_*` |
| FR-8 | Report ROC-AUC, precision, recall, F1, average precision, Brier score on a held-out test set | `modeling.classification_metrics`, `scripts/train.py` | `test_classification_metrics_*` |
| FR-9 | Save and load the trained model with its threshold and reference values | `scripts/train.py`, `predict.load_artifact` | `test_artifact_roundtrip` |
| FR-10 | Score a single customer and a batch, returning probability, flag and risk band | `predict.predict_frame` | `test_predict_*` |
| FR-11 | Explain the main drivers of a customer's score | `predict.explain_customer` | `test_explain_*` |
| FR-12 | Dashboard with Overview, Predict, Batch and Performance pages | `app/streamlit_app.py` | `test_app.py` |
| FR-13 | Batch results downloadable as CSV; invalid uploads rejected with a message | `page_batch` | `test_predict_rejects_bad_input`; manual check |
| FR-14 | Adjustable decision threshold | sidebar slider | `test_threshold_changes_flags_monotonically` |

## 3. Non-functional requirements
| ID | Requirement | Status |
|---|---|---|
| NFR-1 | Reproducible: fixed random seeds, pinned dependencies | Met (re-running training gives identical metrics) |
| NFR-2 | Performance: dashboard pages respond within a few seconds on a laptop; model file under 10 MB | Model is about 3 KB; page responsiveness observed in tests, not formally benchmarked |
| NFR-3 | Usability: clear empty, loading, error and success states | Implemented; verified by automated page tests, not by user testing |
| NFR-4 | Security: no secrets in the repository, no known vulnerable dependencies | Secret scan, Ruff security rules and `pip-audit` clean at time of check |
| NFR-5 | Maintainability: modular code, lint-clean, tested | Met |
| NFR-6 | Portability: runs on Python 3.12 on Linux, macOS, Windows | Tested on Linux with Python 3.12 only |
| NFR-7 | Accessibility: default Streamlit components, colour never the only signal (risk band has text) | Not audited with assistive tools |

## 4. Technical requirements
Python 3.12; packages pinned in `requirements.txt`; no external services or credentials.

## 5. Assumptions and constraints
- The dataset is a representative sample for demonstration purposes.
- Retention economics in `config.py` are illustrative.
- The model must be loaded with the scikit-learn version it was trained with.
