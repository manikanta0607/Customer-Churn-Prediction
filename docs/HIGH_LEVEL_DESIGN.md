# High-Level Design

## Modules and responsibilities
| Module | Responsibility |
|---|---|
| `config` | Paths, column groups, random seed, split sizes, illustrative business economics |
| `data` | Load, validate, clean, split |
| `features` | Engineered features and the preprocessing transformer |
| `modeling` | Pipeline construction, candidate models and grids, cross-validated comparison, model selection, threshold selection, metrics |
| `evaluate` | ROC, precision-recall, confusion matrix, threshold curve, permutation importance |
| `predict` | Load artifact, score a table, risk bands, per-customer explanations |
| `scripts/train.py` | Orchestrates training and writes the artifact, metrics and plots |
| `scripts/eda.py` | Generates EDA figures and the data-driven report |
| `app/streamlit_app.py` | Dashboard |

## Training workflow
1. Load and clean the data (`data`).
2. Stratified 80/20 split (random state 42).
3. For each candidate, grid search with 5-fold stratified CV, scoring ROC-AUC, average precision and F1.
4. Select the simplest model within one standard deviation of the best CV ROC-AUC.
5. Obtain out-of-fold probabilities for the selected pipeline on the training set and choose the threshold with the highest expected net benefit.
6. Evaluate once on the test set, at the chosen threshold and at 0.5.
7. Save the artifact (`models/churn_model.joblib`), metrics (`reports/metrics.json`) and plots.

## Inference workflow
Uploaded or entered data is cleaned, passed through the saved pipeline, and returned with a probability, a flag (probability at or above the threshold) and a risk band. For single customers the app also computes grouped what-if explanations.

## Dashboard pages
| Page | Purpose |
|---|---|
| Overview | Filterable KPIs and charts on the whole customer base |
| Predict one customer | Form, probability, risk band, drivers |
| Batch scoring | CSV in, ranked scores out, download |
| Model performance | Metrics, comparison, plots, threshold economics |

## Risk bands
High: probability at or above the threshold. Medium: at or above half the threshold. Low: below that.

## Why no database or API?
The only data is a 7,043-row static CSV, and the dashboard is the only consumer. A database or REST layer would add moving parts without solving a problem here. If the project grew to need multiple clients, a FastAPI service wrapping `predict.py` would be the natural next step.
