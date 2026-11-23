# Testing Documentation

All results below come from commands actually run during development on Linux with Python 3.12.3, pytest 9.2.0, scikit-learn 1.9.1.

## Summary
| Check | Result |
|---|---|
| `pytest` | **43 passed** in about 5 seconds |
| `ruff check .` (includes security rules) | All checks passed |
| `ruff format --check .` | Clean |
| Secret scan (grep for keys, tokens, passwords) | Nothing found except the `.gitignore` entry for Streamlit secrets |
| `pip-audit -r requirements.txt` | No known vulnerabilities at time of check |
| Training reproducibility | Two separate runs produced identical metrics |
| Real server launch | Streamlit started, health endpoint returned `ok`, root page returned HTTP 200 |

## Test inventory
| File | Tests | Covers |
|---|---|---|
| `test_data.py` | 12 | Schema validation, missing file, blank `TotalCharges`, label merging, target encoding, duplicate removal, no mutation of input, non-numeric charges, stratified disjoint split |
| `test_features.py` | 5 | Engineered columns, tenure 0, add-on counting, tenure group boundaries, preprocessor output and unseen categories |
| `test_modeling.py` | 6 | Net benefit arithmetic, threshold choice, metrics, one-standard-deviation selection, pipeline training (ROC-AUC above 0.80), candidate list |
| `test_predict.py` | 13 | Risk bands, output columns and ranges, threshold monotonicity, input without target, bad input, unseen category, explanations, model-missing error, artifact round trip, committed model, batch CSV round trip and rejection |
| `test_app.py` | 7 | App starts, each of the four pages renders without exception or error, prediction form returns a probability, high-risk profile scores above low-risk profile |

## Test types
- **Unit:** data, features, modeling, predict functions.
- **Integration:** full pipeline training and prediction, artifact save/load, CSV round trip.
- **Frontend:** Streamlit's `AppTest` harness simulates the app, including the form submission.
- **Model evaluation:** held-out test-set metrics in `reports/metrics.json` (see README).
- **Input validation and error handling:** missing columns, empty data, malformed values, unseen categories, missing model or dataset.
- **Security:** secret scan, Ruff security rules, dependency audit.

## Do the tests detect real faults?
As a spot check, two bugs were introduced deliberately (removing the blank `TotalCharges` handling and altering the risk-band comparison). The suite failed, and passed again after the code was restored. This was a manual check on two mutations, not systematic mutation testing.

## Not verified
- **Browser end-to-end tests:** a real browser could not be installed in the development environment. `AppTest` simulates the app but does not render it, and the file-upload widget itself is not driven by tests (the CSV logic behind it is).
- **Docker:** image build and container run were not possible (base images could not be pulled).
- **GitHub Actions:** the workflow has not run on GitHub yet; check the first run after pushing.
- **Other operating systems and Python versions:** only Linux and Python 3.12.
- **Load, accessibility and user-experience testing.**
- **Deployment:** the application has not been deployed.

## Running the tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
