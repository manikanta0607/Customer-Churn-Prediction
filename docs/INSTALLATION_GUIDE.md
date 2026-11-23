# Installation Guide

## Requirements
- Python 3.12 (developed and tested on 3.12; 3.11 may work but is untested)
- git
- About 200 MB of disk space for dependencies
- No API keys, accounts or environment variables are needed

## 1. Get the code
```bash
git clone <your-repo-url> churn-prediction
cd churn-prediction
```

## 2. Create a virtual environment
macOS / Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```bash
pip install -r requirements.txt          # run the app
pip install -r requirements-dev.txt      # also tests and linting
```
Versions are pinned. The saved model was trained with scikit-learn 1.9.1; installing a different version may trigger a warning or an error when loading it. In that case retrain (step 5).

## 4. Run the dashboard
```bash
streamlit run app/streamlit_app.py
```
Open the URL shown in the terminal (normally http://localhost:8501).

## 5. Reproduce the analysis (optional)
```bash
python scripts/download_data.py   # skips if data/raw already has the file
python scripts/eda.py             # regenerates figures and docs/reports/EDA_REPORT.md
python scripts/train.py           # retrains, rewrites models/ and reports/ and plots
```
Training takes about a minute on a laptop.

## 6. Run the tests and linting
```bash
pytest
ruff check .
```

## Docker (untested in the development environment)
```bash
docker compose up --build
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Model not found" in the app | Run `python scripts/train.py` |
| "Dataset not found" | Run `python scripts/download_data.py` |
| Warning about scikit-learn version in the sidebar | Run `python scripts/train.py` to retrain with your installed version |
| `ModuleNotFoundError: churn_prediction` when running pytest | Run pytest from the repository root (configuration adds `src/` to the path) |
| Port already in use | `streamlit run app/streamlit_app.py --server.port 8600` |
