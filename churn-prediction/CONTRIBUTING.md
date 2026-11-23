# Contributing

Thanks for your interest in improving this project.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Workflow
1. Create a branch from `main`.
2. Make a focused change with tests.
3. Run the checks:
   ```bash
   ruff format . && ruff check .
   pytest
   ```
4. If you change modelling code, run `python scripts/train.py` and commit the updated `models/` and `reports/` files, and update any numbers quoted in the README and docs.
5. Open a pull request describing what changed and why.

## Guidelines
- Keep feature engineering inside the scikit-learn pipeline so training and inference stay identical.
- Do not tune hyper-parameters or the threshold on the test set.
- Never commit secrets, and never document features that do not exist.
- Report metrics exactly as produced by the scripts.

## Commit messages
Short imperative summary, for example `Add probability calibration step`.
