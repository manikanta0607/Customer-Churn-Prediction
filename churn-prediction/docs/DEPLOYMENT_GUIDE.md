# Deployment Guide

**Status: not deployed.** The options below are prepared but have not been run, so verify each yourself. No secrets or environment variables are required.

## Option 1: Streamlit Community Cloud (simplest, free)
1. Push the repository to GitHub (public, or private with access granted).
2. Sign in at https://share.streamlit.io with GitHub and choose **Create app**.
3. Select the repository and branch `main`, and set the main file to `app/streamlit_app.py`.
4. Set the Python version to 3.12 in the advanced settings so it matches `requirements.txt`.
5. Deploy. The repository already contains the dataset, model and metrics the app needs.
6. After deploying, open every page and test a prediction. Add the live URL to the README.

## Option 2: Docker
```bash
docker compose up --build        # http://localhost:8501
```
or
```bash
docker build -t churn-dashboard .
docker run -p 8501:8501 churn-dashboard
```
The container runs as a non-root user and includes a health check. This Dockerfile has **not** been built yet. The same image can be deployed to any container host (Render, Fly.io, Google Cloud Run, AWS App Runner and similar).

## Option 3: Hugging Face Spaces
Create a Streamlit Space, upload the repository files, and set the app file to `app/streamlit_app.py`.

## Before going public
- Confirm no secrets are committed (none are used by this project).
- Check that `LICENSE` has your name.
- Keep the Apache-2.0 notice for the dataset in `data/raw/`.

## Updating the model
Run `python scripts/train.py`, commit the changed files in `models/`, `reports/` and `docs/images/`, and redeploy.
