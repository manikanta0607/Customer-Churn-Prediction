# Project Overview and Charter

## Objective
Predict which subscription customers are likely to churn and give a retention team a ranked, explainable list so they can intervene early.

## Problem statement
Acquiring a new customer typically costs more than retaining an existing one, yet retention budgets are limited. Contacting every customer wastes money; contacting nobody loses revenue. The project estimates each customer's churn probability so effort goes where it is most likely to pay off.

## Target users
- **Retention / marketing manager:** needs a ranked list of at-risk customers and the main reasons behind each score.
- **Analyst:** needs to inspect data patterns and model quality.
- **Reviewer / recruiter:** needs a project that is easy to run, read and verify.

## Proposed solution
A reproducible ML pipeline (clean, engineer features, compare models, choose a cost-based threshold) exposed through a Streamlit dashboard with single-customer and batch scoring.

## Scope
**In scope:** data cleaning, EDA, feature engineering, comparison of three classifiers, evaluation, explanation, Streamlit dashboard, tests, CI, Docker files, documentation.

**Out of scope:** user authentication, a production database, a REST API, real-time data feeds, causal analysis. The data is one static table, so these would add complexity without value.

## Technology choices
| Need | Choice | Why |
|---|---|---|
| Data handling | pandas, NumPy | Standard for tabular data |
| Modelling | scikit-learn | Pipelines keep training and inference identical |
| Charts | Matplotlib (reports), Plotly (dashboard) | Static figures for docs, interactive for the app |
| UI | Streamlit | Fast, Python-only dashboards |
| Quality | pytest, Ruff, GitHub Actions | Standard, lightweight tooling |

## Milestones (all completed)
1. Repository skeleton, environment, data script
2. Cleaning and EDA
3. Training pipeline, comparison and threshold selection
4. Evaluation and explanation
5. Streamlit dashboard
6. Tests, CI, Docker files
7. Documentation
8. Publishing (performed by the project owner)

## Risks and how they were handled
| Risk | Handling |
|---|---|
| Small, clean dataset invites inflated claims | Reported held-out metrics only; stated uncertainty and limitations |
| Class imbalance (26.5% churn) | Stratified splits; metrics beyond accuracy; explicit threshold |
| Tied model scores | One-standard-deviation rule, documented |
| Docker could not be built in the dev environment | Marked untested in all docs |
| Explanation artefacts from correlated features | Grouped feature substitution |
