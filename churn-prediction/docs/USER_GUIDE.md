# User Guide

Start the dashboard with `streamlit run app/streamlit_app.py`. Use the sidebar to switch pages and to change the **decision threshold**.

## Decision threshold
Customers whose churn probability is at or above the threshold are flagged for action. The default (0.35) comes from the training run. Lower it to contact more customers and catch more churners (more false alarms); raise it to contact fewer, more certain cases.

## Risk bands
- **High:** probability at or above the threshold
- **Medium:** at or above half the threshold
- **Low:** below that

## Overview
Choose contract types and internet services to filter. You see customer count, churn rate, average tenure and average monthly charge, plus charts of churn by contract, tenure group and payment method, and monthly charges for churned versus retained customers. If the filters match nobody, a warning is shown.

## Predict one customer
1. Fill in the account, services and demographic details.
2. Press **Predict churn risk**.
3. Read the churn probability, risk band and recommendation.
4. The "What drives this score?" chart shows which groups of features raise or lower the risk compared with a typical customer, in percentage points. Hover or read the table for the actual values. This is an approximation, and the bars do not add up to the total.

Notes: add-on services are disabled when the customer has no internet service, and total charges are calculated as tenure times monthly charges.

## Batch scoring
1. Optionally download the sample CSV to see the expected format.
2. Upload a CSV containing the same columns as the IBM Telco dataset (a `Churn` column is ignored if present).
3. Review the counts of High, Medium and Low risk customers and the ranked table.
4. Press **Download scored results** to save the full output.

If a required column is missing or the file is not valid CSV, an error message explains the problem. Required columns are listed in the error.

## Model performance
Shows the selected model, test-set metrics, the cross-validated comparison of the three algorithms, the assumptions behind the threshold, and the evaluation charts. The retention economics there are illustrative.

## Taking screenshots
Screenshots were not captured during development. To add them for your README:
1. Run the dashboard and open each page.
2. Save screenshots as `docs/images/dashboard_overview.png`, `dashboard_predict.png`, `dashboard_batch.png` and `dashboard_performance.png`.
3. Add them to the README, for example `![Overview](docs/images/dashboard_overview.png)`.
