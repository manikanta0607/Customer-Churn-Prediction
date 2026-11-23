"""Streamlit dashboard for the customer churn prediction project."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import sklearn  # noqa: E402
import streamlit as st  # noqa: E402

from churn_prediction import config, data, features, predict  # noqa: E402

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

RISK_COLORS = {"High": "#D64545", "Medium": "#E0A100", "Low": "#2E8B57"}


@st.cache_resource(show_spinner="Loading model...")
def get_artifact() -> dict:
    return predict.load_artifact()


@st.cache_data(show_spinner="Loading dataset...")
def get_dataset() -> pd.DataFrame:
    return data.clean(data.load_raw(), require_target=True)


@st.cache_data
def get_metrics() -> dict | None:
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    return None


def load_or_stop() -> dict:
    try:
        return get_artifact()
    except predict.ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()


# ----------------------------------------------------------------------- pages
def page_overview() -> None:
    st.header("Customer base overview")
    try:
        df = get_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    contracts = left.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
    internet = right.multiselect("Internet service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique()))
    view = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet)]
    if view.empty:
        st.warning("No customers match the selected filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(view):,}")
    c2.metric("Churn rate", f"{view[config.TARGET].mean():.1%}")
    c3.metric("Avg tenure (months)", f"{view['tenure'].mean():.1f}")
    c4.metric("Avg monthly charge", f"${view['MonthlyCharges'].mean():.2f}")

    enriched = features.add_features(view)
    a, b = st.columns(2)
    for col, container, title in [("Contract", a, "Churn rate by contract"), ("tenure_group", b, "Churn rate by tenure group")]:
        rate = enriched.groupby(col, observed=True)[config.TARGET].mean().mul(100).reset_index()
        fig = px.bar(rate, x=col, y=config.TARGET, title=title, labels={config.TARGET: "Churn rate (%)"})
        fig.update_traces(marker_color="#E4572E")
        container.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    rate = view.groupby("PaymentMethod")[config.TARGET].mean().mul(100).reset_index()
    fig = px.bar(rate, x="PaymentMethod", y=config.TARGET, title="Churn rate by payment method", labels={config.TARGET: "Churn rate (%)"})
    fig.update_traces(marker_color="#4C78A8")
    a.plotly_chart(fig, width="stretch")
    hist = view.assign(Status=view[config.TARGET].map({0: "Stayed", 1: "Churned"}))
    fig = px.histogram(hist, x="MonthlyCharges", color="Status", barmode="overlay", opacity=0.65, title="Monthly charges by outcome")
    b.plotly_chart(fig, width="stretch")
    st.caption("Source: IBM Telco Customer Churn sample dataset. Patterns are correlations, not proof of cause.")


def customer_form() -> pd.DataFrame | None:
    yn = ["No", "Yes"]
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.selectbox("Paperless billing", yn, index=1)
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
        with c2:
            st.subheader("Services")
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            phone = st.selectbox("Phone service", yn, index=1)
            lines = st.selectbox("Multiple lines", yn)
            st.caption("Add-ons only apply when the customer has internet service.")
            addons = {a: st.selectbox(a, yn, disabled=internet == "No") for a in config.ADD_ON_SERVICES}
        with c3:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", yn)
            partner = st.selectbox("Partner", yn)
            dependents = st.selectbox("Dependents", yn)
        submitted = st.form_submit_button("Predict churn risk", type="primary")
    if not submitted:
        return None
    if internet == "No":
        addons = {a: "No" for a in config.ADD_ON_SERVICES}
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": round(tenure * monthly, 2),
    }
    return pd.DataFrame([row])


def page_predict(artifact: dict, threshold: float) -> None:
    st.header("Predict churn for one customer")
    row = customer_form()
    if row is None:
        st.info("Fill in the customer details and press **Predict churn risk**.")
        return
    try:
        with st.spinner("Scoring..."):
            scored = predict.predict_frame(artifact, row, threshold)
            drivers = predict.explain_customer(artifact, row)
    except data.DataValidationError as exc:
        st.error(f"Invalid input: {exc}")
        return
    prob = float(scored.loc[0, "churn_probability"])
    band = scored.loc[0, "risk_band"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{prob:.1%}")
    c2.markdown(f"**Risk band**<br><span style='font-size:2rem;color:{RISK_COLORS[band]}'>{band}</span>", unsafe_allow_html=True)
    c3.metric("Decision threshold", f"{threshold:.2f}")
    if band == "High":
        st.error("This customer is above the threshold: a retention action is recommended.")
    elif band == "Medium":
        st.warning("Elevated risk, below the action threshold. Worth monitoring.")
    else:
        st.success("Low churn risk.")
    st.subheader("What drives this score?")
    if drivers.empty:
        st.write("This customer looks like a typical customer on every feature.")
    else:
        drivers = drivers.assign(direction=drivers["effect_pp"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk"))
        fig = px.bar(
            drivers.iloc[::-1],
            x="effect_pp",
            y="factor",
            orientation="h",
            color="direction",
            hover_data=["details"],
            color_discrete_map={"Raises risk": "#D64545", "Lowers risk": "#2E8B57"},
            labels={"effect_pp": "Effect vs typical customer (percentage points)", "factor": ""},
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(drivers[["factor", "effect_pp", "details"]], width="stretch", hide_index=True)
        st.caption(
            "Approximation: each group of related features is swapped for typical values and the change in probability is measured. Effects do not add up to the total score."
        )


def page_batch(artifact: dict, threshold: float) -> None:
    st.header("Score a customer file")
    st.write("Upload a CSV with the same columns as the IBM Telco dataset (`Churn` is optional and ignored).")
    sample = data.load_raw().head(5) if config.DATA_PATH.exists() else None
    if sample is not None:
        st.download_button("Download sample CSV", sample.drop(columns=[config.TARGET]).to_csv(index=False), "sample_customers.csv", "text/csv")
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is None:
        st.info("Upload a CSV file to begin.")
        return
    try:
        raw = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the file as CSV: {exc}")
        return
    try:
        with st.spinner("Scoring customers..."):
            scored = predict.predict_frame(artifact, raw, threshold)
    except data.DataValidationError as exc:
        st.error(f"Invalid file: {exc}")
        st.caption("Required columns: " + ", ".join(config.FEATURE_COLUMNS))
        return
    counts = scored["risk_band"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{len(scored):,}")
    c2.metric("High risk", int(counts["High"]))
    c3.metric("Medium risk", int(counts["Medium"]))
    c4.metric("Low risk", int(counts["Low"]))
    cols = [c for c in [config.ID_COLUMN, "churn_probability", "predicted_churn", "risk_band", "Contract", "tenure", "MonthlyCharges"] if c in scored.columns]
    ranked = scored.sort_values("churn_probability", ascending=False)
    st.dataframe(ranked[cols], width="stretch", hide_index=True)
    buf = io.StringIO()
    ranked.to_csv(buf, index=False)
    st.download_button("Download scored results", buf.getvalue(), "scored_customers.csv", "text/csv", type="primary")


def page_performance(artifact: dict) -> None:
    st.header("Model performance")
    metrics = get_metrics()
    if metrics is None:
        st.warning("No metrics file found. Run `python scripts/train.py`.")
        return
    t = metrics["test_metrics_at_selected_threshold"]
    st.write(f"**Selected model:** {metrics['selected_model']} &nbsp;|&nbsp; **Trained:** {metrics['trained_at_utc']}")
    cols = st.columns(5)
    for col, (label, key) in zip(
        cols, [("ROC-AUC", "roc_auc"), ("Recall", "recall"), ("Precision", "precision"), ("F1", "f1"), ("Accuracy", "accuracy")], strict=True
    ):
        col.metric(label, f"{t[key]:.3f}")
    st.caption(
        f"Held-out test set ({metrics['dataset']['test_rows']:,} customers) at threshold {t['threshold']:.2f}. "
        f"A model that always predicts 'stays' would score {metrics['baseline_majority_class_accuracy']:.1%} accuracy, "
        "so accuracy alone is not a useful yardstick here."
    )
    st.subheader("Model comparison (5-fold cross-validation on training data)")
    comp = pd.DataFrame(metrics["model_comparison"])[["model", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_average_precision_mean", "cv_f1_at_0.5_mean"]]
    st.dataframe(comp.round(4), width="stretch", hide_index=True)
    st.caption("Scores are statistically close, so the simplest model within one standard deviation of the best was selected.")
    a = metrics["economic_assumptions"]
    st.info(
        f"Threshold chosen by expected net benefit using **illustrative** assumptions: customer value ${a['customer_value']:.0f}, "
        f"offer cost ${a['offer_cost']:.0f}, offer success rate {a['offer_success_rate']:.0%}. Adjust these in `config.py` for a real business."
    )
    images = [
        ("eval_roc_curve.png", "ROC curve"),
        ("eval_pr_curve.png", "Precision-recall"),
        ("eval_confusion_matrix.png", "Confusion matrix"),
        ("eval_threshold_curve.png", "Threshold selection"),
        ("eval_feature_importance.png", "Feature importance"),
    ]
    cols = st.columns(2)
    for i, (name, cap) in enumerate(images):
        path = config.IMAGES_DIR / name
        if path.exists():
            cols[i % 2].image(str(path), caption=cap, width="stretch")


# ------------------------------------------------------------------------ main
def main() -> None:
    st.sidebar.title("📉 Churn Prediction")
    page = st.sidebar.radio("Page", ["Overview", "Predict one customer", "Batch scoring", "Model performance"])
    artifact = load_or_stop()
    saved = float(artifact["threshold"])
    threshold = st.sidebar.slider(
        "Decision threshold", 0.05, 0.95, saved, 0.01, help="Customers at or above this probability are flagged for retention action."
    )
    st.sidebar.caption(f"Saved model threshold: {saved:.2f}")
    trained_with = artifact["metadata"].get("scikit_learn_version")
    if trained_with and trained_with != sklearn.__version__:
        st.sidebar.warning(
            f"Model trained with scikit-learn {trained_with}; you have {sklearn.__version__}. Retrain with `python scripts/train.py` if you see errors."
        )
    if page == "Overview":
        page_overview()
    elif page == "Predict one customer":
        page_predict(artifact, threshold)
    elif page == "Batch scoring":
        page_batch(artifact, threshold)
    else:
        page_performance(artifact)


main()
