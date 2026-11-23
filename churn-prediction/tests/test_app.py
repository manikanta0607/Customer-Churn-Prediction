import pytest
from streamlit.testing.v1 import AppTest

from churn_prediction import config

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="trained model not present")
APP = str(config.ROOT / "app" / "streamlit_app.py")


def _app():
    return AppTest.from_file(APP, default_timeout=90).run()


def test_app_starts_without_errors():
    at = _app()
    assert not at.exception and not at.error


@pytest.mark.parametrize("page", ["Overview", "Predict one customer", "Batch scoring", "Model performance"])
def test_every_page_renders(page):
    at = _app()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error


def test_prediction_form_returns_probability():
    at = _app()
    at.sidebar.radio[0].set_value("Predict one customer").run()
    at.button[0].click().run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert "Churn probability" in labels and labels["Churn probability"].endswith("%")


def test_high_risk_profile_scores_higher_than_low_risk():
    def score(contract, tenure):
        at = _app()
        at.sidebar.radio[0].set_value("Predict one customer").run()
        for sb in at.selectbox:
            if sb.label == "Contract":
                sb.set_value(contract)
        for sl in at.slider:
            if sl.label == "Tenure (months)":
                sl.set_value(tenure)
        at.button[0].click().run()
        return float({m.label: m.value for m in at.metric}["Churn probability"].rstrip("%"))

    assert score("Month-to-month", 1) > score("Two year", 60)
