import joblib
import pandas as pd
import streamlit as st
import json
import os
import urllib.error
import urllib.parse
import urllib.request


MODEL_PATH = "final_model.joblib"
SCALER_PATH = "scaler_for_api.joblib"
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


def get_default_feature_order():
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError as exc:
        st.error(
            f"Missing file: {exc}. Keep {MODEL_PATH} and {SCALER_PATH} in the project root."
        )
        st.stop()
    except ModuleNotFoundError as exc:
        st.error(
            f"Dependency missing while loading model: {exc}. Install required packages (including xgboost)."
        )
        st.stop()


def get_expected_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return get_default_feature_order()


def preprocess_input(input_df, expected_columns, scaler):
    processed_df = input_df.copy()

    for column in expected_columns:
        if column not in processed_df.columns:
            processed_df[column] = 0.0

    processed_df = processed_df[expected_columns]
    processed_df = processed_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if "Time" in processed_df.columns and "Amount" in processed_df.columns:
        processed_df[["Time", "Amount"]] = scaler.transform(processed_df[["Time", "Amount"]])

    return processed_df


def fetch_json(url, timeout=5):
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def fetch_monitoring_summary(api_base_url, window_minutes, alert_threshold):
    query = urllib.parse.urlencode(
        {
            "window_minutes": window_minutes,
            "alert_threshold": alert_threshold,
        }
    )
    url = f"{api_base_url.rstrip('/')}/monitoring/summary?{query}"
    return fetch_json(url)


def fetch_monitoring_events(api_base_url, limit, alert_threshold):
    query = urllib.parse.urlencode(
        {
            "limit": limit,
            "alert_threshold": alert_threshold,
        }
    )
    url = f"{api_base_url.rstrip('/')}/monitoring/events?{query}"
    return fetch_json(url)


model, scaler = load_model_and_scaler()
expected_columns = get_expected_columns(model)

st.title("Credit Card Fraud Detection")
st.caption("Model: final_model.joblib (Tuned XGBoost) | Preprocessing: scale Time and Amount")

st.sidebar.header("Transaction Input")
st.sidebar.header("Monitoring Settings")
api_base_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE_URL)
window_minutes = st.sidebar.slider("Summary Window (minutes)", min_value=1, max_value=120, value=15)
event_limit = st.sidebar.slider("Recent Events", min_value=10, max_value=200, value=50)
alert_threshold = st.sidebar.slider(
    "Alert Threshold (Fraud Probability)", min_value=0.10, max_value=0.99, value=0.80, step=0.01
)
auto_refresh = st.sidebar.checkbox("Auto-refresh dashboard", value=True)
refresh_seconds = st.sidebar.slider("Refresh every (seconds)", min_value=2, max_value=60, value=5)

if auto_refresh:
    st.markdown(
        f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
        unsafe_allow_html=True,
    )

time_value = st.sidebar.number_input(
    "Time (seconds since first transaction)", min_value=0.0, value=10000.0, step=1.0
)
amount_value = st.sidebar.number_input("Amount", min_value=0.0, value=50.0, step=0.01)

v_inputs = {}
for i in range(1, 29):
    v_inputs[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0, step=0.01, format="%.6f")

raw_input = {"Time": time_value, **v_inputs, "Amount": amount_value}
input_df = pd.DataFrame([raw_input])

if st.button("Predict", type="primary"):
    try:
        model_input = preprocess_input(input_df, expected_columns, scaler)
        prediction = int(model.predict(model_input)[0])

        fraud_probability = None
        if hasattr(model, "predict_proba"):
            fraud_probability = float(model.predict_proba(model_input)[:, 1][0])

        st.subheader("Prediction")
        if prediction == 1:
            st.error("Fraudulent transaction detected")
        else:
            st.success("Legitimate transaction")

        if fraud_probability is not None:
            st.metric("Fraud Probability", f"{fraud_probability:.4f}")

        with st.expander("Processed features sent to model"):
            st.dataframe(model_input)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("Model details"):
    st.write(f"Expected feature count: {len(expected_columns)}")
    st.write(expected_columns)


st.divider()
st.header("Real-Time Monitoring and Alerting")

refresh_clicked = st.button("Refresh Monitoring", type="secondary")
if refresh_clicked:
    st.rerun()

try:
    summary = fetch_monitoring_summary(api_base_url, window_minutes, alert_threshold)
    events_payload = fetch_monitoring_events(api_base_url, event_limit, alert_threshold)

    summary_data = summary if isinstance(summary, dict) else {}
    events_data = events_payload if isinstance(events_payload, dict) else {}
    events = events_data.get("events", []) if isinstance(events_data, dict) else []

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Requests", int(summary_data.get("total_requests", 0)))
    metric_col_2.metric("Fraud Flags", int(summary_data.get("fraud_count", 0)))
    metric_col_3.metric("Alerts", int(summary_data.get("alert_count", 0)))
    metric_col_4.metric("Errors", int(summary_data.get("error_count", 0)))

    perf_col_1, perf_col_2 = st.columns(2)
    perf_col_1.metric("Avg Fraud Probability", f"{float(summary_data.get('avg_fraud_probability', 0.0)):.4f}")
    perf_col_2.metric("Avg Latency (ms)", f"{float(summary_data.get('avg_latency_ms', 0.0)):.2f}")

    if events:
        events_df = pd.DataFrame(events)
        if "is_alert" in events_df.columns:
            alert_df = events_df[events_df["is_alert"] == True]
        else:
            alert_df = pd.DataFrame()

        st.subheader("Recent Alerts")
        if not alert_df.empty:
            alert_view = alert_df[[
                "event_time",
                "prediction",
                "fraud_probability",
                "amount",
                "latency_ms",
                "status",
            ]].copy()
            st.dataframe(alert_view, use_container_width=True)
        else:
            st.success("No active alerts in the selected window.")

        st.subheader("Recent Events")
        display_columns = [
            "event_time",
            "prediction",
            "fraud_probability",
            "amount",
            "latency_ms",
            "status",
            "error_message",
            "is_alert",
        ]
        available_columns = [col for col in display_columns if col in events_df.columns]
        st.dataframe(events_df[available_columns], use_container_width=True)
    else:
        st.info("No monitoring events yet. Trigger predictions via API or this app to populate the dashboard.")

except urllib.error.URLError as exc:
    st.warning(
        f"Unable to reach monitoring API at {api_base_url}. Start app.py to enable live dashboard. Details: {exc.reason}"
    )
except Exception as exc:
    st.error(f"Failed to load monitoring dashboard: {exc}")

    
