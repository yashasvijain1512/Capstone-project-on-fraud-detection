
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import os
import sqlite3
import time
import joblib
import pandas as pd

app = Flask(__name__)

MONITORING_DB_PATH = os.getenv("MONITORING_DB_PATH", "monitoring.db")
DEFAULT_ALERT_THRESHOLD = 0.8
MODEL_PATH = os.getenv("MODEL_PATH", "final_model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_for_api.joblib")


@app.route('/', methods=['GET'])
def index():
    return jsonify(
        {
            "service": "credit-card-fraud-detection-api",
            "status": "ok",
            "message": "API is running. Use POST /predict for predictions.",
            "endpoints": {
                "predict": {
                    "method": "POST",
                    "path": "/predict",
                },
                "monitoring_summary": {
                    "method": "GET",
                    "path": "/monitoring/summary",
                },
                "monitoring_events": {
                    "method": "GET",
                    "path": "/monitoring/events",
                },
                "health": {
                    "method": "GET",
                    "path": "/health",
                },
            },
        }
    )


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

# Load the trained model
model = joblib.load(MODEL_PATH)

# Assuming the scaler was fitted on 'Time' and 'Amount' from the training data
# In a real deployment, you would save and load the fitted scaler.
# For this example, we re-initialize and fit on dummy data to match structure.
# Ideally, this scaler object would also be serialized and loaded.

# Create a dummy scaler object for demonstration. THIS SHOULD BE YOUR *FITTED* SCALER.
# In a production setup, you would load the actual fitted scaler.
# For this demo, we will use a global scaler object that was fitted earlier in the notebook.
# In a complete standalone API, you would save and load the fitted scaler, not re-fit it.
# For the purpose of this Colab demo, we will assume `scaler_for_api` is the one from the notebook state.

# Replace this with the actual loaded scaler from your workflow
# For the context of this Colab, we'll make the scaler global to match the existing notebook
global_scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully within Flask app!")


def get_db_connection():
    connection = sqlite3.connect(MONITORING_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_monitoring_store():
    connection = get_db_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                prediction INTEGER,
                fraud_probability REAL,
                amount REAL,
                latency_ms REAL,
                status TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def log_prediction_event(
    prediction=None,
    fraud_probability=None,
    amount=None,
    latency_ms=None,
    status="success",
    error_message=None,
):
    connection = get_db_connection()
    try:
        connection.execute(
            """
            INSERT INTO prediction_events (
                event_time,
                prediction,
                fraud_probability,
                amount,
                latency_ms,
                status,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                prediction,
                fraud_probability,
                amount,
                latency_ms,
                status,
                error_message,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def get_summary(window_minutes=15, alert_threshold=DEFAULT_ALERT_THRESHOLD):
    cutoff = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat(timespec="seconds")

    connection = get_db_connection()
    try:
        total = connection.execute(
            "SELECT COUNT(*) AS count FROM prediction_events WHERE event_time >= ?",
            (cutoff,),
        ).fetchone()["count"]

        fraud_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM prediction_events
            WHERE event_time >= ? AND status = 'success' AND prediction = 1
            """,
            (cutoff,),
        ).fetchone()["count"]

        error_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM prediction_events
            WHERE event_time >= ? AND status = 'error'
            """,
            (cutoff,),
        ).fetchone()["count"]

        alert_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM prediction_events
            WHERE event_time >= ?
            AND status = 'success'
            AND (prediction = 1 OR fraud_probability >= ?)
            """,
            (cutoff, alert_threshold),
        ).fetchone()["count"]

        averages = connection.execute(
            """
            SELECT AVG(fraud_probability) AS avg_probability, AVG(latency_ms) AS avg_latency
            FROM prediction_events
            WHERE event_time >= ? AND status = 'success'
            """,
            (cutoff,),
        ).fetchone()

        avg_probability = float(averages["avg_probability"]) if averages["avg_probability"] is not None else 0.0
        avg_latency_ms = float(averages["avg_latency"]) if averages["avg_latency"] is not None else 0.0

        return {
            "window_minutes": window_minutes,
            "alert_threshold": float(alert_threshold),
            "total_requests": int(total),
            "fraud_count": int(fraud_count),
            "legit_count": int(max(total - fraud_count - error_count, 0)),
            "error_count": int(error_count),
            "alert_count": int(alert_count),
            "avg_fraud_probability": round(avg_probability, 6),
            "avg_latency_ms": round(avg_latency_ms, 2),
        }
    finally:
        connection.close()


def get_recent_events(limit=100, alert_threshold=DEFAULT_ALERT_THRESHOLD):
    connection = get_db_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, event_time, prediction, fraud_probability, amount, latency_ms, status, error_message
            FROM prediction_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        events = []
        for row in rows:
            prediction = row["prediction"]
            fraud_probability = row["fraud_probability"]

            is_alert = (
                row["status"] == "success"
                and (
                    prediction == 1
                    or (
                        fraud_probability is not None
                        and float(fraud_probability) >= float(alert_threshold)
                    )
                )
            )

            events.append(
                {
                    "id": int(row["id"]),
                    "event_time": row["event_time"],
                    "prediction": int(prediction) if prediction is not None else None,
                    "fraud_probability": (
                        round(float(fraud_probability), 6)
                        if fraud_probability is not None
                        else None
                    ),
                    "amount": round(float(row["amount"]), 2) if row["amount"] is not None else None,
                    "latency_ms": round(float(row["latency_ms"]), 2) if row["latency_ms"] is not None else None,
                    "status": row["status"],
                    "error_message": row["error_message"],
                    "is_alert": is_alert,
                }
            )

        return events
    finally:
        connection.close()


initialize_monitoring_store()

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.perf_counter()
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Ensure the columns are in the same order as the training data
        # This is critical for consistent predictions
        # Assuming X_train_scaled.columns is the correct order
        # In a production API, you would predefine this order or validate input.
        expected_columns = model.feature_names_in_ # Get feature names from the fitted XGBoost model
        df = df[expected_columns]

        # Apply the same scaling as during training
        # Assuming 'Time' and 'Amount' were scaled
        if global_scaler is None:
             return jsonify({'error': 'Scaler not loaded'}), 500

        df[['Time', 'Amount']] = global_scaler.transform(df[['Time', 'Amount']])

        # Make prediction
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[:, 1]

        prediction_value = int(prediction[0])
        fraud_probability = float(prediction_proba[0])
        latency_ms = (time.perf_counter() - start_time) * 1000
        amount = float(data.get("Amount")) if isinstance(data, dict) and data.get("Amount") is not None else None

        log_prediction_event(
            prediction=prediction_value,
            fraud_probability=fraud_probability,
            amount=amount,
            latency_ms=latency_ms,
            status="success",
        )

        return jsonify({"prediction": prediction_value, "fraud_probability": fraud_probability})

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        amount = None
        try:
            if isinstance(request.get_json(silent=True), dict):
                raw_amount = request.get_json(silent=True).get("Amount")
                amount = float(raw_amount) if raw_amount is not None else None
        except Exception:
            amount = None

        log_prediction_event(
            prediction=None,
            fraud_probability=None,
            amount=amount,
            latency_ms=latency_ms,
            status="error",
            error_message=str(e),
        )
        return jsonify({"error": str(e)}), 400


@app.route('/monitoring/summary', methods=['GET'])
def monitoring_summary():
    try:
        window_minutes = int(request.args.get("window_minutes", 15))
    except ValueError:
        return jsonify({"error": "window_minutes must be an integer"}), 400

    try:
        alert_threshold = float(request.args.get("alert_threshold", DEFAULT_ALERT_THRESHOLD))
    except ValueError:
        return jsonify({"error": "alert_threshold must be a float"}), 400

    summary = get_summary(window_minutes=window_minutes, alert_threshold=alert_threshold)
    return jsonify(summary)


@app.route('/monitoring/events', methods=['GET'])
def monitoring_events():
    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    try:
        alert_threshold = float(request.args.get("alert_threshold", DEFAULT_ALERT_THRESHOLD))
    except ValueError:
        return jsonify({"error": "alert_threshold must be a float"}), 400

    limit = max(1, min(limit, 500))
    events = get_recent_events(limit=limit, alert_threshold=alert_threshold)
    return jsonify({"count": len(events), "events": events})

if __name__ == '__main__':
    # For deployment, use a production-ready WSGI server like Gunicorn
    # For local testing, you can run:
    # flask run --host=0.0.0.0 --port=5000
    # Or directly using app.run() in development mode (not recommended for production)
    app.run(host='0.0.0.0', port=5000)
