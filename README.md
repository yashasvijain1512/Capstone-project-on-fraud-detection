# Credit Card Fraud Detection API

A production-ready API for detecting fraudulent credit card transactions using an XGBoost machine learning model.

## Overview

This project provides a fraud detection pipeline that:
- Predicts fraud probability for credit card transactions
- Exposes predictions through a REST API (`/predict`)
- Stores request telemetry (prediction, probability, latency, errors) in SQLite for observability
- Exposes monitoring APIs for live metrics and event feeds
- Uses an optimized XGBoost classifier trained on imbalanced transaction data
- Scales transaction features (`Time`, `Amount`) using StandardScaler

## Dataset & Features

**Data Source**: Credit Card Fraud Detection Dataset (highly imbalanced)
- **Target**: Binary classification (0 = Non-Fraud, 1 = Fraud)
- **Features**: 30 predictors
  - `Time`: Seconds elapsed between first transaction and current transaction
  - `Amount`: Transaction amount
  - `V1-V28`: Principal Component Analysis (PCA) transformed features from transaction data

**Data Characteristics**:
- Highly imbalanced dataset (fraud cases represent ~0.17% of transactions)
- PCA features (V1-V28) are scaled and anonymized for privacy
- `Time` and `Amount` are scaled using StandardScaler before model inference

## Model

**Algorithm**: XGBoost Classifier (Tuned)
**Selection Criteria**: Superior performance in balancing precision and recall while minimizing false positives
**Files**:
- `final_model.joblib` - Trained XGBoost model
- `scaler_for_api.joblib` - StandardScaler for feature preprocessing

## API Endpoint

**POST `/predict`**
- **Input**: JSON object with 30 features (Time, V1-V28, Amount)
- **Output**: JSON object with `prediction` (0 or 1) and `fraud_probability` (0.0-1.0)

## Monitoring Endpoints

**GET `/monitoring/summary`**
- Query params:
  - `window_minutes` (default `15`)
  - `alert_threshold` (default `0.8`)
- Returns aggregate metrics for the selected time window, including request volume, fraud count, alert count, errors, average fraud probability, and average latency.

**GET `/monitoring/events`**
- Query params:
  - `limit` (default `100`, max `500`)
  - `alert_threshold` (default `0.8`)
- Returns recent prediction events with alert flags, status, latency, and error details.

### Alert Logic

An event is marked as an alert when all of the following are true:
- Request status is `success`
- Either `prediction = 1` or `fraud_probability >= alert_threshold`

This same rule is used for:
- `alert_count` in `/monitoring/summary`
- `is_alert` for each item in `/monitoring/events`

## Installation

**Requirements**: Python 3.8+

1. **Prerequisites**:
   - `final_model.joblib` and `scaler_for_api.joblib` must be in the project directory

2. **Install dependencies**:
   ```bash
  pip install -r requirements.txt
   ```

## Running the API

**Development** (Local testing):
```bash
python app.py
```
API accessible at `http://127.0.0.1:5000`

### Streamlit App (Prediction + Monitoring + Model Comparison)
```bash
streamlit run streamlit_app.py
```

The Streamlit app has three separate sections (selected from sidebar `Dashboard Section`):

1. **Prediction Input**
- Manual transaction input (`Time`, `V1`-`V28`, `Amount`)
- Prediction result (`Fraudulent` or `Legitimate`)
- Fraud probability display
- Processed feature preview sent to model

2. **Monitoring & Alerts**
- Monitoring settings: API URL, summary window, recent event limit, alert threshold
- Optional auto-refresh controls (`Auto-refresh dashboard`, `Refresh every`)
- Summary metric cards:
  - `Requests`: total monitored requests in selected window
  - `Fraud Flags`: successful predictions where `prediction = 1`
  - `Alerts`: events that satisfy alert logic
  - `Errors`: failed prediction requests
- Performance cards:
  - `Avg Fraud Probability`
  - `Avg Latency (ms)`
- `Recent Alerts` table: only rows with `is_alert = true`
- `Recent Events` table: full event feed including `status`, `error_message`, and `is_alert`

3. **Model Comparison**
- Loads tuned models from `Model_DIR/`:
  - `best_logistic_regression_rs_model.joblib`
  - `best_nn_model.joblib`
  - `best_xgboost_model.joblib`
- Uses local `credit_card.csv` (or uploaded CSV) for evaluation
- Compares Precision, Recall, F1, PR-AUC, ROC-AUC, false positives/negatives, and business cost
- Shows PR/ROC curves, threshold sensitivity chart, and confusion matrices
- Provides winner summary based on selected ranking metric
- Supports sampled evaluation for faster rendering on large datasets
- Supports CSV export of model comparison summary

Separating these sections prevents fast monitoring refresh from interrupting user input while entering transaction details.

### Standalone Model Comparison Dashboard (Notebook Reference)

A dedicated dashboard for model comparison is also available in `Dashboard/`:

Run from project root:

```bash
streamlit run Dashboard/model_comparison_dashboard.py
```


This standalone page uses the same model-comparison component as the integrated `Model Comparison` section in `streamlit_app.py`.

#### Quick Tour (First-Time Users)

Use this short flow to explore the app end-to-end:

1. Start the API:
  ```bash
  python app.py
  ```
2. Start Streamlit in another terminal:
  ```bash
  streamlit run streamlit_app.py
  ```
3. Open the Streamlit URL (usually `http://127.0.0.1:8501`).
4. In sidebar `Dashboard Section`, select **Prediction Input**.
5. Enter sample transaction values and click **Predict**.
6. Review:
  - Classification result (`Fraudulent` or `Legitimate`)
  - Fraud probability score
  - Processed feature table
7. Switch sidebar `Dashboard Section` to **Monitoring & Alerts**.
8. Set:
  - `Summary Window (minutes)`
  - `Alert Threshold (Fraud Probability)`
  - `Recent Events`
9. Observe dashboard blocks:
  - Top metrics (`Requests`, `Fraud Flags`, `Alerts`, `Errors`)
  - Performance (`Avg Fraud Probability`, `Avg Latency (ms)`)
  - `Recent Alerts` table
  - `Recent Events` table
10. Switch to **Model Comparison** to evaluate all three tuned models on `credit_card.csv`.
11. Turn on `Auto-refresh dashboard` only when actively monitoring live traffic.

Recommended screenshots for documentation:
- Prediction Input view with filled transaction fields and prediction result
- Monitoring & Alerts view showing metric cards and at least one alert row
- Model Comparison view with summary table and PR/ROC charts

**Production** (Using Gunicorn):
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## Docker Deployment

### Build and Run Both Services (API + Dashboard)
```bash
docker compose up --build
```

### Access Services
- API: `http://127.0.0.1:5000`
- Streamlit Dashboard: `http://127.0.0.1:8501`

### Run in Detached Mode
```bash
docker compose up --build -d
```

### Stop Services
```bash
docker compose down
```

### Persistent Monitoring Data
- Monitoring events are stored in a Docker volume (`monitoring_data`) at `/app/data/monitoring.db`.
- This allows telemetry to persist across container restarts.

## Testing the API

### Run Demo with Pre-trained Model (without Flask):
```bash
python demo_model_samples.py
```

### Run API Demo Tests (with Flask running):
1. Start the API:
   ```bash
   python app.py
   ```

2. In another terminal, run:
   ```bash
   python run_api_demo_samples.py
   ```

### Manual Testing with curl:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 12345.0,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
    "V4": 1.378155, "V5": -0.338320, "V6": 0.462388,
    "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
    "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
    "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
    "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
    "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
    "V28": -0.021053, "Amount": 149.62
  }'
```

### Response Format:
```json
{
  "prediction": 0,
  "fraud_probability": 0.000303
}
```
- `prediction`: 0 (non-fraud) or 1 (fraud)
- `fraud_probability`: Float between 0 and 1 (higher = more likely fraud)

## Deployment Considerations

- **Scalability**: Use Docker & Kubernetes for containerized deployment on cloud platforms (GCP, AWS, Azure)
- **Security**: Implement API authentication, HTTPS, and access controls
- **Monitoring**: Set up logging to track API performance and predictions
- **Configuration**: Use environment variables for model paths and thresholds

## Further Enhancements

- Add model explainability (SHAP, LIME) for fraud investigation support
- Implement comprehensive input validation and error handling
- Add batch prediction endpoint for bulk transaction processing
- Implement model versioning and A/B testing framework
