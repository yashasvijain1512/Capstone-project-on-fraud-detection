## README: Fraud Detection API

This document outlines the setup, deployment, and usage of the Fraud Detection API. This API uses a pre-trained XGBoost model to predict the likelihood of a transaction being fraudulent based on transaction features.

### 1. Project Objective

To develop a fraud detection pipeline that predicts fraud probability, evaluates performance using imbalanced classification metrics, and exposes predictions through a simple API.

### 2. Model

The final model used in this API is a **Tuned XGBoost Classifier**, saved as `final_model.joblib`. It was selected due to its superior performance in balancing precision and recall, and minimizing false positives on the highly imbalanced credit card fraud dataset. A `StandardScaler` (`scaler_for_api.joblib`) is used for preprocessing the `Time` and `Amount` features, as was done during model training.

### 3. API Endpoints

The API provides the following endpoint:

*   **`/predict` (POST)**: Accepts transaction data as a JSON payload and returns a fraud prediction (0 for non-fraud, 1 for fraud) along with the probability of fraud.

### 4. Setup and Dependencies

To set up and run the API, ensure you have Python 3.8+ installed. Then, follow these steps:

#### A. Prerequisites:

1.  **Model and Scaler Files**: Ensure the `final_model.joblib` and `scaler_for_api.joblib` files are in the same directory as `app.py`. These files should have been generated and saved during the model training and preparation phase in the Colab notebook.

#### B. Installation:

1.  **Clone the repository (if applicable) or create the necessary files:**
    *   Create `app.py` (content generated in the Colab notebook).
    *   Ensure `final_model.joblib` and `scaler_for_api.joblib` are present.
2.  **Install Python dependencies:**

    ```bash
    pip install Flask gunicorn pandas scikit-learn xgboost joblib
    ```

### 5. Running the API

You can run the API either for local development or for production deployment.

#### A. For Local Testing/Development (not recommended for production):

Open your terminal in the directory containing `app.py` and run:

```bash
python app.py
```

The API will be accessible at `http://127.0.0.1:5000`.

#### B. For Production (using Gunicorn):

For a more robust and production-ready deployment, use Gunicorn:

Open your terminal in the directory containing `app.py` and run:

```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

The API will be accessible on port `5000` of the host it's running on (e.g., `http://your-server-ip:5000`).

### 6. Testing the API

Once the API is running, you can send POST requests to the `/predict` endpoint. The request body should be a JSON object containing the 30 features of a transaction (`Time`, `V1` through `V28`, and `Amount`).

#### A. Example Request Structure:

```json
{
  "Time": 12345.0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338320,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

#### B. Sample Predictions (using `requests` in Python):

To test the API programmatically, you would use a library like `requests`. Replace `http://localhost:5000/predict` with the actual URL where your API is running.

```python
import requests
import json

api_url = 'http://localhost:5000/predict' # Or your deployed API URL
headers = {'Content-Type': 'application/json'}

def send_prediction_request(transaction_data):
    try:
        response = requests.post(api_url, data=json.dumps(transaction_data), headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is it running?")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# --- Sample 1: Non-Fraudulent Transaction (from X_test, Class 0) ---
# Example using the first row of X_test from the Colab notebook (actual class: 0)
sample_non_fraud = {
  "Time": 160760.0,
  "V1": -0.674466064578314,
  "V2": 1.40810501967799,
  "V3": -1.11062205357093,
  "V4": -1.32836577843066,
  "V5": 1.38899603254837,
  "V6": -1.30843906707795,
  "V7": 1.88587890268717,
  "V8": -0.614232966299775,
  "V9": 0.311652212453101,
  "V10": 0.65075700363522,
  "V11": -0.857784661547805,
  "V12": -0.229961445775592,
  "V13": -0.19981700479103,
  "V14": 0.266371326329879,
  "V15": -0.0465441684754424,
  "V16": -0.741398089749789,
  "V17": -0.605616644106022,
  "V18": -0.39256818789208,
  "V19": -0.162648311024695,
  "V20": 0.394321820843914,
  "V21": 0.0800842396026648,
  "V22": 0.810033595602455,
  "V23": -0.224327230436412,
  "V24": 0.707899237446867,
  "V25": -0.13583702273753,
  "V26": 0.0451021964988772,
  "V27": 0.533837219064273,
  "V28": 0.291319252625364,
  "Amount": 23.0
}
print("\n--- Testing Non-Fraudulent Transaction ---")
response = send_prediction_request(sample_non_fraud)
if response: print(json.dumps(response, indent=2))

# --- Sample 2: Another Non-Fraudulent Transaction (from X_test, Class 0) ---
sample_non_fraud_2 = {
  "Time": 406.0,
  "V1": -2.312226542,
  "V2": 1.951992011,
  "V3": -1.609850731,
  "V4": 3.997906069,
  "V5": -0.522187864,
  "V6": -1.426545318,
  "V7": -2.537387306,
  "V8": 1.391657252,
  "V9": -2.770089277,
  "V10": -2.772272145,
  "V11": 3.202033207,
  "V12": -2.899907385,
  "V13": -0.592509655,
  "V14": -4.289253753,
  "V15": 0.38972412,
  "V16": -1.140747185,
  "V17": -2.830055675,
  "V18": -0.016822468,
  "V19": 0.416956198,
  "V20": 0.126910543,
  "V21": 0.517232371,
  "V22": -0.035049368,
  "V23": -0.465211076,
  "V24": 0.320198199,
  "V25": 0.044519356,
  "V26": 0.177839798,
  "V27": 0.261145003,
  "V28": 0.143681423,
  "Amount": 99.0
}
print("\n--- Testing Another Non-Fraudulent Transaction ---")
response = send_prediction_request(sample_non_fraud_2)
if response: print(json.dumps(response, indent=2))

# --- Sample 3: Fraudulent Transaction (from X_test, Class 1) ---
# Example using a known fraudulent transaction (e.g., from X_test where y_test == 1)
sample_fraud = {
  "Time": 472.0,
  "V1": -3.043541334,
  "V2": 3.156179352,
  "V3": -3.213537651,
  "V4": 3.333256037,
  "V5": 2.710898516,
  "V6": -0.466906204,
  "V7": -0.27962483,
  "V8": -1.002212354,
  "V9": 0.697576059,
  "V10": -1.632679237,
  "V11": 0.36350478,
  "V12": -0.748530366,
  "V13": -0.009949673,
  "V14": -0.430459345,
  "V15": 1.157850117,
  "V16": -0.662137976,
  "V17": 0.167444318,
  "V18": 0.162747754,
  "V19": -0.198271501,
  "V20": 0.264561858,
  "V21": 0.661969248,
  "V22": 0.052824982,
  "V23": -0.207963384,
  "V24": -0.071624686,
  "V25": 0.252934812,
  "V26": 0.399710356,
  "V27": 0.141045763,
  "V28": 0.265415307,
  "Amount": 1.0
}
print("\n--- Testing Fraudulent Transaction ---")
response = send_prediction_request(sample_fraud)
if response: print(json.dumps(response, indent=2))

# --- Known Failure Case: Missing a required feature ---
# The API expects all 30 features. Missing one should result in an error.
print("\n--- Testing Failure Case (Missing Feature) ---")
failure_case_data = {
  "Time": 123.0,
  "V1": 0.0,
  "V2": 0.0,
  "Amount": 10.0
  # Many V-features are intentionally missing
}
response = send_prediction_request(failure_case_data)
if response: print(json.dumps(response, indent=2))

# --- Known Failure Case: Invalid data type ---
# Sending a string for a numerical feature should result in an error.
print("\n--- Testing Failure Case (Invalid Data Type) ---")
failure_case_data_2 = {
  "Time": "invalid",
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338320,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
response = send_prediction_request(failure_case_data_2)
if response: print(json.dumps(response, indent=2))
```

#### C. Fraud Risk Interpretation

The API returns a `fraud_probability` which is a float value between 0 and 1. A value closer to 1 indicates a higher probability of the transaction being fraudulent, while a value closer to 0 indicates a lower probability. The `prediction` field provides a binary classification (0 or 1) based on a default threshold (usually 0.5, but adjustable if needed) applied to this probability.

### 7. Deployment Considerations

*   **Scalability**: For high-traffic applications, consider deploying the API using containerization technologies (e.g., Docker) and orchestration platforms (e.g., Kubernetes) on cloud providers (GCP, AWS, Azure).
*   **Security**: Implement API key authentication, HTTPS, and proper access controls to secure your API endpoints.
*   **Monitoring**: Set up logging and monitoring to track API performance, errors, and model predictions in real-time.
*   **Environment Variables**: Use environment variables to manage sensitive information or configuration settings (e.g., model paths, thresholds) rather than hardcoding them in `app.py`.

### 8. Further Enhancements

*   **Error Handling**: Implement more granular error handling and specific error codes.
*   **Input Validation**: Add more robust input validation to ensure incoming data conforms to expected types and ranges.
*   **Explainability**: Integrate model interpretability techniques (e.g., SHAP, LIME) to provide reasons for fraud predictions, which can be crucial for investigations.
