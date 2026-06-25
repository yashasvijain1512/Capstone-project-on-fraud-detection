
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler # For preprocessing

app = Flask(__name__)

# Load the trained model
model = joblib.load('final_model.joblib')

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
global_scaler = None

@app.before_first_request
def load_global_scaler():
    global global_scaler
    # This part should be replaced by loading your *saved* scaler, e.g., joblib.load('scaler.joblib')
    # For this demo, we are using the scaler object generated in the cell above.
    # In a real scenario, the scaler would be saved after fitting and loaded here.
    global_scaler = joblib.load('scaler_for_api.joblib') # Assuming you saved it
    print("Scaler loaded successfully within Flask app!")

@app.route('/predict', methods=['POST'])
def predict():
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

        return jsonify({"prediction": int(prediction[0]), "fraud_probability": float(prediction_proba[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # For deployment, use a production-ready WSGI server like Gunicorn
    # For local testing, you can run:
    # flask run --host=0.0.0.0 --port=5000
    # Or directly using app.run() in development mode (not recommended for production)
    app.run(host='0.0.0.0', port=5000)
