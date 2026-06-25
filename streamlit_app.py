import streamlit as st
import joblib
import pandas as pd
import numpy as np


# --- Configuration --- #
MODEL_PATH = 'final_model.joblib'
SCALER_PATH = 'scaler_for_api.joblib'

# --- Load Model and Scaler --- #
@st.cache_resource # Cache the model and scaler to avoid reloading on every rerun
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler: {e}. Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the correct directory.")
        st.stop() # Stop the app if files are not found

model, scaler = load_model_and_scaler()

# --- Streamlit UI --- #
st.title('Fraud Detection App')
st.write('Enter transaction details to predict if it is fraudulent or not.')

# Input fields for transaction features
st.sidebar.header('Transaction Features')

# For simplicity, we'll create input fields for 'Time', 'Amount', and a few 'V' features.
# In a full application, you might provide all 28 V features or a subset.
# The order of features is crucial, ensure it matches your training data.

time = st.sidebar.number_input('Time (seconds since first transaction)', min_value=0.0, value=10000.0, step=1.0)
amount = st.sidebar.number_input('Amount', min_value=0.0, value=50.0, step=0.01)

st.sidebar.subheader('V-Features (Sample)')
# These are just example V-features. In a real app, you'd list all 28.
# The actual range and distribution of these features are important.
# For now, using generic sliders or number inputs.

v_features = {}
for i in range(1, 29): # V1 to V28
    # Using a placeholder range for demo. Adjust min/max based on actual data distribution.
    # Example: V1 input, adjust default and range as needed.
    if i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]: # Ensure we only create the relevant ones
        v_features[f'V{i}'] = st.sidebar.slider(f'V{i}', min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

# Combine all inputs into a DataFrame
# Ensure the order of columns matches the training data
input_data = pd.DataFrame({
    'Time': [time],
    'Amount': [amount],
    **{f'V{i}': [v_features[f'V{i}']] for i in range(1, 29) if f'V{i}' in v_features}
})

# Reorder columns to match the model's expected input order (X_train_scaled.columns)
# This is critical. We'll derive this from the model's feature_names_in_ attribute.
expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None

if expected_columns is None:
    st.error("Could not determine expected feature order from the model. Make sure your model was trained with scikit-learn compatible feature names.")
    st.stop()

# Filter input_data to only include expected columns and reorder them
# Handle missing columns by adding them with default (e.g., 0) if necessary
processed_input = pd.DataFrame(columns=expected_columns)
for col in expected_columns:
    if col in input_data.columns:
        processed_input[col] = input_data[col]
    else:
        processed_input[col] = 0.0 # Fill missing V-features with 0 or a sensible default


# Apply the same scaling as during training (only to 'Time' and 'Amount')
processed_input_scaled = processed_input.copy()
processed_input_scaled[['Time', 'Amount']] = scaler.transform(processed_input[['Time', 'Amount']])


if st.button('Predict'):
    prediction = model.predict(processed_input_scaled)[0]
    prediction_proba = model.predict_proba(processed_input_scaled)[:, 1][0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f"Fraudulent Transaction Detected! (Probability: {prediction_proba:.4f})")
    else:
        st.success(f"Legitimate Transaction (Probability: {prediction_proba:.4f})")

    st.write(f"Full input data used for prediction:")
    st.dataframe(processed_input_scaled)

# --- Save the Streamlit app content to a file --- #
streamlit_app_content = st.session_state.get('streamlit_app_content', '') # Get content if already exists
with open('streamlit_app.py', 'w') as f:
    f.write(f'''
import streamlit as st
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = '{MODEL_PATH}'
SCALER_PATH = '{SCALER_PATH}'

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler: {{e}}. Please ensure '{{MODEL_PATH}}' and '{{SCALER_PATH}}' are in the correct directory.")
        st.stop()

model, scaler = load_model_and_scaler()

st.title('Fraud Detection App')
st.write('Enter transaction details to predict if it is fraudulent or not.')

st.sidebar.header('Transaction Features')

time = st.sidebar.number_input('Time (seconds since first transaction)', min_value=0.0, value=10000.0, step=1.0)
amount = st.sidebar.number_input('Amount', min_value=0.0, value=50.0, step=0.01)

st.sidebar.subheader('V-Features (Sample)')
v_features = {{}}
for i in range(1, 29):
    if f'V{{i}}' in model.feature_names_in_:
        v_features[f'V{{i}}'] = st.sidebar.slider(f'V{{i}}', min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

input_data = pd.DataFrame({{
    'Time': [time],
    'Amount': [amount],
    **{{f'V{{i}}': [v_features[f'V{{i}}']] for i in range(1, 29) if f'V{{i}}' in v_features}}
}})

expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None

if expected_columns is None:
    st.error("Could not determine expected feature order from the model. Make sure your model was trained with scikit-learn compatible feature names.")
    st.stop()

processed_input = pd.DataFrame(columns=expected_columns)
for col in expected_columns:
    if col in input_data.columns:
        processed_input[col] = input_data[col]
    else:
        processed_input[col] = 0.0

processed_input_scaled = processed_input.copy()
processed_input_scaled[['Time', 'Amount']] = scaler.transform(processed_input[['Time', 'Amount']])


if st.button('Predict'):
    prediction = model.predict(processed_input_scaled)[0]
    prediction_proba = model.predict_proba(processed_input_scaled)[:, 1][0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f"Fraudulent Transaction Detected! (Probability: {{prediction_proba:.4f}})")
    else:
        st.success(f"Legitimate Transaction (Probability: {{prediction_proba:.4f}})")

    st.write(f"Full input data used for prediction:")
    st.dataframe(processed_input_scaled)
''')

st.success("Streamlit app content written to `streamlit_app.py`")

    
