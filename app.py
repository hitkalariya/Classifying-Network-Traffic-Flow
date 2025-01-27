import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("Classifying_Network_Traffic_Flow.pkl")

# Streamlit App Title
st.title("Network Traffic Flow Classifier")
st.write(
    "This app uses a trained machine learning model to classify network traffic flows as either DDoS (1) or BENIGN (0)."
)

# The actual feature names
feature_names = [
    'Bwd Packet Length Max', ' Fwd Packet Length Max',
    'Init_Win_bytes_forward', ' act_data_pkt_fwd', ' Subflow Fwd Bytes'
]

# Input Features
st.header("Enter Feature Values")

# Create dynamic input fields for all features using the actual feature names
features = []
for feature_name in feature_names:
    feature_value = st.number_input(
        f"{feature_name.strip()}", value=0.0, step=0.1, help=f"Enter the value for {feature_name.strip()}."
    )
    features.append(feature_value)

# Prediction Button
if st.button("Predict"):
    try:
        # Convert features to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features_array)
        probabilities = model.predict_proba(features_array)

        # Mapping class labels
        class_mapping = {0: "BENIGN", 1: "DDoS"}  # Adjust based on your dataset
        predicted_class = class_mapping[prediction[0]]

        # Display Results
        st.success(f"Predicted Class: {predicted_class}")
        st.write("Class Probabilities:")
        for i, prob in enumerate(probabilities[0]):
            st.write(f"{class_mapping[i]}: {prob:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
