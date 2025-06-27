import streamlit as st
import numpy as np
import joblib

# Load the model using joblib (NOT pickle)
try:
    model = joblib.load("encrypted_traffic_classifier_reused.pkl")
    le = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Top 15 features used for prediction
feature_names = [
    "max_flowiat", "flowBytesPerSecond", "flowPktsPerSecond",
    "mean_flowiat", "max_fiat", "mean_fiat", "mean_biat",
    "burstiness", "duration", "total_fiat", "std_flowiat",
    "max_biat", "min_flowiat", "total_biat", "min_biat"
]

# Sidebar - display classification report
st.sidebar.header("📊 Classification Report")
st.sidebar.text("""
               precision  recall  f1-score  support
VPN-BROWSING       0.90     0.92     0.91      500
VPN-CHAT           0.77     0.72     0.75      239
VPN-FT             0.85     0.82     0.83      387
VPN-MAIL           0.83     0.91     0.87       98
VPN-P2P            0.80     0.90     0.85      186
VPN-STREAMING      0.92     0.94     0.93       95
VPN-VOIP           1.00     0.98     0.99      454

Accuracy:          0.89
Macro Avg:         0.87     0.88     0.87
Weighted Avg:      0.89     0.89     0.89
""")

# App title and instructions
st.title("🔒 Encrypted Traffic Classifier")
st.write("Please input values for each of the top 15 network flow features below:")

# User form for input
with st.form("prediction_form"):
    user_inputs = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, format="%.6f")
        user_inputs.append(val)
    
    submitted = st.form_submit_button("Predict")

# Prediction section
if submitted:
    input_array = np.array([user_inputs])
    try:
        pred_label = model.predict(input_array)[0]
        pred_class = le.inverse_transform([pred_label])[0]
        st.success(f"🧠 Predicted Traffic Type: **{pred_class}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
