import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io
from scipy.io import arff
from sklearn.metrics import classification_report, f1_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model using joblib (NOT pickle)
try:
    model_path = os.path.join(BASE_DIR, "encrypted_traffic_classifier_reused.pkl")
    le_path = os.path.join(BASE_DIR, "label_encoder.pkl")
    model = joblib.load(model_path)
    le = joblib.load(le_path)
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
st.sidebar.header("📊 Test Dataset Classification Report")
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
st.write("Predict the type of encrypted traffic using manual feature input or bulk ARFF upload.")

# Create tabs for Manual Input and Bulk Upload
tab1, tab2 = st.tabs(["Manual Input", "Bulk ARFF Upload"])

with tab1:
    st.subheader("Manual Feature Input")
    st.write("Please input values for each of the top 15 network flow features below:")
    with st.form("prediction_form"):
        user_inputs = []
        for feature in feature_names:
            val = st.number_input(f"{feature}", value=0.0, format="%.6f")
            user_inputs.append(val)
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_array = np.array([user_inputs])
        try:
            pred_label = model.predict(input_array)[0]
            pred_class = le.inverse_transform([pred_label])[0]
            st.success(f"🧠 Predicted Traffic Type: **{pred_class}**")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

with tab2:
    st.subheader("Bulk Prediction via ARFF")
    st.write(f"Upload an ARFF file containing the necessary {len(feature_names)} features.")
    
    uploaded_file = st.file_uploader("Choose an ARFF file", type=["arff"])
    
    if uploaded_file is not None:
        try:
            # Parse ARFF file
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            data, meta = arff.loadarff(stringio)
            df = pd.DataFrame(data)
            
            # Decode byte strings for object columns (like class1)
            for col in df.select_dtypes([object]).columns:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            
            # Handle missing values (-1 to median)
            df.replace(-1, np.nan, inplace=True)
            df = df.apply(lambda x: x.fillna(x.median()) if pd.api.types.is_numeric_dtype(x) else x)
            
            # Compute missing 'burstiness' if necessary
            if 'burstiness' not in df.columns and 'std_flowiat' in df.columns and 'mean_flowiat' in df.columns:
                df['burstiness'] = df['std_flowiat'] / (df['mean_flowiat'] + 1e-6)
            
            # Check if all required features are present
            missing_features = [f for f in feature_names if f not in df.columns]
            
            if missing_features:
                st.error(f"❌ Missing columns in data: {missing_features}")
                st.stop()
            
            # Select exactly the required columns in the correct order
            input_data = df[feature_names].values
                
            with st.spinner('Predicting...'):
                predictions = model.predict(input_data)
                predicted_classes = le.inverse_transform(predictions)
                
            # Add predictions to the dataframe
            df['Predicted_Class'] = predicted_classes
            
            st.success("✅ Prediction complete!")
            
            # Display abstracted output instead of full dataset
            display_cols = []
            if 'duration' in df.columns: 
                display_cols.append('duration')
            if 'class1' in df.columns: 
                display_cols.append('class1') # Actual class
            display_cols.append('Predicted_Class')
            
            st.dataframe(df[display_cols], use_container_width=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Show Metrics and Graphs
            st.divider()
            st.subheader("📈 Prediction Summary")
            
            st.markdown("### Distribution of Predicted Traffic")
            class_counts = df['Predicted_Class'].value_counts()
            
            # Use columns to put the bar chart and raw counts side-by-side
            col_chart, col_counts = st.columns([2, 1])
            with col_chart:
                st.bar_chart(class_counts)
            with col_counts:
                st.markdown("**Raw Counts**")
                st.dataframe(class_counts.reset_index().rename(columns={'Predicted_Class': 'Class', 'count': 'Count'}), use_container_width=True)
                
            st.divider()
            
            # If ground truth label is present in the dataset
            if 'class1' in df.columns:
                st.markdown("### 📊 Model Evaluation (vs Ground Truth)")
                
                y_true = df['class1']
                y_pred = df['Predicted_Class']
                
                accuracy = (y_true == y_pred).sum() / len(df)
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                
                met1, met2, met3 = st.columns(3)
                met1.metric("Overall Accuracy", f"{accuracy:.2%}")
                met2.metric("Macro F1", f"{macro_f1:.2f}")
                met3.metric("Macro Recall", f"{macro_recall:.2f}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_rep, col_cm = st.columns([1, 1])
                with col_rep:
                    st.markdown("**Classification Report**")
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format(precision=2), use_container_width=True)
                    
                with col_cm:
                    st.markdown("**Confusion Matrix Heatmap**")
                    comparison = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
                    
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(comparison, annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                    except ImportError:
                        st.dataframe(comparison, use_container_width=True)
            else:
                st.info("Ground truth column 'class1' not found. Accuracy metrics are hidden.")
            
            # Allow user to download the results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='predictions_output.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")
