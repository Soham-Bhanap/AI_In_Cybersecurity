import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("url_attack_classifier.pkl")

st.title("URL Attack Type Classifier")

option = st.radio("Select input type:", ["Single URL", "Upload CSV"])

if option == "Single URL":
    user_input = st.text_input("Enter a URL:")
    if user_input:
        pred = model.predict([user_input])[0]
        st.success(f"Predicted attack type: **{pred}**")

elif option == "Upload CSV":
    file = st.file_uploader("Upload a CSV with a 'url' column", type="csv")
    if file:
        df = pd.read_csv(file)
        if "url" in df.columns:
            df["prediction"] = model.predict(df["url"])
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.error("CSV must contain a 'url' column.")
