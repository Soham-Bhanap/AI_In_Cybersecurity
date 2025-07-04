import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import re
from urllib.parse import urlparse
import os

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    features = {
        'url_length': len(url),
        'domain_length': len(domain),
        'num_digits': sum(c.isdigit() for c in url),
        'special_chars': sum(url.count(c) for c in ['@', '?', '=', '.', '-', '_', '/']),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'num_subdomains': domain.count('.') - 1 if domain.count('.') > 1 else 0,
        'path_length': len(path),
        'num_params': path.count('?') + path.count('&'),
        'has_port': 1 if ':' in domain else 0,
        'is_ip': 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0,
        'file_extension': 1 if '.' in path.split('/')[-1] else 0,
        'entropy': calculate_entropy(url),
        'num_redirects': url.count('//') - 1,
        'has_phish_keywords': 1 if any(kw in url.lower() for kw in ['login', 'verify', 'secure', 'account']) else 0
    }
    return features

def calculate_entropy(string):
    prob = [float(string.count(c)) / len(string) for c in set(string)]
    return -sum(p * np.log2(p) for p in prob)

# Model definition (must match training)
class ForestNet(nn.Module):
    def __init__(self, input_size, num_classes, num_trees=300, tree_depth=5):
        super().__init__()
        self.trees = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            ) for _ in range(num_trees)
        ])
    def forward(self, x):
        outputs = [tree(x) for tree in self.trees]
        return torch.stack(outputs).mean(dim=0)

# Load model and label encoder
@st.cache_resource
def load_model():
    checkpoint = torch.load('gpu_forest_model.pth', map_location=torch.device('cpu'))
    input_size = checkpoint['input_size']
    num_classes = checkpoint['num_classes']
    model = ForestNet(input_size, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    le = joblib.load('label_encoder.pkl')
    return model, le

model, le = load_model()

st.set_page_config(page_title="Threat URL Classifier", layout="centered")
st.title("🔒 Threat URL Classifier (GPU Forest Model)")
st.write("Classify URLs as benign, defacement, malware, or phishing using a trained AI model.")

# Show model metrics (static, from notebook)
st.subheader("Model Metrics (Test Set)")
st.markdown('''
- **Accuracy:** 0.90
- **Macro F1-score:** 0.85
- **Class-wise:**
    - Benign: Precision 0.90, Recall 0.99, F1 0.94
    - Defacement: Precision 0.92, Recall 0.96, F1 0.94
    - Malware: Precision 0.94, Recall 0.86, F1 0.90
    - Phishing: Precision 0.86, Recall 0.48, F1 0.62
''')

st.divider()

# Single URL classification
st.header("Classify a Single URL")
url_input = st.text_input("Enter a URL to classify:")
if st.button("Classify URL") and url_input:
    features = pd.DataFrame([extract_features(url_input)])
    X = torch.tensor(features.values.astype(np.float32))
    with torch.no_grad():
        output = model(X)
        pred = output.argmax(dim=1).item()
        label = le.inverse_transform([pred])[0]
    st.success(f"Prediction: **{label}**")

st.divider()

# Batch CSV classification
st.header("Classify URLs from CSV File")
file = st.file_uploader("Upload a CSV file with a 'url' column:", type=['csv'])
if file is not None:
    try:
        df = pd.read_csv(file)
        if 'url' not in df.columns:
            st.error("CSV must contain a 'url' column.")
        else:
            features = df['url'].apply(extract_features).apply(pd.Series)
            X = torch.tensor(features.values.astype(np.float32))
            with torch.no_grad():
                outputs = model(X)
                preds = outputs.argmax(dim=1).numpy()
                labels = le.inverse_transform(preds)
            df['prediction'] = labels
            # Reorder columns: url, prediction, then the rest
            cols = ['url', 'prediction'] + [c for c in df.columns if c not in ['url', 'prediction']]
            st.dataframe(df[cols])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "classified_urls.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}") 