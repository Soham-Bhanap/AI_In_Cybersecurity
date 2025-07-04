# 🔐 AI/ML for Networking – Encrypted Traffic and Threat URL Classification
Intel Unnati, industrial training project

Modern networks face rising threats due to encrypted traffic, increased data volume, and sophisticated cyberattacks. This repository presents an AI-powered solution suite for **real-time traffic classification** and **URL-based threat detection**, targeting next-gen cybersecurity in privacy-preserving and high-throughput network environments.

---

## 🚀 Project Components

### 1. 📡 Traffic Classification

An AI-based system that classifies encrypted network traffic using advanced feature engineering and ensemble learning.

#### Features:
- **Encrypted Traffic Support** – Classifies VPN-based and encrypted application traffic without payload inspection.
- **Behavioral Feature Engineering** – Derives 15+ flow-level statistical and temporal features.
- **Stacked Ensemble Model** – Combines SVM & XGBoost with `SelectKBest`, SMOTE, and hyperparameter tuning.
- **High Accuracy** – Achieves ~89% overall accuracy with F1-score > 0.88 across classes.
- **Streamlit App** – Intuitive UI to input flow features and view predictions live.

#### Input Example (Streamlit):
- `max_flowiat`, `burstiness`, `flow_duration_per_packet`, `fwd_bwd_timing_ratio`, etc.

#### 📁 Key Files:
- `traffic_classifier_notebook.ipynb`: Jupyter Notebook for training.
- `model.pkl`, `label_encoder.pkl`: Serialized trained model.
- `app.py`: Streamlit interface for real-time traffic prediction.

---

### 2. 🧠 Threat URL Classifier

A GPU-optimized deep neural ensemble model (`ForestNet`) to classify URLs into security threat categories.

#### Features:
- **URL Lexical & Structural Analysis** – Extracts 14+ features including entropy, special characters, presence of HTTPS, keywords, IPs.
- **Multi-Class Threat Classification** – Detects `phishing`, `malware`, `defacement`, and `benign` URLs.
- **ForestNet Architecture** – PyTorch model mimicking random forest using 300 mini neural networks.
- **High Accuracy** – ~90% accuracy, macro F1-score ~0.85.
- **Streamlit Web App** – Classify individual or batch URLs via a user-friendly interface.

#### Example Features:
- URL length, domain length, number of digits, entropy, keyword presence (e.g., "login", "secure")

#### 📁 Key Files:
- `threat_classifier_notebook.ipynb`: Full model training notebook.
- `gpu_forest_model.pth`: Trained ForestNet model.
- `label_encoder.pkl`: Encoded label mappings.
- `app.py`: Streamlit app for URL classification.

---

## 📊 Evaluation Metrics

| Model                | Accuracy | Macro F1 | Notes                          |
|---------------------|----------|----------|--------------------------------|
| Traffic Classifier  | 89%      | 0.88+    | Strong performance on VPN/Encrypted |
| Threat Classifier   | 90%      | 0.85     | Phishing class lower recall (~0.48) |

---

## 📦 Installation & Usage

### ✅ Prerequisites
```bash
pip install -r requirements.txt
# or manually install:
pip install streamlit scikit-learn xgboost imbalanced-learn torch joblib numpy pandas
