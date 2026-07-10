# 🏥 MedPredict — AI-Based Smart Healthcare System

> **B.Tech Final Year Project**
> Early Detection of Diabetes, Heart Disease, and Alzheimer's Disease using Machine Learning

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [ML Models & Algorithms](#-ml-models--algorithms)
- [Setup & Installation](#-setup--installation)
- [Training the Models](#-training-the-models)
- [Running the Web Application](#-running-the-web-application)
- [Evaluation Metrics](#-evaluation-metrics)
- [Explainable AI (XAI)](#-explainable-ai-xai)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

**MedPredict** is an AI-powered smart healthcare system that predicts the risk of three critical diseases:

| Disease | Input Type | Models Used |
|---------|-----------|-------------|
| **Diabetes** | Tabular health data | SVM, Random Forest, XGBoost |
| **Heart Disease** | Cardiovascular parameters | Logistic Regression, Random Forest, XGBoost |
| **Alzheimer's** | Brain MRI images | Custom CNN, VGG16 Transfer Learning |

Users enter health parameters or upload MRI scans through a modern web interface, and the system provides:
- **Risk prediction** with confidence probability
- **Explainable AI** — SHAP feature importance analysis
- **Clinical recommendations** based on results

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedPredict System                        │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Data Layer  │  ML Pipeline │  XAI Engine  │  Web Dashboard    │
│              │              │              │                   │
│  • CSV Data  │  • SVM       │  • SHAP      │  • Streamlit      │
│  • MRI Imgs  │  • RF/XGB    │  • Feature   │  • Plotly Charts  │
│  • Scalers   │  • CNN/VGG16 │    Importance │  • Glassmorphism  │
│  • Cleaning  │  • Training  │  • Per-pred   │  • Probability    │
│  • Splitting │  • Saving    │    Explain    │    Gauges         │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

---

## 📁 Project Structure

```
MedPredict/
│
├── datasets/                    # Dataset storage
│   ├── diabetes/                # Pima Indians Diabetes CSV (auto-downloaded)
│   ├── heart/                   # UCI Heart Disease CSV (auto-downloaded)
│   └── alzheimers/              # Alzheimer MRI images (manual download)
│
├── notebooks/                   # Jupyter notebooks for exploration
│
├── src/                         # Source code
│   ├── preprocessing/           # Data preprocessing modules
│   │   ├── diabetes_preprocessing.py
│   │   ├── heart_preprocessing.py
│   │   └── alzheimer_preprocessing.py
│   │
│   ├── models/                  # ML model architectures
│   │   ├── diabetes_model.py    # SVM, Random Forest, XGBoost
│   │   ├── heart_model.py       # LogReg, Random Forest, XGBoost
│   │   └── alzheimer_model.py   # Custom CNN, VGG16 Transfer Learning
│   │
│   ├── training/                # Training scripts
│   │   ├── train_diabetes_model.py
│   │   ├── train_heart_model.py
│   │   └── train_alzheimer_model.py
│   │
│   └── utils/                   # Shared utilities
│       ├── evaluation.py        # Metrics, confusion matrix, ROC curves
│       └── explainability.py    # SHAP explanations, feature importance
│
├── saved_models/                # Trained model files (generated)
│
├── streamlit_app/               # Web application
│   └── app.py                   # Main Streamlit dashboard
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📊 Datasets

### 1. Diabetes — Pima Indians Diabetes Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples**: 768 | **Features**: 8
- **Auto-downloaded** during training

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic diabetes likelihood |
| Age | Age in years |

### 2. Heart Disease — UCI Heart Disease Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Samples**: 303 | **Features**: 13
- **Auto-downloaded** during training

### 3. Alzheimer's — Brain MRI Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
- **Samples**: ~6,400 MRI images | **Classes**: 4
- **Manual download required**

Download and extract into `datasets/alzheimers/` with this structure:
```
datasets/alzheimers/
    NonDemented/
    VeryMildDemented/
    MildDemented/
    ModerateDemented/
```

---

## 🤖 ML Models & Algorithms

| Disease | Algorithm | Type | Key Strengths |
|---------|-----------|------|---------------|
| Diabetes | SVM | Kernel-based | High-dimensional data |
| Diabetes | Random Forest | Ensemble | Feature importance |
| Diabetes | XGBoost | Gradient Boosting | Best accuracy |
| Heart | Logistic Regression | Linear | Interpretable |
| Heart | Random Forest | Ensemble | Handles non-linearity |
| Heart | XGBoost | Gradient Boosting | Best accuracy |
| Alzheimer | Custom CNN | Deep Learning | End-to-end learning |
| Alzheimer | VGG16 | Transfer Learning | Pre-trained features |

**Best performing models** are automatically selected and saved during training.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone and navigate
```bash
cd MedPredict
```

### Step 2: Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Models

### Train Diabetes Model
```bash
python src/training/train_diabetes_model.py
```
> Auto-downloads the dataset, trains 3 models, saves the best one.

### Train Heart Disease Model
```bash
python src/training/train_heart_model.py
```
> Auto-downloads the dataset, trains 3 models, saves the best one.

### Train Alzheimer's Model
```bash
python src/training/train_alzheimer_model.py
```
> Requires the MRI dataset to be placed in `datasets/alzheimers/` first.

After training, models are saved in `saved_models/`:
```
saved_models/
├── diabetes_model.pkl
├── diabetes_scaler.pkl
├── heart_model.pkl
├── heart_scaler.pkl
├── alzheimer_model.h5
└── *.png  (evaluation plots)
```

---

## 🌐 Running the Web Application

```bash
streamlit run streamlit_app/app.py
```

The dashboard will open at `http://localhost:8501`.

### Features:
- **Home**: Project overview with system architecture
- **Diabetes Prediction**: Enter 8 health parameters → get risk assessment
- **Heart Disease Prediction**: Enter 13 cardiovascular parameters → get risk score
- **Alzheimer's Classification**: Upload brain MRI → get 4-class classification
- **About**: Tech stack, datasets, and future improvements

---

## 📈 Evaluation Metrics

All models are evaluated with:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / (True + False positives) |
| **Recall** | True positives / (True + False negatives) |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve |
| **Confusion Matrix** | Visual breakdown of predictions |

Comparison tables and plots are generated automatically during training.

---

## 🔍 Explainable AI (XAI)

MedPredict uses **SHAP (SHapley Additive exPlanations)** for model interpretability:

- **Global explanations**: Summary plots showing feature importance across all predictions
- **Local explanations**: Per-prediction feature contribution bar charts
- **Feature importance**: Tree-based model feature ranking

SHAP explanations are available for Diabetes and Heart Disease predictions directly in the web UI.

---

## 🚀 Future Improvements

| Area | Description |
|------|-------------|
| **☁️ Cloud Deployment** | Deploy on AWS/GCP/Azure with REST API endpoints for hospital integration |
| **⌚ IoT Wearable Integration** | Connect with smartwatches for real-time health data streaming |
| **📊 Real-Time Monitoring** | Live dashboard with streaming vitals and alert systems |
| **📱 Mobile Health App** | Flutter/React Native app for on-the-go health assessments |
| **🔒 Data Privacy** | HIPAA-compliant data encryption and secure patient records |
| **🧬 Multi-Disease Expansion** | Add lung cancer, kidney disease, and stroke prediction |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core programming language |
| Scikit-learn | ML models (SVM, RF, LR) |
| XGBoost | Gradient boosting classifier |
| TensorFlow/Keras | Deep learning (CNN, VGG16) |
| SHAP | Explainable AI |
| Streamlit | Web application framework |
| Plotly | Interactive data visualization |
| Pandas/NumPy | Data processing |
| Matplotlib/Seaborn | Static plots |

---

## ⚕️ Disclaimer

This system is developed for **educational and research purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

---

<p align="center">
  Made with ❤️ for B.Tech Final Year Project
</p>
