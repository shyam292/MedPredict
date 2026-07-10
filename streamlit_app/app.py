"""
MedPredict — Streamlit Web Application
=========================================
AI-Based Smart Healthcare System for Early Detection of
Diabetes, Heart Disease, and Alzheimer's Disease.

Features:
- Modern dark-themed glassmorphism dashboard
- Disease prediction with probability gauges
- MRI image upload for Alzheimer's classification
- SHAP-based explainable AI visualizations
- Interactive health statistics

Usage:
    cd MedPredict
    streamlit run streamlit_app/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import matplotlib.pyplot as plt

# ── Path configuration ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & GLOBAL CSS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedPredict — AI Healthcare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    """Inject custom CSS for a premium healthcare dashboard."""
    st.markdown("""
    <style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f4e 30%, #0d1b2a 70%, #0a0e27 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1538 0%, #1a1f4e 50%, #0d1b2a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
        padding: 8px 12px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(99, 102, 241, 0.15);
    }

    /* ── Headings ── */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    h1 {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(15, 23, 72, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 72, 0.8), rgba(30, 41, 82, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 500;
    }

    /* ── Result Cards ── */
    .result-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }

    .result-negative {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* ── Inputs ── */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: rgba(15, 23, 72, 0.6) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ── Divider ── */
    .gradient-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #6366f1, #a78bfa, transparent);
        border: none;
        margin: 20px 0;
        border-radius: 2px;
    }

    /* ── General text ── */
    p, li, span, label, .stMarkdown {
        color: #cbd5e1 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 72, 0.5);
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #94a3b8;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeInUp 0.6s ease forwards;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 72, 0.4);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """Load a pickled or h5 model."""
    if model_path.endswith(".h5"):
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(model_path)
    return joblib.load(model_path)


def create_gauge_chart(value: float, title: str, color_scheme: str = "risk") -> go.Figure:
    """Create a probability gauge chart using Plotly."""
    if color_scheme == "risk":
        bar_color = "#ef4444" if value > 0.5 else "#22c55e"
        steps = [
            {"range": [0, 30], "color": "rgba(34, 197, 94, 0.15)"},
            {"range": [30, 60], "color": "rgba(250, 204, 21, 0.15)"},
            {"range": [60, 100], "color": "rgba(239, 68, 68, 0.15)"},
        ]
    else:
        bar_color = "#6366f1"
        steps = [{"range": [0, 100], "color": "rgba(99, 102, 241, 0.1)"}]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#e2e8f0"}},
        title={"text": title, "font": {"size": 16, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(15, 23, 72, 0.3)",
            "borderwidth": 1,
            "bordercolor": "rgba(99, 102, 241, 0.3)",
            "steps": steps,
            "threshold": {
                "line": {"color": "#f59e0b", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def render_shap_explanation(model, scaler, feature_names: list, input_data: np.ndarray):
    """Render SHAP feature importance for a prediction."""
    try:
        from src.utils.explainability import get_shap_explainer, explain_single_prediction

        model_type = "tree" if hasattr(model, "feature_importances_") else "kernel"

        # Create a small background dataset
        rng = np.random.RandomState(42)
        background = rng.randn(50, len(feature_names))

        explainer = get_shap_explainer(model, background, model_type=model_type)
        contributions = explain_single_prediction(explainer, input_data, feature_names)

        # Create bar chart
        features = list(contributions.keys())[:8]
        values = [contributions[f] for f in features]
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
            marker_line_color="rgba(99, 102, 241, 0.3)",
            marker_line_width=1,
        ))

        fig.update_layout(
            title={"text": "🔍 Feature Contributions (SHAP)", "font": {"color": "#e2e8f0", "size": 16}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="SHAP Value", color="#94a3b8", gridcolor="rgba(99, 102, 241, 0.1)"),
            yaxis=dict(color="#94a3b8"),
            height=350,
            margin=dict(l=10, r=10, t=50, b=30),
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"ℹ️ SHAP analysis not available: {str(e)[:100]}")


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════

def page_home():
    """Render the home / dashboard page."""

    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0;">🏥 MedPredict</h1>
        <p style="font-size: 1.3rem; color: #94a3b8 !important; margin-top: 8px;">
            AI-Powered Smart Healthcare System for Early Disease Detection
        </p>
        <div class="gradient-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("3", "Disease Models", "🧬"),
        ("8+", "ML Algorithms", "🤖"),
        ("95%+", "Avg Accuracy", "🎯"),
        ("XAI", "Explainability", "🔍"),
    ]
    for col, (val, label, icon) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disease Cards
    st.markdown("## 🩺 Supported Disease Predictions")
    col1, col2, col3 = st.columns(3)

    diseases = [
        {
            "icon": "💉", "name": "Diabetes",
            "desc": "Predict diabetes risk using blood glucose, BMI, insulin levels, and family history.",
            "models": "SVM • Random Forest • XGBoost",
            "color": "#6366f1",
        },
        {
            "icon": "❤️", "name": "Heart Disease",
            "desc": "Assess cardiovascular disease risk using ECG, cholesterol, blood pressure, and more.",
            "models": "Logistic Regression • RF • XGBoost",
            "color": "#ec4899",
        },
        {
            "icon": "🧠", "name": "Alzheimer's",
            "desc": "Classify brain MRI scans into 4 stages from healthy to Alzheimer's disease.",
            "models": "Custom CNN • VGG16 Transfer Learning",
            "color": "#8b5cf6",
        },
    ]

    for col, d in zip([col1, col2, col3], diseases):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid {d['color']};">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">{d['icon']}</div>
                <h3 style="margin: 0; font-size: 1.3rem;">{d['name']}</h3>
                <p style="font-size: 0.9rem; margin: 10px 0;">{d['desc']}</p>
                <p style="font-size: 0.8rem; color: #6366f1 !important; font-weight: 600;">
                    {d['models']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture Diagram
    st.markdown("## 🏗️ System Architecture")
    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; text-align: center;">
            <div>
                <div style="font-size: 2rem;">📊</div>
                <p style="font-weight: 600; color: #a78bfa !important;">Data Pipeline</p>
                <p style="font-size: 0.8rem;">Collection → Cleaning<br/>→ Feature Engineering</p>
            </div>
            <div style="font-size: 2rem; color: #6366f1; align-self: center;">→</div>
            <div>
                <div style="font-size: 2rem;">🤖</div>
                <p style="font-weight: 600; color: #a78bfa !important;">ML Pipeline</p>
                <p style="font-size: 0.8rem;">Training → Evaluation<br/>→ Model Selection</p>
            </div>
            <div style="font-size: 2rem; color: #6366f1; align-self: center;">→</div>
            <div>
                <div style="font-size: 2rem;">🔍</div>
                <p style="font-weight: 600; color: #a78bfa !important;">XAI Engine</p>
                <p style="font-size: 0.8rem;">SHAP → Feature<br/>Importance</p>
            </div>
            <div style="font-size: 2rem; color: #6366f1; align-self: center;">→</div>
            <div>
                <div style="font-size: 2rem;">🌐</div>
                <p style="font-weight: 600; color: #a78bfa !important;">Web Dashboard</p>
                <p style="font-size: 0.8rem;">Streamlit → Prediction<br/>→ Visualization</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DIABETES PREDICTION
# ══════════════════════════════════════════════════════════════

def page_diabetes():
    """Render the Diabetes prediction page."""
    st.markdown("""
    <div style="text-align: center;">
        <h1>💉 Diabetes Risk Prediction</h1>
        <p style="color: #94a3b8 !important;">Enter your health parameters below to assess diabetes risk</p>
        <div class="gradient-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    st.markdown("### 📋 Health Parameters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        insulin = st.number_input("Insulin (μU/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)

    with col2:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)

    with col3:
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

    with col4:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    if st.button("🔬 Predict Diabetes Risk", use_container_width=True):
        model_path = os.path.join(SAVED_MODELS_DIR, "diabetes_model.pkl")
        scaler_path = os.path.join(SAVED_MODELS_DIR, "diabetes_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("⚠️ Model not found! Please train the model first by running:\n\n`python src/training/train_diabetes_model.py`")
            return

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Prepare input
        input_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                     insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_features)

        # Predict
        prediction = model.predict(input_scaled)[0]
        try:
            probability = model.predict_proba(input_scaled)[0]
            risk_prob = probability[1]  # Probability of diabetes
        except AttributeError:
            risk_prob = float(prediction)

        st.markdown("<br>", unsafe_allow_html=True)

        # Results
        col_result, col_gauge = st.columns([1, 1])

        with col_gauge:
            fig = create_gauge_chart(risk_prob, "Diabetes Risk Score")
            st.plotly_chart(fig, use_container_width=True)

        with col_result:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2 style="color: #ef4444 !important; margin-top: 0;">⚠️ High Risk Detected</h2>
                    <p style="font-size: 1.1rem;">The model predicts a <strong>positive risk</strong> of diabetes
                    with <strong>{risk_prob*100:.1f}%</strong> confidence.</p>
                    <hr style="border-color: rgba(239, 68, 68, 0.2);">
                    <p style="font-size: 0.9rem;"><strong>Recommendations:</strong></p>
                    <ul>
                        <li>Consult an endocrinologist immediately</li>
                        <li>Monitor blood glucose levels regularly</li>
                        <li>Adopt a balanced diet low in refined sugars</li>
                        <li>Increase physical activity (30 min/day)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2 style="color: #22c55e !important; margin-top: 0;">✅ Low Risk</h2>
                    <p style="font-size: 1.1rem;">The model predicts a <strong>low risk</strong> of diabetes
                    with <strong>{(1-risk_prob)*100:.1f}%</strong> confidence.</p>
                    <hr style="border-color: rgba(34, 197, 94, 0.2);">
                    <p style="font-size: 0.9rem;"><strong>Suggestions:</strong></p>
                    <ul>
                        <li>Continue regular health check-ups</li>
                        <li>Maintain a healthy BMI</li>
                        <li>Stay physically active</li>
                        <li>Limit processed food intake</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        # SHAP Explanation
        st.markdown("### 🔍 Explainable AI — Feature Contributions")
        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        render_shap_explanation(model, scaler, feature_names, input_scaled)


# ══════════════════════════════════════════════════════════════
# PAGE: HEART DISEASE PREDICTION
# ══════════════════════════════════════════════════════════════

def page_heart():
    """Render the Heart Disease prediction page."""
    st.markdown("""
    <div style="text-align: center;">
        <h1>❤️ Heart Disease Risk Prediction</h1>
        <p style="color: #94a3b8 !important;">Enter your cardiovascular parameters for risk assessment</p>
        <div class="gradient-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Cardiovascular Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1, key="heart_age")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130, step=1)
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Other"][x])

    with col2:
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: ["Female", "Male"][x])
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240, step=1)
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150, step=1)
        slope = st.selectbox("Slope of Peak ST Segment", options=[0, 1, 2],
                              format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    with col3:
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                           format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                    "Non-Anginal", "Asymptomatic"][x])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1],
                            format_func=lambda x: ["No", "Yes"][x])
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                              format_func=lambda x: ["No", "Yes"][x])
        ca = st.number_input("Number of Major Vessels (CA)", min_value=0, max_value=4, value=0, step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔬 Predict Heart Disease Risk", use_container_width=True):
        model_path = os.path.join(SAVED_MODELS_DIR, "heart_model.pkl")
        scaler_path = os.path.join(SAVED_MODELS_DIR, "heart_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("⚠️ Model not found! Please train the model first by running:\n\n`python src/training/train_heart_model.py`")
            return

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        input_features = np.array([[age, sex, cp, trestbps, chol, fbs,
                                     restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_features)

        prediction = model.predict(input_scaled)[0]
        try:
            probability = model.predict_proba(input_scaled)[0]
            risk_prob = probability[1]
        except AttributeError:
            risk_prob = float(prediction)

        st.markdown("<br>", unsafe_allow_html=True)

        col_result, col_gauge = st.columns([1, 1])

        with col_gauge:
            fig = create_gauge_chart(risk_prob, "Heart Disease Risk Score")
            st.plotly_chart(fig, use_container_width=True)

        with col_result:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2 style="color: #ef4444 !important; margin-top: 0;">⚠️ High Risk Detected</h2>
                    <p style="font-size: 1.1rem;">The model predicts a <strong>positive risk</strong> of heart disease
                    with <strong>{risk_prob*100:.1f}%</strong> confidence.</p>
                    <hr style="border-color: rgba(239, 68, 68, 0.2);">
                    <p style="font-size: 0.9rem;"><strong>Recommendations:</strong></p>
                    <ul>
                        <li>Visit a cardiologist for detailed assessment</li>
                        <li>Monitor blood pressure and cholesterol</li>
                        <li>Adopt a heart-healthy diet (low sodium, low fat)</li>
                        <li>Quit smoking and limit alcohol intake</li>
                        <li>Exercise regularly with doctor's clearance</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2 style="color: #22c55e !important; margin-top: 0;">✅ Low Risk</h2>
                    <p style="font-size: 1.1rem;">The model predicts a <strong>low risk</strong> of heart disease
                    with <strong>{(1-risk_prob)*100:.1f}%</strong> confidence.</p>
                    <hr style="border-color: rgba(34, 197, 94, 0.2);">
                    <p style="font-size: 0.9rem;"><strong>Suggestions:</strong></p>
                    <ul>
                        <li>Continue annual cardiovascular screenings</li>
                        <li>Maintain healthy cholesterol levels</li>
                        <li>Keep blood pressure under control</li>
                        <li>Stay active and manage stress</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        # SHAP Explanation
        st.markdown("### 🔍 Explainable AI — Feature Contributions")
        feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                         "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        render_shap_explanation(model, scaler, feature_names, input_scaled)


# ══════════════════════════════════════════════════════════════
# PAGE: ALZHEIMER'S PREDICTION
# ══════════════════════════════════════════════════════════════

def page_alzheimer():
    """Render the Alzheimer's Disease prediction page."""
    st.markdown("""
    <div style="text-align: center;">
        <h1>🧠 Alzheimer's Disease Classification</h1>
        <p style="color: #94a3b8 !important;">Upload a brain MRI scan for AI-powered classification</p>
        <div class="gradient-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    CLASS_NAMES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
    CLASS_COLORS = ["#22c55e", "#f59e0b", "#f97316", "#ef4444"]
    CLASS_ICONS = ["✅", "⚠️", "🔶", "🔴"]

    # Upload section
    st.markdown("### 📤 Upload MRI Brain Scan")
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image (JPG, PNG, or JPEG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan image for Alzheimer's classification."
    )

    if uploaded_file is not None:
        from PIL import Image

        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.markdown("#### 🖼️ Uploaded MRI Scan")
            st.image(image, use_container_width=True)

        with col_result:
            model_path = os.path.join(SAVED_MODELS_DIR, "alzheimer_model.h5")

            if not os.path.exists(model_path):
                st.error("⚠️ Model not found! Please train the Alzheimer's model first:\n\n`python src/training/train_alzheimer_model.py`")
                return

            with st.spinner("🔄 Analyzing MRI scan..."):
                from src.preprocessing.alzheimer_preprocessing import preprocess_single_image
                model = load_model(model_path)

                # Preprocess the image
                processed_img = preprocess_single_image(image)
                predictions = model.predict(processed_img)[0]

                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]

            # Display result
            class_name = CLASS_NAMES[predicted_class]
            class_color = CLASS_COLORS[predicted_class]
            class_icon = CLASS_ICONS[predicted_class]

            severity = "Low" if predicted_class == 0 else "Moderate" if predicted_class <= 1 else "High" if predicted_class <= 2 else "Very High"

            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {class_color}; text-align: center;">
                <div style="font-size: 3rem;">{class_icon}</div>
                <h2 style="color: {class_color} !important; margin: 10px 0;">{class_name}</h2>
                <p style="font-size: 1.2rem;">Confidence: <strong>{confidence*100:.1f}%</strong></p>
                <p style="font-size: 0.9rem; color: #94a3b8 !important;">Severity Level: {severity}</p>
            </div>
            """, unsafe_allow_html=True)

        # Class probability chart
        st.markdown("### 📊 Classification Probabilities")
        fig = go.Figure(go.Bar(
            x=predictions * 100,
            y=CLASS_NAMES,
            orientation="h",
            marker_color=CLASS_COLORS,
            marker_line_color="rgba(99, 102, 241, 0.3)",
            marker_line_width=1,
            text=[f"{p*100:.1f}%" for p in predictions],
            textposition="auto",
            textfont=dict(color="#e2e8f0"),
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Probability (%)", color="#94a3b8", gridcolor="rgba(99, 102, 241, 0.1)"),
            yaxis=dict(color="#94a3b8"),
            height=300,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("### 💡 Clinical Recommendations")
        recommendations = {
            0: [
                "No signs of dementia detected",
                "Continue regular cognitive health screenings",
                "Maintain an active lifestyle with mental exercises",
                "Follow a Mediterranean-style diet for brain health",
            ],
            1: [
                "Very mild cognitive impairment detected",
                "Schedule a detailed neurological evaluation",
                "Consider cognitive behavioral therapy",
                "Monitor memory and cognitive function regularly",
                "Stay socially engaged and mentally active",
            ],
            2: [
                "Mild cognitive impairment detected",
                "Seek immediate neurologist consultation",
                "Begin structured cognitive rehabilitation",
                "Evaluate medication options with your doctor",
                "Arrange support systems for daily activities",
            ],
            3: [
                "Moderate Alzheimer's indicators detected",
                "Urgent neurological assessment recommended",
                "Discuss treatment plans with a specialist team",
                "Plan for comprehensive caregiving support",
                "Explore clinical trial opportunities",
            ],
        }

        for rec in recommendations[predicted_class]:
            st.markdown(f"- {rec}")


# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════

def page_about():
    """Render the About page."""
    st.markdown("""
    <div style="text-align: center;">
        <h1>📖 About MedPredict</h1>
        <p style="color: #94a3b8 !important;">B.Tech Final Year Project</p>
        <div class="gradient-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Project Overview</h3>
        <p>
            <strong>MedPredict</strong> is an AI-based smart healthcare system designed for the
            early detection of <strong>Diabetes</strong>, <strong>Heart Disease</strong>, and
            <strong>Alzheimer's Disease</strong> using state-of-the-art machine learning algorithms.
        </p>
        <p>
            The system processes both structured tabular data (blood tests, vital signs) and
            unstructured medical imaging data (brain MRI scans) to provide accurate risk predictions
            with explainable AI insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech Stack
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🛠️ Technology Stack</h3>
            <ul>
                <li><strong>Python 3.10+</strong> — Core language</li>
                <li><strong>Scikit-learn</strong> — ML models (SVM, RF, LR)</li>
                <li><strong>XGBoost</strong> — Gradient boosting</li>
                <li><strong>TensorFlow/Keras</strong> — Deep learning (CNN, VGG16)</li>
                <li><strong>SHAP</strong> — Explainable AI</li>
                <li><strong>Streamlit</strong> — Web dashboard</li>
                <li><strong>Plotly</strong> — Interactive charts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Datasets Used</h3>
            <ul>
                <li><strong>Pima Indians Diabetes</strong> — 768 samples, 8 features</li>
                <li><strong>UCI Heart Disease</strong> — 303 samples, 13 features</li>
                <li><strong>Alzheimer MRI Dataset</strong> — 6,400 brain MRI images</li>
            </ul>
            <h3 style="margin-top: 20px;">📈 ML Algorithms</h3>
            <ul>
                <li>Support Vector Machine (SVM)</li>
                <li>Logistic Regression</li>
                <li>Random Forest</li>
                <li>XGBoost (Gradient Boosting)</li>
                <li>Custom CNN</li>
                <li>VGG16 Transfer Learning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Future Improvements
    st.markdown("""
    <div class="glass-card">
        <h3>🚀 Future Improvements</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div>
                <h4 style="color: #a78bfa !important;">🌐 Cloud Deployment</h4>
                <p style="font-size: 0.9rem;">Deploy on AWS/GCP/Azure with REST API endpoints
                for hospital integration and remote diagnostics.</p>
            </div>
            <div>
                <h4 style="color: #a78bfa !important;">⌚ IoT Wearable Integration</h4>
                <p style="font-size: 0.9rem;">Connect with smartwatches and fitness bands for
                real-time health data streaming and continuous monitoring.</p>
            </div>
            <div>
                <h4 style="color: #a78bfa !important;">📊 Real-Time Monitoring</h4>
                <p style="font-size: 0.9rem;">Build a live dashboard with streaming data
                visualization and alert systems for critical health events.</p>
            </div>
            <div>
                <h4 style="color: #a78bfa !important;">📱 Mobile Health App</h4>
                <p style="font-size: 0.9rem;">Develop a Flutter/React Native mobile app for
                on-the-go health assessments and push notification alerts.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    inject_css()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 2.5rem;">🏥</div>
            <h2 style="font-size: 1.4rem; margin: 8px 0;">MedPredict</h2>
            <p style="font-size: 0.8rem; color: #94a3b8 !important;">AI Healthcare System</p>
            <div class="gradient-divider"></div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=["🏠 Home", "💉 Diabetes", "❤️ Heart Disease", "🧠 Alzheimer's", "📖 About"],
            label_visibility="collapsed",
        )

        st.markdown("<br>" * 3, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; padding: 10px; background: rgba(99, 102, 241, 0.1);
                    border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
            <p style="font-size: 0.75rem; margin: 0; color: #94a3b8 !important;">
                ⚕️ Medical Disclaimer<br>
                <span style="font-size: 0.7rem;">This tool is for educational purposes only.
                Always consult a qualified healthcare provider.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Route to selected page
    if page == "🏠 Home":
        page_home()
    elif page == "💉 Diabetes":
        page_diabetes()
    elif page == "❤️ Heart Disease":
        page_heart()
    elif page == "🧠 Alzheimer's":
        page_alzheimer()
    elif page == "📖 About":
        page_about()


if __name__ == "__main__":
    main()
