import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Water Quality Prediction Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Bright UI CSS Injection (No Emojis, No Unit 4)
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0284C7;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 1.8rem;
    }
    .safe-box {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        color: #15803D;
        padding: 1.2rem;
        border-radius: 0.75rem;
        font-size: 1.25rem;
        font-weight: 800;
        text-align: center;
    }
    .unsafe-box {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #B91C1C;
        padding: 1.2rem;
        border-radius: 0.75rem;
        font-size: 1.25rem;
        font-weight: 800;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, 'best_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    imputer_path = os.path.join(BASE_DIR, 'imputer.pkl')
    json_path = os.path.join(BASE_DIR, 'model_summary.json')
    data_path = os.path.join(BASE_DIR, 'water_potability.csv')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(imputer_path, 'rb') as f:
        imputer = pickle.load(f)
    with open(json_path, 'r') as f:
        summary = json.load(f)

    df = pd.read_csv(data_path)
    return model, scaler, imputer, summary, df

model, scaler, imputer, summary, df = load_artifacts()

# Sidebar Navigation
st.sidebar.markdown("### Water Quality ML System")
page = st.sidebar.radio("Navigate", [
    "Live Water Predictor",
    "Dataset Explorer (EDA)",
    "Model Benchmarks",
    "Project Documentation"
])

st.sidebar.markdown("---")
st.sidebar.markdown("#### Project Scope")
st.sidebar.caption("""
• Domain: ML & Data Analysis
• Focus: Water Potability Classification
• Target: UN Sustainable Development Goal 6
""")

WHO_LIMITS = {
    'ph': {'min': 6.5, 'max': 8.5, 'unit': 'pH scale', 'name': 'pH Level'},
    'Hardness': {'min': 0, 'max': 250.0, 'unit': 'mg/L', 'name': 'Hardness'},
    'Solids': {'min': 0, 'max': 25000.0, 'unit': 'ppm', 'name': 'Total Dissolved Solids (TDS)'},
    'Chloramines': {'min': 0, 'max': 8.0, 'unit': 'ppm', 'name': 'Chloramines'},
    'Sulfate': {'min': 0, 'max': 350.0, 'unit': 'mg/L', 'name': 'Sulfate'},
    'Conductivity': {'min': 0, 'max': 500.0, 'unit': 'μS/cm', 'name': 'Conductivity'},
    'Organic_carbon': {'min': 0, 'max': 15.0, 'unit': 'ppm', 'name': 'Organic Carbon'},
    'Trihalomethanes': {'min': 0, 'max': 80.0, 'unit': 'μg/L', 'name': 'Trihalomethanes'},
    'Turbidity': {'min': 0, 'max': 4.5, 'unit': 'NTU', 'name': 'Turbidity'}
}

# ==========================================
# PAGE 1: LIVE PREDICTOR
# ==========================================
if page == "Live Water Predictor":
    st.markdown('<div class="main-header">Water Quality Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input physical & chemical sample parameters to test water potability instantly</div>', unsafe_allow_html=True)

    st.markdown("##### Quick Preset Water Samples:")
    col_p1, col_p2, col_p3 = st.columns(3)

    preset_ph, preset_hard, preset_solids, preset_chlor, preset_sulf, preset_cond, preset_oc, preset_thm, preset_turb = 7.2, 185.0, 14000.0, 6.5, 310.0, 410.0, 12.0, 60.0, 3.2

    if col_p1.button("Clean Drinking Tap Water"):
        preset_ph, preset_hard, preset_solids, preset_chlor, preset_sulf, preset_cond, preset_oc, preset_thm, preset_turb = 7.4, 160.0, 12000.0, 5.0, 240.0, 380.0, 9.5, 45.0, 2.8

    if col_p2.button("Toxic Industrial Runoff"):
        preset_ph, preset_hard, preset_solids, preset_chlor, preset_sulf, preset_cond, preset_oc, preset_thm, preset_turb = 4.8, 310.0, 48000.0, 11.2, 450.0, 680.0, 22.5, 110.0, 6.2

    if col_p3.button("Natural Spring Mineral Water"):
        preset_ph, preset_hard, preset_solids, preset_chlor, preset_sulf, preset_cond, preset_oc, preset_thm, preset_turb = 8.1, 230.0, 24000.0, 7.1, 340.0, 490.0, 14.1, 72.0, 4.1

    st.markdown("---")
    st.markdown("#### Chemical Parameter Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        val_ph = st.slider("pH Level", 0.0, 14.0, float(preset_ph), 0.1, help="Safe WHO Range: 6.5 - 8.5")
        val_hard = st.slider("Hardness (mg/L)", 0.0, 400.0, float(preset_hard), 5.0, help="Safe WHO Range: < 250 mg/L")
        val_solids = st.slider("Solids / TDS (ppm)", 0.0, 70000.0, float(preset_solids), 500.0, help="Safe WHO Range: < 25,000 ppm")

    with col2:
        val_chlor = st.slider("Chloramines (ppm)", 0.0, 20.0, float(preset_chlor), 0.1, help="Safe WHO Range: < 8.0 ppm")
        val_sulf = st.slider("Sulfate (mg/L)", 0.0, 600.0, float(preset_sulf), 5.0, help="Safe WHO Range: < 350 mg/L")
        val_cond = st.slider("Conductivity (μS/cm)", 0.0, 900.0, float(preset_cond), 10.0, help="Safe WHO Range: < 500 μS/cm")

    with col3:
        val_oc = st.slider("Organic Carbon (ppm)", 0.0, 35.0, float(preset_oc), 0.5, help="Safe WHO Range: < 15.0 ppm")
        val_thm = st.slider("Trihalomethanes (μg/L)", 0.0, 150.0, float(preset_thm), 1.0, help="Safe WHO Range: < 80 μg/L")
        val_turb = st.slider("Turbidity (NTU)", 0.0, 10.0, float(preset_turb), 0.1, help="Safe WHO Range: < 4.5 NTU")

    input_data = np.array([[val_ph, val_hard, val_solids, val_chlor, val_sulf, val_cond, val_oc, val_thm, val_turb]])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Machine Learning Diagnosis", type="primary"):
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]

        st.markdown("### Prediction Diagnosis Result")
        c_res1, c_res2 = st.columns([1, 1.5])

        with c_res1:
            if pred == 1:
                st.markdown('<div class="safe-box">WATER IS POTABLE<br><span style="font-size:0.9rem; font-weight:normal;">Safe for Human Consumption</span></div>', unsafe_allow_html=True)
                st.markdown(f"**Safety Confidence:** `{prob[1]*100:.1f}%`")
                st.progress(float(prob[1]))
            else:
                st.markdown('<div class="unsafe-box">WATER IS NON-POTABLE<br><span style="font-size:0.9rem; font-weight:normal;">Unsafe / Requires Chemical Treatment</span></div>', unsafe_allow_html=True)
                st.markdown(f"**Contamination Risk Score:** `{prob[0]*100:.1f}%`")
                st.progress(float(prob[0]))

        with c_res2:
            st.markdown("##### WHO Parameter Safety Standards Check:")
            checks = []
            user_vals = {
                'ph': val_ph, 'Hardness': val_hard, 'Solids': val_solids,
                'Chloramines': val_chlor, 'Sulfate': val_sulf, 'Conductivity': val_cond,
                'Organic_carbon': val_oc, 'Trihalomethanes': val_thm, 'Turbidity': val_turb
            }
            for k, info in WHO_LIMITS.items():
                v = user_vals[k]
                is_safe = (v >= info['min']) and (v <= info['max'])
                status = "PASS" if is_safe else "EXCEEDS LIMIT"
                checks.append({
                    'Metric': info['name'],
                    'Sample Input': f"{v:.2f} {info['unit']}",
                    'WHO Limit': f"{info['min']} - {info['max']} {info['unit']}",
                    'Status': status
                })
            st.dataframe(pd.DataFrame(checks))

# ==========================================
# PAGE 2: DATASET EXPLORER
# ==========================================
elif page == "Dataset Explorer (EDA)":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze water sample distributions and chemical feature correlations</div>', unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Water Samples", f"{len(df):,}")
    col_m2.metric("Chemical Metrics", "9 Features")
    col_m3.metric("Potable Samples", f"{(df['Potability']==1).sum():,}", delta=f"{((df['Potability']==1).mean()*100):.1f}%")
    col_m4.metric("Non-Potable Samples", f"{(df['Potability']==0).sum():,}", delta=f"{((df['Potability']==0).mean()*100):.1f}%", delta_color="inverse")

    st.markdown("---")
    t1, t2, t3 = st.tabs(["Feature Distributions", "Correlation Heatmap", "Raw Dataset Table"])

    with t1:
        st.markdown("#### Feature Distribution by Potability Target")
        feat = st.selectbox("Select Parameter:", [c for c in df.columns if c != 'Potability'])
        
        fig, ax = plt.subplots(figsize=(8, 3.8))
        sns.set_theme(style="whitegrid")
        sns.histplot(data=df, x=feat, hue='Potability', kde=True, palette={0: '#EF4444', 1: '#10B981'}, ax=ax)
        ax.set_title(f"Distribution of {feat} (Red = Non-Potable, Green = Potable)", color='#0F172A', fontweight='bold')
        st.pyplot(fig)

    with t2:
        st.markdown("#### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues', ax=ax)
        st.pyplot(fig)

    with t3:
        st.markdown("#### Dataset Statistics")
        st.dataframe(df.describe().T)

# ==========================================
# PAGE 3: MODEL BENCHMARKS
# ==========================================
elif page == "Model Benchmarks":
    st.markdown('<div class="main-header">Model Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics across 7 Machine Learning algorithms</div>', unsafe_allow_html=True)

    metrics_dict = summary['metrics']
    clean_rows = []
    for m_name, m_vals in metrics_dict.items():
        clean_rows.append({
            'Model Name': m_name,
            'Accuracy': m_vals['Accuracy'],
            'Precision': m_vals['Precision'],
            'Recall': m_vals['Recall'],
            'F1-Score': m_vals['F1-Score'],
            'ROC-AUC': m_vals['ROC-AUC']
        })
    metrics_df = pd.DataFrame(clean_rows).set_index('Model Name')

    st.dataframe(metrics_df)

    st.markdown("---")
    cb1, cb2 = st.columns(2)

    with cb1:
        st.markdown("#### F1-Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        metrics_df['F1-Score'].plot(kind='barh', color='#0284C7', ax=ax)
        ax.set_xlabel("F1-Score")
        ax.set_xlim(0, 1.0)
        st.pyplot(fig)

    with cb2:
        st.markdown(f"#### Top Feature Importances ({summary['best_model_name']})")
        fi = pd.Series(summary['feature_importances']).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fi.plot(kind='barh', color='#0D9488', ax=ax)
        ax.set_xlabel("Importance Weight")
        st.pyplot(fig)

# ==========================================
# PAGE 4: DOCUMENTATION
# ==========================================
elif page == "Project Documentation":
    st.markdown('<div class="main-header">Project Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Requirements, environmental alignment, and technical design</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Project Overview
    - **Topic**: Machine Learning-Based Water Quality Prediction
    - **Methodology**: Implemented 7 ML Classifiers (Random Forest, XGBoost, Decision Tree, Gradient Boosting, SVM, ANN, Logistic Regression).
    - **Preprocessing**: Used `KNNImputer` for missing sensor data and `SMOTE` for class imbalance resolution.
    - **Environmental Alignment**: Supports UN Sustainable Development Goal 6 (Clean Water and Sanitation).
    """)
