            checks.append({
                'Metric': info['name'],
                'Sample Input': f"{v:.2f} {info['unit']}",
                'Project Reference Range': f"{info['min']} - {info['max']} {info['unit']}",
                'Status': status
            })

        with c_res1:
            # Show the overall screening outcome first. The ML classification
            # remains visible separately and is never presented as a safety certificate.
            if out_of_range:
                st.markdown('<div class="unsafe-box">NOT CLEARED FOR DRINKING<br><span style="font-size:0.9rem; font-weight:normal;">Further laboratory testing required</span></div>', unsafe_allow_html=True)
                st.error("One or more parameters are outside this project's configured reference ranges. Do not treat this sample as safe to drink.")
                st.markdown("**Parameters outside range:** " + ", ".join(out_of_range))
            else:
                st.markdown('<div class="safe-box">NO OUT-OF-RANGE PARAMETERS DETECTED<br><span style="font-size:0.9rem; font-weight:normal;">This is not a drinking-water safety certification</span></div>', unsafe_allow_html=True)
                st.info("All entered parameters are within this project's configured reference ranges. Laboratory testing is still required to establish drinking-water safety.")

            st.markdown("#### Separate ML model result")
            if pred == 1:
                st.success("ML model prediction: POTABLE")
                if potable_probability is not None:
                    st.markdown(f"**Model prediction confidence:** `{potable_probability * 100:.1f}%`")
                    st.progress(potable_probability)
            else:
                st.warning("ML model prediction: NON-POTABLE")
                if non_potable_probability is not None:
                    st.markdown(f"**Model prediction confidence:** `{non_potable_probability * 100:.1f}%`")
                    st.progress(non_potable_probability)

        with c_res2:
            st.markdown("##### Parameter Reference-Range Screening:")
            st.caption("These are the reference ranges configured in this project. They are screening checks, not a complete official drinking-water certification standard.")
            st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

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
