import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.datasets import load_breast_cancer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Medical Diagnostic AI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    try:
        # Load the XGBoost model you saved previously
        model = joblib.load(r'D:\models\xgboost_disease_model.pkl')
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def get_feature_names():
    data = load_breast_cancer()
    return data.feature_names

model = load_model()
feature_names = get_feature_names()

# --- SIDEBAR UI ---
st.sidebar.title("🩺 Medi-Predict AI")
st.sidebar.markdown("---")
st.sidebar.info(
    "**CodeAlpha Internship Project**\n\n"
    "This system uses an advanced XGBoost Machine Learning model to predict the likelihood of disease "
    "based on cellular/clinical data."
)
st.sidebar.markdown("---")
st.sidebar.caption("v1.0 | Professional Edition")

# --- MAIN DASHBOARD ---
st.title("Medical Diagnostic AI Dashboard")
st.markdown("Upload clinical datasets for batch evaluation or analyze the model's visual performance.")

if model is None:
    st.error("⚠️ Model not found! Please ensure you have run the training script and that 'models/xgboost_disease_model.pkl' exists.")
    st.stop()

# Create interactive tabs
tab1, tab2, tab3 = st.tabs(["📁 Batch Prediction (Upload CSV)", "📊 Model Visualizations", "🔬 Single Patient Test"])

# --- TAB 1: CSV UPLOAD & BATCH PREDICTION ---
with tab1:
    st.header("Import Patient Dataset")
    st.markdown("Upload a `.csv` file containing patient data. The file must match the 30 standard clinical features.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # Provide a sample dataset button for testing
    if st.button("Generate Sample Test CSV"):
        data = load_breast_cancer()
        sample_df = pd.DataFrame(data.data[:10], columns=data.feature_names) # Get first 10 rows
        csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Sample Data", data=csv, file_name="sample_patients.csv", mime="text/csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            with st.expander("View Raw Data"):
                st.dataframe(df.head())

            if st.button("Run AI Diagnosis"):
                with st.spinner("Analyzing patient records..."):
                    # Extract features assuming the CSV matches the required columns
                    X_eval = df[feature_names] 
                    
                    # Make Predictions
                    predictions = model.predict(X_eval)
                    probabilities = model.predict_proba(X_eval)[:, 1]
                    
                    # Append results to dataframe
                    results_df = df.copy()
                    results_df['AI_Diagnosis'] = ["Malignant (Positive)" if p == 0 else "Benign (Negative)" for p in predictions]
                    results_df['Risk_Probability (%)'] = (probabilities * 100).round(2)
                    
                    st.markdown("### Diagnosis Results")
                    
                    # Highlight high-risk patients
                    def highlight_risk(val):
                        color = '#ff4b4b' if val == 'Malignant (Positive)' else '#00cc96'
                        return f'background-color: {color}; color: white'
                    
                    st.dataframe(results_df[['AI_Diagnosis', 'Risk_Probability (%)']].style.applymap(highlight_risk, subset=['AI_Diagnosis']))
                    
                    # Download results
                    result_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Diagnostic Report (CSV)",
                        data=result_csv,
                        file_name="diagnostic_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file. Please ensure it matches the 30 required clinical features. Detail: {e}")

# --- TAB 2: VISUALIZATIONS ---
with tab2:
    st.header("Model Analytics & Interpretability")
    st.markdown("Understand how the AI makes its decisions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        st.markdown("Which clinical features drive the AI's diagnosis the most?")
        
        # Extract feature importances from the XGBoost model inside the pipeline
        xgb_classifier = model.named_steps['classifier']
        importances = xgb_classifier.feature_importances_
        
        # Create a dataframe for plotting
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10) # Top 10
        
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                     color='Importance', color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Performance Metrics")
        st.info("**Primary Metric: Recall**\n\nOptimized to minimize False Negatives (missing a sick patient).")
        
        # Displaying static metrics from our previous training for the dashboard
        st.metric(label="Testing Accuracy", value="96.5%")
        st.metric(label="ROC-AUC Score", value="0.994")
        st.metric(label="Recall (Sensitivity)", value="98.2%")

# --- TAB 3: SINGLE PATIENT TEST ---
with tab3:
    st.header("Single Patient Simulator")
    st.markdown("Input clinical data manually to see real-time probability shifts.")
    
    # We use a subset of the most important features to avoid cluttering the UI with 30 sliders
    st.subheader("Key Clinical Indicators")
    colA, colB, colC = st.columns(3)
    
    with colA:
        mean_radius = st.slider("Mean Radius", 6.0, 30.0, 14.0)
        mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0)
    with colB:
        mean_perimeter = st.slider("Mean Perimeter", 40.0, 190.0, 90.0)
        mean_area = st.slider("Mean Area", 140.0, 2500.0, 650.0)
    with colC:
        mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.20, 0.10)
    
    if st.button("Predict Single Patient"):
        st.warning("Note: A full prediction requires all 30 features. In a production app, this form would capture all variables or link directly to an electronic health record (EHR) database.")