# Medical Diagnostic AI Dashboard 🩺

An advanced, production-ready healthcare analytics dashboard that utilizes machine learning pipelines to predict the likelihood of diseases based on clinical features. Built during my Machine Learning Internship at CodeAlpha, this project evaluates multiple classification architectures side-by-side, optimizing specifically for high **Recall** to minimize false negatives in medical diagnostics.

## 🚀 Live Web Dashboard Preview
The application deploys a polished, high-UI/UX dark-themed Streamlit interface featuring three integrated modules:
1. **Batch Prediction Engine:** Drag-and-drop zone to upload patient clinical data (CSV) and download comprehensive diagnostic reports.
2. **Model Analytics:** Live extraction of feature importances directly from trained pipeline architectures.
3. **Single Patient Simulator:** Interactive multi-indicator sliders allowing clinicians to simulate shifts in risk probability profiles.

---

## 🛠️ System Architecture & Engineering

To ensure production-grade reliability, the codebase is built around **Scikit-Learn Pipelines**. This standardizes feature engineering across training and testing data splits, completely mitigating data leakage.

### Core Technologies
* **Frameworks:** Python, Streamlit, Scikit-Learn, XGBoost, Joblib.
* **Visualizations:** Plotly Express (Interactive charts).
* **Optimization:** `GridSearchCV` implementing a 5-Fold Cross-Validation strategy focused heavily on the **Recall / Sensitivity** scoring metric.

### Model Performance Matrix
The models are trained using the industry-standard **Breast Cancer Wisconsin (Diagnostic) Dataset**. Through automated hyperparameter tuning, the final production model (XGBoost) achieves high precision and clinical sensitivity:

| Model | Evaluation Strategy | Optimized Metric | Primary Focus |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 5-Fold CV + Grid Search | **Recall (Sensitivity)** | Minimizing False Negatives |
| **Random Forest** | 5-Fold CV + Grid Search | Recall (Sensitivity) | Feature Ensemble Tree Depth |
| **Logistic Regression** | 5-Fold CV + Grid Search | Recall (Sensitivity) | Baseline Linear Separability |

---

## 📂 Project Directory Structure

```text
CodeAlpha_Disease_Prediction/
│
├── models/                     # Saved binary model artifacts (.pkl)
│   └── xgboost_disease_model.pkl
│
├── src/                        # Model Training Framework
│   └── Disease_Prediction.py   # Complete tuning & preprocessing pipeline
│
├── app.py                      # Streamlit UI/UX Dashboard Application
├── requirements.txt            # System dependencies
└── README.md                   # System documentation
