# Medical Diagnostic AI Dashboard 🩺

[cite_start]An advanced, production-ready healthcare analytics dashboard that utilizes machine learning pipelines to predict the likelihood of diseases based on clinical features[cite: 4, 44, 45]. [cite_start]Built during my Machine Learning Internship at CodeAlpha, this project evaluates multiple classification architectures side-by-side, optimizing specifically for high **Recall** to minimize false negatives in medical diagnostics[cite: 1, 3, 7].

## 🚀 Live Web Dashboard Preview
The application deploys a polished, high-UI/UX dark-themed Streamlit interface featuring three integrated modules:
1. [cite_start]**Batch Prediction Engine:** Drag-and-drop zone to upload patient clinical data (CSV) and download comprehensive diagnostic reports[cite: 55].
2. [cite_start]**Model Analytics:** Live extraction of feature importances directly from trained pipeline architectures[cite: 7].
3. **Single Patient Simulator:** Interactive multi-indicator sliders allowing clinicians to simulate shifts in risk probability profiles.

---

## 🛠️ System Architecture & Engineering

[cite_start]To ensure production-grade reliability, the codebase is built around **Scikit-Learn Pipelines**[cite: 6]. [cite_start]This standardizes feature engineering across training and testing data splits, completely mitigating data leakage[cite: 7].

### Core Technologies
* [cite_start]**Frameworks:** Python, Streamlit, Scikit-Learn, XGBoost, Joblib[cite: 6].
* **Visualizations:** Plotly Express (Interactive charts).
* [cite_start]**Optimization:** `GridSearchCV` implementing a 5-Fold Cross-Validation strategy focused heavily on the **Recall / Sensitivity** scoring metric[cite: 7].

### Model Performance Matrix
[cite_start]The models are trained using the industry-standard **Breast Cancer Wisconsin (Diagnostic) Dataset**[cite: 48]. [cite_start]Through automated hyperparameter tuning, the final production model (XGBoost) achieves high precision and clinical sensitivity[cite: 7, 47]:

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
