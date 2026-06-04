import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

def load_data():
    """Loads the Breast Cancer dataset from sklearn."""
    print("Loading dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

def build_pipelines():
    """Constructs ML pipelines for different algorithms."""
    pipelines = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=10000, random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
    }
    return pipelines

def get_param_grids():
    """Defines hyperparameter grids for tuning."""
    return {
        'LogisticRegression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        'RandomForest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    }

def main():
    # 1. Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipelines = build_pipelines()
    param_grids = get_param_grids()
    
    best_models = {}
    
    # 2. Train and Tune Models
    print("\nStarting Model Training and Tuning...")
    for model_name in pipelines.keys():
        print(f"\nTraining {model_name}...")
        
        # Using GridSearchCV focusing on 'recall' because this is medical data
        grid_search = GridSearchCV(
            estimator=pipelines[model_name],
            param_grid=param_grids[model_name],
            cv=5,
            scoring='recall', 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    
    # 3. Evaluate Models
    print("\n--- Final Evaluation on Test Set ---")
    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n[{model_name}] Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    # 4. Save the best performing model
    print("\nSaving the XGBoost model for deployment...")
    
    # This automatically creates the 'models' folder if it doesn't exist
    os.makedirs('models', exist_ok=True) 
    
    joblib.dump(best_models['XGBoost'], 'models/xgboost_disease_model.pkl')
    print("Pipeline execution complete. Model saved successfully!")

if __name__ == "__main__":
    main()