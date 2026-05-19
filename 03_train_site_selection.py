"""Phase 2, step 3: Train Site Selection Model with interpretability and robustness.

Features: Demographics and OSM proximities.
Baseline: Logistic Regression (L2 penalized).
Core Model: Regularized LightGBM with Grouped CV (by ZIP).
Uncertainty & Explainability: Grouped Bootstrap for prediction intervals and SHAP stability.

Outputs:
- models/lr_site_selection.pkl
- models/lgb_site_selection.txt
- reports/shap_summary.png
"""

import os
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

FEATURES_FILE = "data/processed/site_features.csv"
MODEL_DIR = "models"
REPORTS_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_logistic_baseline(X, y, groups):
    print("\n--- Training Logistic Regression Baseline ---")
    gkf = GroupKFold(n_splits=5)
    pr_aucs = []
    
    # We will just do a simple 5-fold CV to get the PR-AUC
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict_proba(X_test)[:, 1]
        pr_aucs.append(average_precision_score(y_test, y_pred))
        
    print(f"LR Baseline CV PR-AUC: {np.mean(pr_aucs):.4f}")
    
    # Train on full data
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000))
    ])
    pipe.fit(X, y)
    
    # Print coefficients
    coefs = pd.Series(pipe.named_steps['lr'].coef_[0], index=X.columns).sort_values()
    print("Top LR Coefficients:")
    print(coefs.tail(5))
    print("Bottom LR Coefficients:")
    print(coefs.head(5))
    
    joblib.dump(pipe, os.path.join(MODEL_DIR, "lr_site_selection.pkl"))
    return pipe

def train_lightgbm_grouped_cv(X, y, groups):
    print("\n--- Training Regularized LightGBM ---")
    
    # Tight regularized parameters for small data (~150 positives)
    params = {
        "objective": "binary",
        "metric": "average_precision", # fallback metric
        "boosting_type": "gbdt",
        "max_depth": 4,
        "num_leaves": 10,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "learning_rate": 0.05,
        "is_unbalance": True, # handled class imbalance
        "verbose": -1,
        "seed": 42
    }
    
    gkf = GroupKFold(n_splits=5)
    pr_aucs = []
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        pr_aucs.append(average_precision_score(y_test, y_pred))
        
    print(f"LightGBM CV PR-AUC: {np.mean(pr_aucs):.4f}")
    
    # Train full model
    train_data = lgb.Dataset(X, label=y)
    full_model = lgb.train(params, train_data, num_boost_round=50)
    full_model.save_model(os.path.join(MODEL_DIR, "lgb_site_selection.txt"))
    
    return full_model, params

def grouped_bootstrap(X, y, groups, params, B=50):
    print(f"\n--- Running Grouped Bootstrap (B={B}) ---")
    unique_groups = groups.unique()
    
    feature_importances = []
    
    for b in range(B):
        # Resample groups with replacement
        boot_groups = resample(unique_groups, random_state=b)
        
        # Build bootstrap dataset
        boot_idx = []
        for g in boot_groups:
            boot_idx.extend(groups[groups == g].index.tolist())
            
        X_boot = X.loc[boot_idx]
        y_boot = y.loc[boot_idx]
        
        train_data = lgb.Dataset(X_boot, label=y_boot)
        # We use a small number of iterations since we just want the model for SHAP stability
        model = lgb.train(params, train_data, num_boost_round=30)
        
        # Get SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_boot)
        # LightGBM binary classification shap_values can be a list or array depending on version
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
            
        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_importances.append(mean_abs_shap)
        
    print("Bootstrap completed.")
    
    # Analyze SHAP stability
    fi_matrix = np.array(feature_importances) # shape (B, n_features)
    mean_fi = fi_matrix.mean(axis=0)
    
    # Rank features by mean importance
    ranks = pd.Series(mean_fi, index=X.columns).sort_values(ascending=False)
    print("\nTop Features by Bootstrap Mean |SHAP|:")
    print(ranks.head(10))

def main():
    sites = pd.read_csv(FEATURES_FILE)
    
    # define features
    exclude_cols = ['site_id', 'lat', 'lon', 'label', 'source', 'name', 'zip']
    feature_cols = [c for c in sites.columns if c not in exclude_cols]
    
    # Impute remaining NaNs with median just in case
    for col in feature_cols:
        sites[col] = sites[col].fillna(sites[col].median())
        
    X = sites[feature_cols]
    y = sites['label']
    groups = sites['zip']
    
    print(f"Dataset: {len(X)} sites ({y.sum()} positives), {len(feature_cols)} features.")
    
    # 1. LR Baseline
    lr_model = train_logistic_baseline(X, y, groups)
    
    # 2. LightGBM
    lgb_model, lgb_params = train_lightgbm_grouped_cv(X, y, groups)
    
    # 3. Global SHAP explanation
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
        
    plt.figure()
    shap.summary_plot(sv, X, show=False)
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"), bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary to {os.path.join(REPORTS_DIR, 'shap_summary.png')}")
    
    # 4. Grouped Bootstrap for uncertainty/stability
    grouped_bootstrap(X, y, groups, lgb_params, B=30)
    
if __name__ == "__main__":
    main()
