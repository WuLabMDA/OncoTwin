"""
=============================================================================
Demo: Model Construction — XGBoost-AFT Survival Model
=============================================================================
Description:
    This script demonstrates how to construct and train an XGBoost-AFT
    (Accelerated Failure Time) survival model for PFS (Progression-Free
    Survival) prediction using volumetric measurements + clinical features.

    Steps:
      1. Load and preprocess training and external test cohort data
      2. Feature engineering: combine pre/post/delta volumetric and clinical features
      3. Remove highly correlated features (utils: remove_highly_correlated_features)
      4. Format survival labels as structured arrays
      5. Train XGBoost-AFT model with GPU acceleration
      6. Evaluate concordance index (C-index) on training and test sets
      7. Compute and visualize SHAP feature importance

Dependencies:
    numpy, pandas, xgboost, xgbse, lifelines, shap, sklearn, matplotlib
    utils_survival.py  (project utility functions)

Usage:
    python demo1_model_construction.py
    # or run cell-by-cell in a Jupyter notebook

Inputs:  (replace paths with your own data directories)
    - VolOriginal_Base_overall_MDACC.xlsx  : overall features including clinical, blood test, CT-based tumor volumetrics measuremnets
    (Same structure required for the external validation cohort.)

Outputs:
    - results/<work_name>/xgb_model.json          : saved XGBoost model
    - results/<work_name>/*_feature_importance.*  : gain-based importance
    - results/<work_name>/SHAP_summary_*.png      : SHAP beeswarm plot
    - results/<work_name>/SHAP_importance_*.csv   : SHAP importance table
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. Environment check (optional)
# ---------------------------------------------------------------------------
import numpy; import pandas; import sklearn; import xgboost; import lifelines
print("numpy    :", numpy.__version__)
print("pandas   :", pandas.__version__)
print("sklearn  :", sklearn.__version__)
print("xgboost  :", xgboost.__version__)
print("lifelines:", lifelines.__version__)


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
from xgbse.metrics import concordance_index as xgbse_cindex
from xgbse.converters import convert_data_to_xgb_format

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")
plt.style.use("bmh")
np.random.seed(42)

# Project utility functions (see utils_survival.py)
from utils_survival import remove_highly_correlated_features, safe_shap_summary_plot


# ---------------------------------------------------------------------------
# 2. Output directory
# ---------------------------------------------------------------------------
work_name = "model_output"
path_name = os.path.join("results", work_name)
os.makedirs(path_name, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. XGBoost-AFT hyperparameters
# ---------------------------------------------------------------------------
PARAMS_XGB_AFT = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.0,   # controls predicted survival months
    "tree_method": "hist",
    "device": "cuda",                      # change to "cpu" if no GPU available
    "learning_rate": 5e-3,
    "max_depth": 8,
    "booster": "dart",
    "subsample": 0.5,
    "min_child_weight": 7,
    "colsample_bynode": 0.7,
}
NUM_BOOST_ROUND = 1000


# ---------------------------------------------------------------------------
# 4. Data loading — Training cohort
# ---------------------------------------------------------------------------
# NOTE: Replace 'data/train/' with your actual data directory.
train_data_dir = "data/train"

df_train = pd.read_excel(os.path.join(train_data_dir, "VolOriginal_Base_overall_MDACC.xlsx"))
print("Training cohort shape:", df_train.shape)

# ---------------------------------------------------------------------------
# 5. Feature engineering — Training cohort
# ---------------------------------------------------------------------------
# --- Categorical encoding ---
df_train["Sex"]       = df_train["Sex"].map({"F": 0, "M": 1})
df_train["Race"]      = df_train["Race"].map({"White": 0, "Others": 1})
df_train["Ethnicity"] = df_train["Ethnicity"].map({"Non-Hispanic": 0, "Hispanic": 1})
df_train["Smoker"]    = df_train["Smoker"].map({"No": 0, "Yes": 1})
df_train["Pathology"] = df_train["Pathology"].map({"ADC": 0, "Others": 1})
df_train["RECIST"] = df_train["RECIST"].map({"CR": 0, "PR": 1, "SD": 2, "PD": 3})

# --- Volumetric feature blocks (adjust column ranges to match your data) ---
x_radio = df_train.loc[:, "TV_Lung":"ATS_AllTumor"].copy()

# Remove highly correlated radiomic features (threshold = 0.98)
x_radio, removed_features = remove_highly_correlated_features(x_radio, threshold=0.98, plot=True)
print(f"Removed {len(removed_features)} highly correlated features.")

# --- Clinical feature blocks ---
cat_cols = ["Sex", "Smoker", "Race", "Ethnicity", "Pathology", "Drug", "RECIST"]
cont_cols = [
    "Age",
    "WBC_1", "RBC_1", "HGB_1", "MCV_1", "RDW_1", "PLT_1", "NEUT_1", "NEUT%_1",
    "LYM_1", "NLR_1", "Na_1", "K_1", "CL_1", "CO2_1", "BUN_1", "CREAT_1",
    "GLU_1", "CA_1", "ALT_1", "AST_1", "ALP_1", "DBILI_1", "TBILI_1", "ALB_1", "PROT_1",
    # ... Replace lab test with your actual data 
]
clin_cols = cat_cols + cont_cols

x_train = pd.concat([x_radio, df_train[clin_cols]], axis=1)
feat_names = x_train.columns.tolist()
print("Feature matrix shape:", x_train.shape)

# --- Survival label (structured array) ---
y_train = np.zeros(
    len(df_train),
    dtype=[("PFS_events", "bool"), ("PFS_time", "f8")]
)
y_train["PFS_events"] = df_train["PFS_events"].tolist()
y_train["PFS_time"]   = df_train["PFS"].tolist()

# ---------------------------------------------------------------------------
# 6. Data loading — External test cohort (same preprocessing pipeline)
# ---------------------------------------------------------------------------
# NOTE: Replace 'data/train/' with your actual data directory.
test_data_dir = "data/test"
df_test = pd.read_excel(os.path.join(test_data_dir, "VolOriginal_Base_overall_BrightStar.xlsx"))
print("Testing cohort shape:", df_test.shape)

# Same categorical encoding
df_test["Sex"]       = df_test["Sex"].map({"F": 0, "M": 1})
df_test["Race"]      = df_test["Race"].map({"White": 0, "Others": 1})
df_test["Ethnicity"] = df_test["Ethnicity"].map({"Non-Hispanic": 0, "Hispanic": 1})
df_test["Smoker"]    = df_test["Smoker"].map({"No": 0, "Yes": 1})
df_test["Pathology"] = df_test["Pathology"].map({"ADC": 0, "Others": 1})
df_test["RECIST_before"] = df_test["RECIST_before"].map({"CR": 0, "PR": 1, "SD": 2, "PD": 3})

x_test_radio = df_test.loc[:, "TV_Lung":"ATS_AllTumor"].copy()
x_test_radio = x_test_radio.drop(columns=removed_features)  # apply same feature mask

x_test = pd.concat([x_test_radio, df_test[clin_cols]], axis=1)
x_test = pd.DataFrame(x_test, columns=feat_names)

# --- Survival label (structured array) for Test ---
y_test = np.zeros(
    len(df_test),
    dtype=[("PFS_events", "bool"), ("PFS_time", "f8")]
)
y_test["PFS_events"] = df_test["PFS_events"].tolist()
y_test["PFS_time"]   = df_test["PFS"].tolist()


# ---------------------------------------------------------------------------
# 7. Convert to XGBoost DMatrix format
# ---------------------------------------------------------------------------
dtrain = convert_data_to_xgb_format(x_train, y_train, "survival:aft")
dtest  = convert_data_to_xgb_format(x_test,  y_test,  "survival:aft")

# ---------------------------------------------------------------------------
# 8. Train XGBoost-AFT model
# ---------------------------------------------------------------------------
print("Training XGBoost-AFT model...")
bst = xgb.train(
    PARAMS_XGB_AFT,
    dtrain,
    num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=10,
    evals=[(dtrain, "val")],
    verbose_eval=False,
)

# Save model
model_path = os.path.join(path_name, "xgb_model.json")
bst.save_model(model_path)
print(f"Model saved to: {model_path}")

# ---------------------------------------------------------------------------
# 9. Prediction and C-index evaluation
# ---------------------------------------------------------------------------
preds_train = bst.predict(dtrain)
preds_test  = bst.predict(dtest)

train_ci = xgbse_cindex(y_train, -preds_train, risk_strategy="precomputed")
test_ci  = xgbse_cindex(y_test,  -preds_test,  risk_strategy="precomputed")
print(f"Training C-index : {train_ci:.4f}")
print(f"Test C-index     : {test_ci:.4f}")

# Normalize predictions (fit on train, transform on test)
scaler = StandardScaler()
preds_train_norm = scaler.fit_transform(preds_train.reshape(-1, 1)).ravel()
preds_test_norm  = scaler.transform(preds_test.reshape(-1, 1)).ravel()

# ---------------------------------------------------------------------------
# 10. Feature importance (gain-based)
# ---------------------------------------------------------------------------
feat_importance = bst.get_score(importance_type="gain")
sorted_imp = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
features_sorted, scores_sorted = zip(*sorted_imp)
df_importance = pd.DataFrame({"Feature": features_sorted, "Importance": scores_sorted})
df_importance.to_csv(os.path.join(path_name, "feature_importance_gain.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.barh(list(features_sorted)[:20], list(scores_sorted)[:20], color="skyblue")
plt.title("Feature Importance (Gain) — Top 20", fontsize=16)
plt.xlabel("Gain", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(path_name, "feature_importance_gain.png"), dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 11. SHAP analysis
# ---------------------------------------------------------------------------
# Load model on CPU for SHAP (avoids GPU memory issues)
cpu_model = xgb.Booster()
cpu_model.load_model(model_path)

safe_shap_summary_plot(
    model=cpu_model,
    x_test_df=x_test,
    feature_names=feat_names,
    save_dir=os.path.join(path_name, "SHAP"),
    fold_name="TestCohort",
    max_samples=500,
    top_k=10,
)

print("Demo 2 complete. Results saved to:", path_name)
