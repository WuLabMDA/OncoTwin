"""
=============================================================================
Demo: Model Prediction, Calibration & Digital Twin Prognosis
=============================================================================
Description:
    This script demonstrates post-hoc calibration of XGBoost-AFT predictions
    using drug-stratified Weibull survival calibrators, and
    constructs individual Digital Twin (DT) survival curves.

    The calibration pipeline converts raw risk scores into well-calibrated,
    patient-specific survival probability curves — one per drug arm — and
    supports Individual Treatment Effect (ITE) estimation for LCT
    (Local Consolidative Therapy) vs. no-LCT counterfactual comparisons.

    Steps:
      1. Load trained model and run inference on training and test cohorts
      2. Fit drug-stratified survival calibrators (full_calibration_pipeline)
      3. Evaluate integrated Brier score (IBS) on test cohort
      4. Plot individual OncoTwin-predicted Survival curves
      5. Estimate ITE (ΔRMST) for LCT vs. no-LCT counterfactual comparison
      6. Plot patient-level counterfactual survival curves
      7. Save Digital Twin predictions and ITE to CSV

Dependencies:
    numpy, pandas, xgboost, lifelines, matplotlib, scipy
    utils_survival.py            (project utility functions)
    utils_calibration.py         (calibration pipeline utilities)

Usage:
    python demo3_calibration_digital_twin.py
    # Requires demo2_model_construction.py to have been run first
    # (needs: xgb_model.json, feature matrices, survival labels)

Inputs:
    - results/model_output/xgb_model.json     : trained XGBoost-AFT model
    - x_train, y_train                        : training feature matrix & labels
    - x_test, y_test                          : test feature matrix & labels
    (See demo2_model_construction.py for how these are constructed.)

Outputs:
    - results/<work_name>/Calibration_Train/        : calibration plots (train)
    - results/<work_name>/Calibration_Test/         : calibration plots (test)
    - results/<work_name>/ibs_curve_*.png           : Brier score curves
    - results/<work_name>/patients_*_curves_only.png: individual DT curves
    - results/<work_name>/ITE_Testing_*.csv         : per-patient ITE table
    - results/<work_name>/patient_*_counterfactual.png
=============================================================================
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from xgbse.metrics import concordance_index as xgbse_cindex
from xgbse.converters import convert_data_to_xgb_format

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")
plt.style.use("bmh")

# Project utility functions
from utils_calibration import (
    full_calibration_pipeline,
    plot_ibs_curve_DT,
    plot_patients_survival_DT_split_legend,
    estimate_LCT_ITE_for_test,
    plot_lct_counterfactual_for_patient,
)

# ---------------------------------------------------------------------------
# 2. Paths and settings
# ---------------------------------------------------------------------------
work_name = "model_output"
path_name = os.path.join("results", work_name)
model_path = os.path.join(path_name, "xgb_model.json")

# NOTE: x_train, y_train, x_test, y_test should be loaded/constructed
# as in demo2_model_construction.py.  Here we assume they are in scope
# (e.g., this script is run in the same session or the objects are loaded
# from disk).  Replace with your own loading logic if running standalone.

# ---------------------------------------------------------------------------
# 3. Load model and run inference
# ---------------------------------------------------------------------------
bst = xgb.Booster()
bst.load_model(model_path)
print("Model loaded from:", model_path)

dtrain = convert_data_to_xgb_format(x_train, y_train, "survival:aft")
dtest  = convert_data_to_xgb_format(x_test,  y_test,  "survival:aft")

preds_train = bst.predict(dtrain)
preds_test  = bst.predict(dtest)

# Normalize (fit on train, apply to test)
scaler = StandardScaler()
preds_train_norm = scaler.fit_transform(preds_train.reshape(-1, 1)).ravel()
preds_test_norm  = scaler.transform(preds_test.reshape(-1, 1)).ravel()

train_ci = xgbse_cindex(y_train, -preds_train, risk_strategy="precomputed")
test_ci  = xgbse_cindex(y_test,  -preds_test,  risk_strategy="precomputed")
print(f"C-index — Train: {train_ci:.4f} | Test: {test_ci:.4f}")


# ---------------------------------------------------------------------------
# 4. Calibration on Training cohort
# ---------------------------------------------------------------------------
# Prepare inputs
T_train    = y_train["PFS_time"]
E_train    = y_train["PFS_events"]
Drug_train = x_train["Drug"]          # drug arm label column (adjust name)

outdir_train = os.path.join(path_name, "Calibration_Train")
os.makedirs(outdir_train, exist_ok=True)

print("Fitting calibration pipeline on Training cohort...")
DT_R_train = full_calibration_pipeline(
    T_train, E_train, Drug_train,
    T_train, E_train, Drug_train,   # evaluate on itself (in-sample)
    preds_train, preds_train,
    EVAL_TIMES=[6, 12, 24],
    N_BINS=4,
    TAU_IBS=60,
    CALIBRATOR_CHOICE='weibull',         # set as 'None' will auto-select best family per drug
    outdir=outdir_train,
    save_plot=True,
    show_plot=True,
)


# ---------------------------------------------------------------------------
# 5. Calibration on External Test cohort (Digital Twin construction)
# ---------------------------------------------------------------------------
T_test    = y_test["PFS_time"]
E_test    = y_test["PFS_events"]
Drug_test = x_test["Drug"]

outdir_test = os.path.join(path_name, "Calibration_Test")
os.makedirs(outdir_test, exist_ok=True)

print("Fitting calibration pipeline on Test cohort (Digital Twins)...")
DT_R = full_calibration_pipeline(
    T_train, E_train, Drug_train,
    T_test,  E_test,  Drug_test,
    preds_train, preds_test,
    EVAL_TIMES=[6, 12, 24],
    N_BINS=4,
    TAU_IBS=60,
    CALIBRATOR_CHOICE=None,
    outdir=outdir_test,
    save_plot=True,
    show_plot=True,
)


# ---------------------------------------------------------------------------
# 6. Integrated Brier Score (IBS) evaluation
# ---------------------------------------------------------------------------
ibs_res = plot_ibs_curve_DT(
    DT_R=DT_R,
    T=T_test,
    E=E_test,
    groups=Drug_test,
    horizon=40,
    outpath=os.path.join(path_name, "ibs_curve_test_0-40m.png"),
    title="Brier Score & IBS — Test Cohort (0–40 months)",
    show=True,
)
print("IBS (overall)  :", ibs_res["ibs_all"])
print("IBS (by group) :", ibs_res["ibs_by_group"])


# ---------------------------------------------------------------------------
# 7. Individual Digital Twin survival curves
# ---------------------------------------------------------------------------
ids_to_plot = [0, 1, 2, 4, 5]   # replace with patient row indices of interest

res = plot_patients_survival_DT_split_legend(
    DT_R=DT_R,
    ids=ids_to_plot,
    indices=None, 
    index_like=Drug_test.index,
    r=preds_test,
    drug=Drug_test,
    T=T_test,
    E=E_test,
    annotate=("median", "mean"),
    outpath_curves_no_legend=os.path.join(
        outdir_test,
        f"patients_{'_'.join(map(str, ids_to_plot))}_curves_only.png"
    ),
    outpath_legend_only=os.path.join(
        outdir_test,
        f"patients_{'_'.join(map(str, ids_to_plot))}_legend_only.png"
    ),
    legend_ncol=1,
    title_curves="Selected Patients — Digital Twin Survival Curves",
    show=True,
)


# ---------------------------------------------------------------------------
# 8. Individual Treatment Effect (ITE) for LCT vs. no-LCT
# ---------------------------------------------------------------------------
print("Estimating LCT ITE for test patients...")
df_ite, curves, cals_lct, months_grid = estimate_LCT_ITE_for_test(
    T_train, E_train, Drug_train,
    T_test,  E_test,  Drug_test,
    preds_train, preds_test,
    family_noLCT="weibull",
    family_LCT="weibull",
    use_transport_weights=True,
    X_train_transport=None,
    X_test_transport=None,
    times=(6, 12, 24, 36, 48, 60),
    rmst_tau=36,
    months_max=60,
    landmark_month=1.5,
    n_boot=500,
    boot_seed=42,
)
print(df_ite.head())

# Save ITE table alongside patient metadata
cols_meta = ["PatientID", "Drug", "PFS", "PFS_events"]  # adjust to your column names
df_out = pd.concat([df_test[cols_meta].reset_index(drop=True), df_ite], axis=1)
ite_path = os.path.join(path_name, "ITE_test_cohort_60mo.csv")
df_out.to_csv(ite_path, index=False)
print("ITE table saved to:", ite_path)


# ---------------------------------------------------------------------------
# 9. Counterfactual curve for a specific patient
# ---------------------------------------------------------------------------
patient_idx = 2   # change to any test row index

plot_lct_counterfactual_for_patient(
    calibrators_B=cals_lct,
    r_value=float(preds_test[patient_idx]),
    months_grid=months_grid,
    observed_time=float(
        T_test.iloc[patient_idx] if hasattr(T_test, "iloc") else T_test[patient_idx]
    ),
    observed_event=int(
        E_test.iloc[patient_idx] if hasattr(E_test, "iloc") else E_test[patient_idx]
    ),
    title=None,
    outpath=os.path.join(
        path_name, f"patient_{patient_idx}_LCT_vs_noLCT_counterfactual.png"
    ),
    legend_outpath=os.path.join(
        path_name, f"patient_{patient_idx}_LCT_vs_noLCT_legend.png"
    ),
    show=True,
)


# ---------------------------------------------------------------------------
# 10. Aggregate Digital Twin PFS and save to CSV
# ---------------------------------------------------------------------------
PrePFS = pd.DataFrame({
    "PFS_events": DT_R["PFS_event_sim"],
    "PFS_time":   DT_R["PFS_time_sim"],
})

# Median Digital Twin PFS
kmf_dt = KaplanMeierFitter()
kmf_dt.fit(PrePFS["PFS_time"], event_observed=PrePFS["PFS_events"])
med_dt = kmf_dt.median_survival_time_
ci_dt  = median_survival_times(kmf_dt.confidence_interval_)
print(f"Digital Twin PFS: median = {med_dt:.2f} mo  "
      f"(95% CI [{ci_dt.iloc[0,0]:.2f}, {ci_dt.iloc[0,1]:.2f}])")

# Median Real PFS
kmf_real = KaplanMeierFitter()
kmf_real.fit(T_test, event_observed=E_test)
med_real = kmf_real.median_survival_time_
ci_real  = median_survival_times(kmf_real.confidence_interval_)
print(f"Observed PFS:     median = {med_real:.2f} mo  "
      f"(95% CI [{ci_real.iloc[0,0]:.2f}, {ci_real.iloc[0,1]:.2f}])")

# Log-rank test between Digital Twin and observed
lr = logrank_test(
    T_test, PrePFS["PFS_time"],
    event_observed_A=E_test,
    event_observed_B=PrePFS["PFS_events"],
)
print(f"Log-rank p (DT vs Observed): {lr.p_value:.4g}")

# Save combined output
preds_arr = np.array(preds_test, dtype=np.float32)
df_final = pd.concat([
    df_test.reset_index(drop=True),
    pd.DataFrame({
        "risk_score": preds_arr,
        "DT_PFS_events": PrePFS["PFS_events"].values,
        "DT_PFS_time":   PrePFS["PFS_time"].values,
    })
], axis=1)
out_csv = os.path.join(path_name, "digital_twin_predictions.csv")
df_final.to_csv(out_csv, index=False)
print("Digital Twin predictions saved to:", out_csv)

print("\nDemo 2 complete. Results saved to:", path_name)
