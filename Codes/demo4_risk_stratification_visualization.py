"""
=============================================================================
Demo: Risk Stratification & Survival Visualization
=============================================================================
Description:
    This script demonstrates patient risk stratification using an optimal
    log-rank cutoff on model risk scores, and produces Kaplan-Meier survival
    plots along with SHAP feature importance figures.

    Steps:
      1. Determine optimal risk score cutoff via log-rank search (on training)
      2. Stratify training and test cohorts into High/Low risk groups
      3. Plot KM curves for risk groups (train & test)
      4. Plot Digital Twin vs. Observed KM curves 
      5. SHAP value aggregation and beeswarm summary plot

Dependencies:
    numpy, pandas, matplotlib, lifelines, shap, sklearn, scipy
    utils_survival.py    (project utility functions)

Usage:
    python demo4_risk_stratification_visualization.py
    # Requires demo1 and demo2 to have been run first (or loaded from disk).
    # Key variables needed: preds_train, preds_test, y_train, y_test,
    #   x_train, x_test, PrePFS, shap_values, feat_names, df_ite

Outputs:
    - results/<work_name>/KM_train.png
    - results/<work_name>/KM_test.png
    - results/<work_name>/KM_DT_vs_observed.png
    - results/<work_name>/KM_low_risk_subgroup.png
    - results/<work_name>/KM_high_risk_subgroup.png
    - results/<work_name>/SHAP_summary_overall.png
    - results/<work_name>/cox_*.csv  (hazard ratio tables)
=============================================================================
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
from xgbse.metrics import concordance_index as xgbse_cindex

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")
plt.style.use("bmh")

# Project utility functions
from utils_survival import (
    km_analysis,
    plot_km_curve,
    plot_km_curve_dashed,
)


# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
work_name = "model_output"
path_name = os.path.join("results", work_name)
os.makedirs(path_name, exist_ok=True)

# NOTE: The following variables must be in scope (from demo2 + demo3):
#   preds_train, preds_test, y_train, y_test
#   x_train, x_test
#   PrePFS        — Digital Twin PFS DataFrame (PFS_time, PFS_events)
#   shap_values   — SHAP values array  [n_samples × n_features]
#   feat_names    — list of feature names (length = n_features)


# ===========================================================================
# PART A: Risk Stratification
# ===========================================================================

# ---------------------------------------------------------------------------
# 3. Find optimal cutoff on training predictions (log-rank search)
# ---------------------------------------------------------------------------
candidate_percentiles = np.arange(30, 71, 10)
candidate_cutoffs     = np.percentile(preds_train, candidate_percentiles)

best_p, best_cutoff = float("inf"), None
for cutoff in candidate_cutoffs:
    grp1 = y_train[preds_train <= cutoff]
    grp2 = y_train[preds_train >  cutoff]
    res  = logrank_test(
        grp1["PFS_time"], grp2["PFS_time"],
        event_observed_A=grp1["PFS_events"],
        event_observed_B=grp2["PFS_events"],
    )
    if res.p_value < best_p:
        best_p, best_cutoff = res.p_value, cutoff

print(f"Best cutoff : {best_cutoff:.4f}")
print(f"Log-rank p  : {best_p:.3e}")


# ---------------------------------------------------------------------------
# 4. Assign risk groups — Training cohort
# ---------------------------------------------------------------------------
x_train = x_train.copy()
x_train["group"] = np.where(preds_train > best_cutoff, "Low Risk", "High Risk")

low_train  = y_train[x_train["group"] == "Low Risk"]
high_train = y_train[x_train["group"] == "High Risk"]

lr_train = logrank_test(
    low_train["PFS_time"], high_train["PFS_time"],
    event_observed_A=low_train["PFS_events"],
    event_observed_B=high_train["PFS_events"],
)
print(f"[Train] Log-rank p = {lr_train.p_value:.3e}")

# KM plot — Training
plot_km_curve(
    low_risk=low_train, high_risk=high_train,
    time_col="PFS_time", event_col="PFS_events",
    title="Kaplan-Meier — Training Cohort",
    save_path=os.path.join(path_name, "KM_train.png"),
    low_risk_label="Low Risk", high_risk_label="High Risk",
)

# C-index and HR — Training
grp_enc_train = x_train["group"].map({"High Risk": 1, "Low Risk": 0})
ci_train = xgbse_cindex(y_train, grp_enc_train, risk_strategy="precomputed")
print(f"[Train] Group C-index = {ci_train:.4f}")

df_cox_train = pd.concat(
    [grp_enc_train, pd.DataFrame(y_train)], axis=1
).rename(columns={0: "group"})
cph = CoxPHFitter(penalizer=0.05)
cph.fit(df_cox_train, duration_col="PFS_time", event_col="PFS_events")
cph.print_summary()
cph.summary.to_csv(os.path.join(path_name, "cox_train.csv"))


# ---------------------------------------------------------------------------
# 5. Assign risk groups — Test cohort
# ---------------------------------------------------------------------------
x_test = x_test.copy()
x_test["group"]      = np.where(preds_test > best_cutoff, "Low Risk", "High Risk")
x_test["group_pred"] = x_test["group"].copy()

low_test  = y_test[x_test["group"] == "Low Risk"]
high_test = y_test[x_test["group"] == "High Risk"]

lr_test = logrank_test(
    low_test["PFS_time"], high_test["PFS_time"],
    event_observed_A=low_test["PFS_events"],
    event_observed_B=high_test["PFS_events"],
)
print(f"[Test]  Log-rank p = {lr_test.p_value:.3e}")

# KM plot — Test
plot_km_curve(
    low_risk=low_test, high_risk=high_test,
    time_col="PFS_time", event_col="PFS_events",
    title="Kaplan-Meier — Test Cohort",
    save_path=os.path.join(path_name, "KM_test.png"),
    low_risk_label="Low Risk", high_risk_label="High Risk",
)

# C-index and HR — Test
grp_enc_test = x_test["group"].map({"High Risk": 1, "Low Risk": 0})
ci_test = xgbse_cindex(y_test, grp_enc_test, risk_strategy="precomputed")
print(f"[Test]  Group C-index = {ci_test:.4f}")

df_cox_test = pd.concat(
    [grp_enc_test, pd.DataFrame(y_test)], axis=1
).rename(columns={0: "group"})
cph2 = CoxPHFitter(penalizer=0.05)
cph2.fit(df_cox_test, duration_col="PFS_time", event_col="PFS_events")
cph2.print_summary()
cph2.summary.to_csv(os.path.join(path_name, "cox_test.csv"))

# KM summary statistics
print("\n--- KM Summary (Test cohort, risk groups) ---")
km_analysis(low_test, high_test, high_test, low_test, time_point=24)


# ===========================================================================
# PART B: Digital Twin vs. Observed Comparison
# ===========================================================================

# ---------------------------------------------------------------------------
# 6. Overall Digital Twin KM vs. Observed
# ---------------------------------------------------------------------------
GTPFS = y_test   # ground-truth PFS structured array

lr_compare = logrank_test(
    GTPFS["PFS_time"], PrePFS["PFS_time"],
    event_observed_A=GTPFS["PFS_events"],
    event_observed_B=PrePFS["PFS_events"],
)
print(f"\n[DT vs Observed] Log-rank p = {lr_compare.p_value:.3e}")

plot_km_curve_dashed(
    low_risk=GTPFS,
    high_risk=PrePFS,
    time_col="PFS_time", event_col="PFS_events",
    title="Digital Twin vs. Observed PFS",
    save_path=os.path.join(path_name, "KM_DT_vs_observed.png"),
    low_risk_color="black",  high_risk_color="black",
    low_risk_label="Observed (Real)",
    high_risk_label="Digital Twin",
    low_risk_linestyle="-",
    high_risk_linestyle="--",
)

# HR — DT vs Observed
df_hr_all = pd.concat([
    pd.DataFrame({"PFS_time": GTPFS["PFS_time"],   "PFS_events": GTPFS["PFS_events"],   "Group": 0}),
    pd.DataFrame({"PFS_time": PrePFS["PFS_time"],  "PFS_events": PrePFS["PFS_events"],  "Group": 1}),
], ignore_index=True)
cph_all = CoxPHFitter(penalizer=0.0)
cph_all.fit(df_hr_all, duration_col="PFS_time", event_col="PFS_events")
cph_all.print_summary()
HR_all = cph_all.hazard_ratios_["Group"]
print(f"HR (DT vs Observed): {HR_all:.3f}")
cph_all.summary.to_csv(os.path.join(path_name, "cox_DT_vs_observed.csv"), index=False)

km_analysis(PrePFS, GTPFS, GTPFS, PrePFS, time_point=24)


# ---------------------------------------------------------------------------
# 7. Subgroup comparison — Low Risk
# ---------------------------------------------------------------------------
low_obs  = y_test[x_test["group"] == "Low Risk"]
low_dt   = PrePFS[x_test["group_pred"] == "Low Risk"]

lr_low = logrank_test(
    low_obs["PFS_time"], low_dt["PFS_time"],
    event_observed_A=low_obs["PFS_events"],
    event_observed_B=low_dt["PFS_events"],
)
print(f"\n[Low Risk] Log-rank p (Obs vs DT) = {lr_low.p_value:.3e}")

plot_km_curve_dashed(
    low_risk=low_obs, high_risk=low_dt,
    time_col="PFS_time", event_col="PFS_events",
    title="Low-Risk Subgroup",
    save_path=os.path.join(path_name, "KM_low_risk_subgroup.png"),
    low_risk_color="#348ABD", high_risk_color="#348ABD",
    low_risk_label="Observed (ALK+LCT)",
    high_risk_label="Digital Twin (ALK)",
    low_risk_linestyle="-", high_risk_linestyle="--",
)

# HR — Low Risk
df_hr_low = pd.concat([
    pd.DataFrame({"PFS_time": low_obs["PFS_time"], "PFS_events": low_obs["PFS_events"], "Group": 0}),
    pd.DataFrame({"PFS_time": low_dt["PFS_time"],  "PFS_events": low_dt["PFS_events"],  "Group": 1}),
], ignore_index=True)
cph_low = CoxPHFitter(penalizer=0.0)
cph_low.fit(df_hr_low, duration_col="PFS_time", event_col="PFS_events")
HR_low = cph_low.hazard_ratios_["Group"]
print(f"HR (DT vs Obs) in Low Risk: {HR_low:.3f}")
cph_low.summary.to_csv(os.path.join(path_name, "cox_low_risk_subgroup.csv"), index=False)
km_analysis(low_dt, low_obs, low_obs, low_dt, time_point=24)


# ---------------------------------------------------------------------------
# 8. Subgroup comparison — High Risk
# ---------------------------------------------------------------------------
high_obs = y_test[x_test["group"] == "High Risk"]
high_dt  = PrePFS[x_test["group_pred"] == "High Risk"]

lr_high = logrank_test(
    high_obs["PFS_time"], high_dt["PFS_time"],
    event_observed_A=high_obs["PFS_events"],
    event_observed_B=high_dt["PFS_events"],
)
print(f"\n[High Risk] Log-rank p (Obs vs DT) = {lr_high.p_value:.3e}")

plot_km_curve_dashed(
    low_risk=high_obs, high_risk=high_dt,
    time_col="PFS_time", event_col="PFS_events",
    title="High-Risk Subgroup",
    save_path=os.path.join(path_name, "KM_high_risk_subgroup.png"),
    low_risk_color="#A60628", high_risk_color="#A60628",
    low_risk_label="Observed (ALK+LCT)",
    high_risk_label="Digital Twin (ALK)",
    low_risk_linestyle="-", high_risk_linestyle="--",
)

df_hr_high = pd.concat([
    pd.DataFrame({"PFS_time": high_obs["PFS_time"], "PFS_events": high_obs["PFS_events"], "Group": 0}),
    pd.DataFrame({"PFS_time": high_dt["PFS_time"],  "PFS_events": high_dt["PFS_events"],  "Group": 1}),
], ignore_index=True)
cph_high = CoxPHFitter(penalizer=0.0)
cph_high.fit(df_hr_high, duration_col="PFS_time", event_col="PFS_events")
HR_high = cph_high.hazard_ratios_["Group"]
print(f"HR (DT vs Obs) in High Risk: {HR_high:.3f}")
cph_high.summary.to_csv(os.path.join(path_name, "cox_high_risk_subgroup.csv"), index=False)
km_analysis(high_dt, high_obs, high_obs, high_dt, time_point=24)

# Save final combined prediction table
preds_arr = np.array(preds_test, dtype=np.float32)
df_final = pd.concat([
    pd.DataFrame({"risk_score": preds_arr,
                  "DT_PFS_events": PrePFS["PFS_events"].values,
                  "DT_PFS_time":   PrePFS["PFS_time"].values}),
    x_test[["group", "group_pred"]].reset_index(drop=True),
], axis=1)
df_final.to_csv(os.path.join(path_name, "predictions_with_risk_groups.csv"), index=False)


# ===========================================================================
# PART C: SHAP Feature Importance
# ===========================================================================

# ---------------------------------------------------------------------------
# 10. Aggregate and plot overall SHAP summary
# ---------------------------------------------------------------------------
# shap_values : numpy array  [n_samples × n_features]
# feat_names  : list of str  (length = n_features)

max_samples = min(1000, shap_values.shape[0])
sv  = shap_values[:max_samples]
top_k = 15
mean_abs = np.abs(sv).mean(axis=0)
topk_idx = np.argsort(mean_abs)[-top_k:][::-1]

sv_top   = sv[:, topk_idx]
feat_top = [feat_names[i] for i in topk_idx]
data_top = shap_values_data[:max_samples, topk_idx]  # raw input values aligned to sv

shap.summary_plot(sv_top, data_top, feature_names=feat_top, plot_type="dot", show=False)
plt.title("SHAP Feature Importance — Top 15 (Overall)", fontsize=14)
ax = plt.gca()
ax.tick_params(axis="both", labelsize=13, width=1.2)
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("bold")
plt.tight_layout()
plt.savefig(os.path.join(path_name, "SHAP_summary_overall.png"), dpi=300)
plt.show()
plt.close()

# Save SHAP importance CSV
shap_imp_df = pd.DataFrame({
    "Feature": feat_names,
    "MeanAbsSHAP": mean_abs,
}).sort_values(by="MeanAbsSHAP", ascending=False)
shap_imp_df.to_csv(os.path.join(path_name, "SHAP_importance_overall.csv"), index=False)

