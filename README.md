# OncoTwin: Insights from the Phase II BRIGHTSTAR Study in ALK-Rearranged NSCLC 
OncoTwin is a generalizable multimodal digital-twin framework to model individualized treatment trajectories by integrating imaging-derived tumor burden dynamics with routine blood test and demographic variables in ALK-rearranged NSCLC. Across systematic validation spanning real-world datasets and prospective clinical trials, OncoTwin accurately reproduced survival outcomes across TKI generations, simulated virtual control arm, and estimated patient-level treatment effects. Its successful deployment in the Phase II BRIGHTSTAR trial demonstrates the feasibility and translational potential of AI-enabled digital twins and establishes a blueprint for broader applications in oncology. 

![OncoTwin Pipeline](Figures/Fig.%201_pipeline.jpg)

## Key Features
**1.	Digital twin bridging real-world and trial data:** OncoTwin integrates real-world data from MD Anderson with two prospective trial cohorts (Phase III ALTA-1L and Phase II BrightStar) to enable robust risk stratification, calibrated individualized survival prediction, and virtual control-arm simulation in clinical trials.  
**2.	Multimodal, longitudinal modeling with clinically accessible inputs:** OncoTwin lev-erages whole-body 3D tumor burden, routine blood tests, and clinical demographics across baseline, early on-treatment, and delta timepoints, yielding interpretable and readi-ly translatable clinical insights.  
**3.	Trial-integrated clinical applications:** Within the Phase II BrightStar trial, OncoTwin simulated a brigatinib-only control arm to estimate the treatment effect of LCT, and used simulation-based resampling to determine stable sample size thresholds. Individualized treatment-effect estimates further highlight its potential for adaptive and evidence-guided trial design.  
**4.	Biological and therapeutic insight:** Beyond aggregate endpoints, OncoTwin revealed early-response heterogeneity between TKI generations, with second-generation TKIs achieving superior whole-body and organ-specific responses, and enabled TKI-specific risk stratification to support more granular treatment strategies.  
**5.	Spatially resolved disease modeling:**  By capturing inter-patient differences, OncoT-win quantified not only whole-body but also organ-specific tumor burden dynamics, providing a foundation for organ adaptive therapeutic strategies and a deeper under-standing of site-specific resistance.

## Repository Structure

```
OncoTwin/
├── Codes/                                       # Analysis pipeline (run in order)
│   ├── demo1_VolumetricFeatureExtraction.m      # Part 1 – Volumetric feature extraction
│   ├── demo2_model_construction.py              # Part 2 – Feature engineering & XGBoost-AFT training
│   ├── demo3_calibration_digital_twin.py        # Part 3 – Post-hoc calibration & DT survival curves
│   └── demo4_risk_stratification_visualization.py  # Part 4 – Risk stratification & figures
├── Utils/                                        # Shared functions
│   ├── utils_survival.py                         # KM / SHAP / statistical utilities
│   ├── utils_calibration.py                      # Calibration pipeline (full_calibration_pipeline, etc.)
│   └── README_VolumetricFeatureExtraction.md        # KM / SHAP / statistical utilities
├── Figures/                                      # Generated figures & plots
│   └── Fig. 1_pipeline.jpg                       # Fig. 1
└── README.md
```

## Installation
```bash
pip install numpy pandas scikit-learn xgboost xgbse lifelines shap \
            matplotlib seaborn scipy dill openpyxl
```
GPU training requires an NVIDIA GPU with CUDA drivers installed.  
Set `"device": "cpu"` in `PARAMS_XGB_AFT` inside `demo2_model_construction.py` to run on CPU.

### 1. **Tumor Burden Measurement (`demo1_VolumetricFeatureExtraction.mat`)**
- **Description**: Extract volumetric Features
- End-to-end MATLAB pipeline for extracting Base and Advance volumetric / radiomics features from CT volumes and segmentation masks stored in NIfTI format.
- Please see the details in 'README_Demo_VolumetricFeatureExtraction.md'

### 2. **Prepare the data**
Each cohort requires **Excel files contains the input features** (including clinical, routine blood test, CT-based tumor volumetrics measuremnet including pre-treatment, post-treatment and delta), such as:

| File | Description |
|------|-------------|
| `VolOriginal_Base_overall_MDACC.xlsx`  | overall features including clinical, blood test, CT-based tumor volumetrics measuremnets |
| `VolOriginal_Base_overall_BrightStar.xlsx`  | overall features including clinical, blood test, CT-based tumor volumetrics measuremnets |

Each file must contain:
- A unique patient identifier column (e.g. `PatientID`)  
- Tumor volumetrics columns, such as spanning from `TV_Lung` to `ATS_AllTumor`  
- Clinical columns: `Sex`, `Age`, `Smoker`, `Pathology`, `Drug`, `RECIST`, and lab test values  
- Outcome columns: `PFS` (time in months), `PFS_events` (0 = censored, 1 = event)

Update the `train_data_dir` and `test_data_dir` paths at the top of **Demo 1**.  [Check this!!!]

### 3. **Model Construction (`demo2_model_construction.py`)**

| Step | Description |
|------|-------------|
| Data loading | Read feature tables for train and test cohort |
| Categorical encoding | Binary-encode sex, race, smoking, pathology, RECIST |
| Feature filtering | Remove highly correlated radiomic features (Pearson \|r\| > 0.98) |
| XGBoost-AFT training | DART booster, GPU accelerated, 1000 boosting rounds |
| Evaluation | Concordance index (C-index) on train and external test cohort |
| Feature importance | Gain-based bar chart + SHAP beeswarm summary | [REMOVE from the Code!]

**Key outputs:**  
`results/model_output/xgb_model.json`  
`results/model_output/feature_importance_gain.*`  
`results/model_output/SHAP/SHAP_summary_*.png`
`results/<work_name>/SHAP_importance_*.csv`

---
### 4. **Calibration & Digital Twin (`demo3_calibration_digital_twin.py`)**

| Step | Description |
|------|-------------|
| Drug-stratified calibration | Fit Weibull calibrators per drug arm using `full_calibration_pipeline` |
| Brier score | Compute and plot time-varying Brier score + IBS |
| Individual OncoTwin-predicted Survival curves | Plot individual predicted survival curves with median / mean annotations |
| ITE estimation | Estimate invidual ΔRMST (LCT vs. no-LCT) via `estimate_LCT_ITE_for_test` |
| Counterfactual plot | Plot patient's LCT vs. no-LCT counterfactual survival curves |
| Export | Save Digital Twin PFS predictions and ITE table to CSV |

Individual OncoTwin-predicted Survival curves

**Key outputs:**  
`results/model_output/Calibration_*/`  
`results/model_output/ITE_test_cohort_60mo.csv`  
`results/model_output/digital_twin_predictions.csv`

---

### 5. **Risk Stratification & Visualization (`demo4_risk_stratification_visualization.py`)**

| Step | Description |
|------|-------------|
| Optimal cutoff | Log-rank search over percentile grid on training predictions |
| Risk groups | Assign High / Low risk labels to train and test cohorts |
| KM plots | Plot KM curves for risk groups; Digital Twin vs. Observed | 
| HR computation | Cox proportional hazards HR (High vs. Low; DT vs. Observed) |
| SHAP aggregation | Overall beeswarm + correlation-filtered top-15 plot |
| Top-6 bar chart | Publication-style horizontal bar chart of top SHAP features | [REMOVE from the Code!]
| ITE correlation | Spearman ρ between features and ΔRMST, annotated with significance stars | [REMOVE from the Code!]

**Key outputs:**  
`results/model_output/KM_*.png`  
`results/model_output/SHAP_summary_overall.png`  
`results/model_output/cox_*.csv`

---

## Utility Modules

### `utils_survival.py`

Contains all KM plotting, statistical comparison, and SHAP utility functions
used across the three demos.  No patient-specific data or file paths are
hard-coded in this module.

Key functions:

| Function | Description |
|----------|-------------|
| `km_analysis` | Compute median PFS and N-month survival with 95% CI for up to four groups |
| `remove_highly_correlated_features` | Pearson correlation pruning |
| `plot_km_curve` | Two-arm KM plot with at-risk table |
| `plot_km_curve_dashed` | Two-arm KM plot with configurable line styles |
| `compare_low_high` | Cox HR + log-rank p for two survival groups |
| `safe_shap_summary_plot` | Memory-safe SHAP beeswarm + importance CSV |

### `utils_calibration.py`

Calibration-specific utilities.  
The following functions are imported in Demo 2:

- `full_calibration_pipeline` — fits drug-stratified survival calibrators  
- `plot_ibs_curve_DT` — plots time-varying Brier score  
- `plot_patients_survival_DT_split_legend` — plots individual DT survival curves  
- `estimate_LCT_ITE_for_test` — estimates per-patient ITE (ΔRMST) for LCT  
- `plot_lct_counterfactual_for_patient` — counterfactual plot for one patient  

---

## Citation
If you use this framework, please cite our work:
```bash
@article{OncoTwin,
  title={Digital Twin Enabled Translation of Real-World Evidence into Prospective Clinical Trial Design: Insights from the Phase II BRIGHTSTAR Study in ALK-Rearranged NSCLC},
  author={Hui Xu, Yasir Y Elamin, Lingzhi Hong, Saumil N Gandhi, Kyle Concannon, Maliazurina B Saad, Xinyan Xu, Amgad Muneer, Waqas Muhammad, Hui Li, Kang Qin, Xiaoyu Han,Sherif M Ismail, Yuliya Kitsel, Mara B Antonoff, Carol C Wu, Brett W Carter, Girish S Shroff, Simon Heeke, Xiuning Le, Tina Cascone, Natalie I Vokes, Mehmet Altan, Don L Gibbons, David Jaffray, Joe Y Chang, Zhongxing Liao, David Rice, Ara A Vaporiciyan, Stephen G Swisher, J. Jack Lee, Jianjun Zhang, John V Heymach, Jia Wu},
  year={2026}
}
```
For questions, contributions, or issues, please contact us (hxu12@mdanderson.org) or create a new issue in this repository.

## License

This code is released for research and educational use.  
See `LICENSE` for details.














