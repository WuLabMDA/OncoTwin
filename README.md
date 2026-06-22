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
digital_twins_survival/
├── demo1_model_construction.py          # Part 1 – Feature engineering & XGBoost-AFT training
├── demo2_calibration_digital_twin.py    # Part 2 – Post-hoc calibration & DT curve generation
├── demo3_risk_stratification_visualization.py  # Part 3 – Risk stratification & figures
├── utils_survival.py                    # Shared KM / SHAP / stat utility functions
├── utils_calibration.py                 # Calibration pipeline (full_calibration_pipeline, etc.)
└── README.md
```

---
## Quick Start

## Installation
To install the development version of OncoTwin using pip, run the following command:
```bash
pip install git+https://github.com/WuLabMDA/OncoTwin.git
```
```bash
pip install numpy pandas scikit-learn xgboost xgbse lifelines shap \
            matplotlib seaborn scipy dill openpyxl
```
GPU training requires an NVIDIA GPU with CUDA drivers installed.  
Set `"device": "cpu"` in `PARAMS_XGB_AFT` inside `demo1_model_construction.py` to run on CPU.

## The repository contains the following files:
### 1. **Tumor Burden Measurement**
- **File**: `Demo_VolumetricFeatureExtraction.mat`  
- **Description**: Extract volumetric Features
- End-to-end MATLAB pipeline for extracting Base and Advance volumetric / radiomics features from CT volumes and segmentation masks stored in NIfTI format.
- Please see the details in 'README_Demo_VolumetricFeatureExtraction.md'

### 2. **Model Training**
- **File**: `S2_BarPlot.py`  
- **Description**: Visualizes feature selection frequency across models. Generates bar plots and histograms to analyze selected features.  
- **Inputs**: Model outputs and feature metadata (`Type.xlsx`).  
- **Outputs**: Feature selection bar plots.

### 3. **Model Calibration and Prediction**
- **File**: `S2_BarPlot.py`  
- **Description**: Visualizes feature selection frequency across models. Generates bar plots and histograms to analyze selected features.  
- **Inputs**: Model outputs and feature metadata (`Type.xlsx`).  
- **Outputs**: Feature selection bar plots.
- 
### 4. **Risk Stratification and Individual Survival Curves Prediction**
- **File**: `S2_BarPlot.py`  
- **Description**: Visualizes feature selection frequency across models. Generates bar plots and histograms to analyze selected features.  
- **Inputs**: Model outputs and feature metadata (`Type.xlsx`).  
- **Outputs**: Feature selection bar plots.

### 5. **Individual Treatment Effect Estimation**
- **File**: `S2_BarPlot.py`  
- **Description**: Visualizes feature selection frequency across models. Generates bar plots and histograms to analyze selected features.  
- **Inputs**: Model outputs and feature metadata (`Type.xlsx`).  
- **Outputs**: Feature selection bar plots.



## Results
- **Treatment Recommendations**: I-SABR-SELECT identified a significant subgroup of patients benefiting from adding immunotherapy.
- **Improved Outcomes**: Patients treated following model recommendations demonstrated superior event-free survival (EFS) compared to random treatment assignment.


![OncoTwin Pipeline](Figures/Results.png)

## Reference: https://github.com/WuLabMDA/ISABR-SELECT [should delete at the end]

## Citation
If you use this framework, please cite our work:
```bash
@article{ISABRSelect,
  title={Digital Twin Enabled Translation of Real-World Evidence into Prospective Clinical Trial Design: Insights from the Phase II BRIGHTSTAR Study in ALK-Rearranged NSCLC},
  author={},
  journal={},
  year={Year},
  volume={Volume},
  pages={Pages},
  doi={DOI}
}
```
For questions, contributions, or issues, please contact us or create a new issue in this repository.














