# Volumetric Feature Extraction from NIfTI Images

End-to-end MATLAB pipeline for extracting **Base** and **Advance** volumetric / radiomics features from paired CT volumes and segmentation masks stored in NIfTI format. Originally developed and validated in Wu Lab.

---

## Repository Structure

```
.
├── Demo_VolumetricFeatureExtraction.m   ← Main demo + all self-contained functions
└── README.md
```

> **Note:** `Demo_VolumetricFeatureExtraction.m` is a single self-contained script. It embeds `Standard_VolFea_Extract` and `calculateMaxDiameter` as local functions, so no additional `.m` files are required. External helper functions (`nii_tool`, `getSUVbox`, `cc2bw`, `merge2Structs`) must be on the MATLAB path (see *Dependencies* below).

---

## File Descriptions

### File: `Demo_VolumetricFeatureExtraction.m`
* **Description:** Main entry-point demo script. Iterates over a folder tree of patient NIfTI data, loads each CT volume and its associated segmentation mask(s), calls `Standard_VolFea_Extract` to compute volumetric features, validates output integrity, and saves all results to a `.mat` file. Also contains `Standard_VolFea_Extract` and `calculateMaxDiameter` as embedded local functions.
* **Inputs:**
  * `mainpath` *(string)* — Root directory containing patient sub-folders. Each sub-folder must include:
    * One CT file with `"CT.nii"` in the filename
    * One or more segmentation masks with `"RTS"` in the filename (e.g. `RTS_Tumor.nii`)
  * `savepath` *(string)* — Output directory for the saved `.mat` feature file
  * `featureCategory` *(string)* — Feature set to extract: `'base'`, `'advance'`, or `'all'`
* **Outputs:** `Case_v1_all.mat` — a cell array (`Pre_Fea_patient_all`) where each cell holds a struct array of features for one patient, with fields detailed in the table below

---

### File: `Standard_VolFea_Extract` *(embedded local function)*
* **Description:** Core feature extraction function. Accepts a CT / mask NIfTI pair, validates spatial consistency, computes connected-component statistics, and returns a struct of Base and/or Advance volumetric features. Advance features are computed on an isotropic 1×1×1 mm resampling of the mask.
* **Inputs:**
  * `CT` *(NIfTI struct)* — CT volume loaded via `nii_tool`
  * `mask` *(NIfTI struct)* — Binary segmentation mask loaded via `nii_tool`
  * `category` *(string)* — `'base'` | `'advance'` | `'all'`
* **Outputs:** `Feature_all` *(struct)* with the following fields:

| Field | Category | Unit | Description |
|---|---|---|---|
| `Met` | Base | — | Metastasis flag (1 = lesion present, 0 = none) |
| `TC` | Base | count | Total number of connected lesion components |
| `TV` | Base | cm³ | Total tumor volume (sum of all components) |
| `LA` | Base | cm² | Sum of each lesion's largest 2-D cross-sectional area |
| `LD` | Base | cm | Sum of each lesion's largest in-plane diameter |
| `LOV` | Base | cm³ | Volume of the single largest lesion |
| `LOA` | Base | cm² | Largest 2-D area of the single largest lesion |
| `LOD` | Base | cm | Largest in-plane diameter of the single largest lesion |
| `LZR` | Base | cm | Longest extent of the whole mask along the Z (axial) axis |
| `L2A` | Base | cm² | Sum of the two largest lesion areas (simulated RECIST) |
| `L2D` | Base | cm | Sum of the two largest lesion diameters (simulated RECIST) |
| `MaxEquivD` | Advance | mm | Equivalent sphere diameter of the largest lesion |
| `SumEquivD` | Advance | mm | Sum of equivalent diameters across all lesions |
| `StdEquivD` | Advance | mm | Standard deviation of equivalent diameters |
| `MaxMaxAxis` | Advance | mm | Longest principal axis across all lesions |
| `MinMaxAxis` | Advance | mm | Shortest principal axis across all lesions |
| `SumMaxAxis` | Advance | mm | Sum of longest principal axes |
| `StdMaxAxis` | Advance | mm | Std of longest principal axes |
| `MaxSolidity` | Advance | — | Maximum solidity across lesions |
| `MeanSolidity` | Advance | — | Mean solidity across lesions |
| `StdSolidity` | Advance | — | Std of solidity across lesions |
| `Skew` | Advance | — | Skewness of CT Hounsfield values inside the mask |
| `Kurt` | Advance | — | Kurtosis of CT Hounsfield values inside the mask |
| `Entropy` | Advance | — | Entropy of CT Hounsfield values inside the mask |

---

### File: `calculateMaxDiameter` *(embedded local function)*
* **Description:** Computes the maximum caliper diameter (Feret diameter) of a single 2-D binary mask slice by finding the greatest Euclidean distance between any two boundary points. Called internally by `Standard_VolFea_Extract` for every axial slice of each lesion.
* **Inputs:** `mask` *(logical 2-D array)* — binary slice where 1 = foreground, 0 = background
* **Outputs:** `maxDiameter` *(scalar, pixels)* — maximum boundary-to-boundary distance; multiply by pixel spacing (mm) and scale factor to convert to physical units

---

## Dependencies

The following functions must be available on the MATLAB path (Wu Lab toolbox):

| Function | Purpose |
|---|---|
| `nii_tool` | Read/write NIfTI files |
| `getSUVbox` | Crop a bounding box around the mask region |
| `cc2bw` | Convert a connected-component struct to a binary volume |
| `merge2Structs` | Merge two structs with non-overlapping field names |

---

## Quick Start

```matlab
% 1. Add dependencies to path
addpath(genpath('/path/to/WuLab/toolbox'));

% 2. Edit the three config variables at the top of the script
%    mainpath        = '/your/NIFTI/data/root';
%    savepath        = '/your/output/folder';
%    featureCategory = 'all';   % 'base' | 'advance' | 'all'

% 3. Run
Demo_VolumetricFeatureExtraction
```

Results are saved as `Case_v1_all.mat` in `savepath`.

---

## Citation / Acknowledgements

Feature definitions validated by Wu Lab. Core implementation by Hui Xu (2024).
