# OrganNet

A simplified implementation of **OrganNet** for organ-level response prediction.

This GitHub version intentionally contains only two Python files.

## Files

```text
OrganNet/
├── demo_OrganNet_train_test.py
├── utils_OrganNet.py
└── README.md
```

### `utils_OrganNet.py`

Contains:

- data loading and preprocessing
- label transformation
- `OrganDataset`
- `FocalLoss`
- `OrganNet`
- PyTorch Lightning training wrapper
- test/evaluation functions

### `demo_OrganNet_train_test.py`

A minimal example that:

1. reads the original OrganNet input files
2. prepares one predefined cross-validation fold
3. trains OrganNet
4. saves the best checkpoint
5. evaluates the held-out test set
6. saves predictions and attention weights

---

## Organ-level classes

The three output classes are defined as:

```text
Class 0: No metastasis
Class 1: Metastasis with non-PD response
Class 2: Metastasis with PD response
```

Original label mapping:

```text
Original label 0   -> Class 0
Original label >=2 -> Class 1
Original label 1   -> Class 2
```

---

## Input data

The current implementation preserves the original column structure.

### Organ features

```text
TC_Bone : ATS_Primary
```

There are:

```text
7 organs × 8 features = 56 organ features
```

### Organ labels

```text
RECISTLabel_Bone : RECISTLabel_Primary
```

### Clinical features

```text
Stage : Regimen
```

plus:

```text
Drug_v3
```

### Cross-validation split

The predefined split is read from:

```text
Fold1
Fold2
Fold3
...
```

In the demo file:

```python
CV = 0
```

means:

```text
Fold1
```

---

## Requirements

Recommended:

```text
Python >= 3.10
PyTorch
PyTorch Lightning
torchmetrics
numpy
pandas
scikit-learn
openpyxl
joblib
```

Install with:

```bash
pip install torch pytorch-lightning torchmetrics numpy pandas scikit-learn openpyxl joblib
```

---

## Run

Open:

```text
demo_OrganNet_train_test.py
```

and change:

```python
DATA_DIR = Path(
    "/Data/Projects/ALK/Features/Features_for_python"
)
```

to your local data directory.

Then run:

```bash
python demo_OrganNet_train_test.py
```

The script will create:

```text
outputs/fold_1/
├── checkpoints/
├── preprocessor.joblib
├── test_predictions.csv
├── test_attention_weights.npz
└── test_summary.json
```

The `preprocessor.joblib` file stores the scalers and feature definitions learned
from the training partition and should be kept together with the corresponding
model checkpoint.

---

## Model overview

OrganNet contains four major components:

1. organ feature encoder
2. patient-level baseline feature encoder
3. intra-patient self-attention across organs
4. organ-specific global memory

The fused representation is used for 3-class organ-level prediction.

Input:

```text
Organ features:   [batch, 7, 8]
Baseline features:[batch, baseline_dim]
```

Output:

```text
Logits:           [batch, 7, 3]
Attention weights:[batch, 7, 7]
```

---

## Data privacy

No patient-level data should be uploaded to a public GitHub repository.

Avoid committing:

```text
*.xlsx
*.csv
*.ckpt
*.joblib
*.npz
```

when these files contain patient-level information.
