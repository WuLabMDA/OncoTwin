"""
demo_OrganNet_train_test.py

Minimal training + testing example for OrganNet.

Edit the configuration section below, then run:

    python demo_OrganNet_train_test.py
"""

from pathlib import Path
import json

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils_OrganNet import (
    OrganPrognosisSystem,
    evaluate_model,
    load_and_prepare_data,
    predictions_to_dataframe,
    set_seed,
)


# =============================================================================
# 1. User configuration
# =============================================================================

# Directory containing the two Excel files.
DATA_DIR = Path(
    "/Data/Projects/ALK/Features/Features_for_python"
)

# Original OrganNet input files.
ORGAN_FILE = (
    "ALK_allSEG_VolFeaExtent_before_103pts_250222_8F_v3.xlsx"
)
CLINICAL_FILE = (
    "VolOriginal_Base_PFS_before_103pts.xlsx"
)

# Zero-based CV index:
# CV = 0 -> Fold1
# CV = 1 -> Fold2
# CV = 2 -> Fold3
CV = 0

OUTPUT_DIR = Path("outputs") / f"fold_{CV + 1}"

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_RATIO = 0.20

MAX_EPOCHS = 100

LEARNING_RATE = 2e-4
HIDDEN_DIM = 8
NUM_HEADS = 4
MEMORY_SLOTS = 5

# Focal Loss parameters from the original code.
FOCAL_ALPHA = (0.1, 0.2, 0.7)
FOCAL_GAMMA = 2.0

# Original test code used temperature = 2.0.
TEMPERATURE = 2.0


# =============================================================================
# 2. Prepare data
# =============================================================================

set_seed(SEED)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

preprocessor_path = (
    OUTPUT_DIR / "preprocessor.joblib"
)

(
    train_loader,
    val_loader,
    test_loader,
    metadata,
) = load_and_prepare_data(
    data_dir=DATA_DIR,
    organ_file=ORGAN_FILE,
    clinical_file=CLINICAL_FILE,
    cv=CV,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    val_ratio=VAL_RATIO,
    seed=SEED,
    preprocessor_save_path=preprocessor_path,
)

print(
    f"Fold: {metadata['fold']}"
)
print(
    f"Organ input dimension: "
    f"{metadata['num_organs']} organs x "
    f"{metadata['organ_dim']} features"
)
print(
    f"Baseline dimension: "
    f"{metadata['baseline_dim']}"
)


# =============================================================================
# 3. Initialize OrganNet
# =============================================================================

model = OrganPrognosisSystem(
    organ_dim=metadata["organ_dim"],
    num_organs=metadata["num_organs"],
    baseline_dim=metadata["baseline_dim"],
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_classes=metadata["num_classes"],
    memory_slots=MEMORY_SLOTS,
    alpha=FOCAL_ALPHA,
    gamma=FOCAL_GAMMA,
    learning_rate=LEARNING_RATE,
)


# =============================================================================
# 4. Checkpoint and early stopping
# =============================================================================

checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_DIR / "checkpoints",
    filename=(
        "OrganNet-"
        "{epoch:03d}-"
        "{val_auc:.4f}"
    ),
    monitor="val_auc",
    mode="max",
    save_top_k=1,
    save_last=True,
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=15,
)


# =============================================================================
# 5. Train
# =============================================================================

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    callbacks=[
        checkpoint_callback,
        early_stopping,
    ],
    deterministic=True,
    log_every_n_steps=1,
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)


# =============================================================================
# 6. Load best checkpoint
# =============================================================================

best_checkpoint = (
    checkpoint_callback.best_model_path
)

if best_checkpoint == "":
    best_checkpoint = (
        checkpoint_callback.last_model_path
    )

print(
    f"Best checkpoint: {best_checkpoint}"
)

best_model = (
    OrganPrognosisSystem
    .load_from_checkpoint(
        best_checkpoint
    )
)


# =============================================================================
# 7. Test
# =============================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

(
    metrics,
    all_probs,
    all_preds,
    all_labels,
    all_attention,
) = evaluate_model(
    model=best_model,
    test_loader=test_loader,
    device=device,
    temperature=TEMPERATURE,
)

print("\nTest results")
print(
    f"Overall accuracy: "
    f"{metrics['total_accuracy']:.4f}"
)
print(
    f"Global macro AUC: "
    f"{metrics['global_macro_auc']:.4f}"
)

for organ_name, acc, auc_value in zip(
    metadata["organ_names"],
    metrics["organ_accuracy"],
    metrics["organ_auc"],
):
    print(
        f"{organ_name:>10s} | "
        f"ACC = {acc:.4f} | "
        f"AUC = {auc_value:.4f}"
    )


# =============================================================================
# 8. Save predictions and attention weights
# =============================================================================

prediction_df = predictions_to_dataframe(
    mrns=metadata["mrn_test"],
    organ_names=metadata["organ_names"],
    all_probs=all_probs,
    all_preds=all_preds,
    all_labels=all_labels,
)

prediction_path = (
    OUTPUT_DIR / "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)

attention_path = (
    OUTPUT_DIR / "test_attention_weights.npz"
)

np.savez_compressed(
    attention_path,
    MRN=metadata["mrn_test"],
    organ_names=np.asarray(
        metadata["organ_names"]
    ),
    attention=all_attention,
)

summary = {
    "fold": metadata["fold"],
    "best_checkpoint": best_checkpoint,
    "preprocessor": str(
        preprocessor_path
    ),
    "prediction_file": str(
        prediction_path
    ),
    "attention_file": str(
        attention_path
    ),
    "total_accuracy": (
        metrics["total_accuracy"]
    ),
    "global_macro_auc": (
        metrics["global_macro_auc"]
    ),
    "organ_accuracy": dict(
        zip(
            metadata["organ_names"],
            metrics["organ_accuracy"],
        )
    ),
    "organ_auc": dict(
        zip(
            metadata["organ_names"],
            [
                None
                if np.isnan(v)
                else float(v)
                for v in metrics["organ_auc"]
            ],
        )
    ),
}

with open(
    OUTPUT_DIR / "test_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print(
    f"\nPredictions saved to: "
    f"{prediction_path}"
)
print(
    f"Attention weights saved to: "
    f"{attention_path}"
)
