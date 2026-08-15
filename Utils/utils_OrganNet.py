"""
utils_OrganNet.py

Utility functions and model definitions for OrganNet.

This file contains:
- data loading and preprocessing
- OrganDataset
- FocalLoss
- OrganNet
- PyTorch Lightning training wrapper
- test/inference utilities

The implementation preserves the main logic of the original OrganNet code.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


# =============================================================================
# Dataset
# =============================================================================

class OrganDataset(Dataset):
    """
    Dataset for OrganNet.

    Parameters
    ----------
    organs : np.ndarray
        Shape [N, 7, 8].
    baseline_features : np.ndarray
        Shape [N, baseline_dim].
    labels : np.ndarray
        Shape [N, 7].
    """

    def __init__(
        self,
        organs: np.ndarray,
        baseline_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.organs = torch.as_tensor(organs, dtype=torch.float32)
        self.baseline = torch.as_tensor(
            baseline_features, dtype=torch.float32
        )
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.organs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "organs": self.organs[idx],
            "baseline": self.baseline[idx],
            "labels": self.labels[idx],
        }


# =============================================================================
# Label mapping
# =============================================================================

def transform_recist_label(x):
    """
    Convert the original organ-level label into the 3 OrganNet classes.

    Class 0: No metastasis
    Class 1: Metastasis with non-PD response
    Class 2: Metastasis with PD response

    Original label mapping:
        0   -> Class 0
        >=2 -> Class 1
        1   -> Class 2
    """
    if pd.isna(x):
        raise ValueError("Missing value found in organ-level RECIST label.")

    if x == 0:
        return 0
    elif x >= 2:
        return 1
    else:
        return 2


# =============================================================================
# Data preparation
# =============================================================================

def load_and_prepare_data(
    data_dir: str | Path,
    organ_file: str,
    clinical_file: str,
    cv: int,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.20,
    seed: int = 42,
    preprocessor_save_path: Optional[str | Path] = None,
):
    """
    Load the original OrganNet Excel files and prepare train/validation/test data.

    Notes
    -----
    This preserves the original column-reading logic:
    - organ features: TC_Bone : ATS_Primary
    - organ labels: RECISTLabel_Bone : RECISTLabel_Primary
    - clinical variables: Stage : Regimen
    - additional feature: Drug_v3
    - predefined CV split: Fold1, Fold2, ...

    Parameters
    ----------
    cv : int
        Zero-based fold index.
        cv=0 uses Fold1, cv=1 uses Fold2, etc.

    Returns
    -------
    train_loader, val_loader, test_loader, metadata
    """

    set_seed(seed)

    data_dir = Path(data_dir)
    organ_path = data_dir / organ_file
    clinical_path = data_dir / clinical_file

    df_data = pd.read_excel(organ_path)

    # -------------------------------------------------------------------------
    # Organ features
    # -------------------------------------------------------------------------
    df_vol = df_data.loc[:, "TC_Bone":"ATS_Primary"].copy()

    if df_vol.shape[1] != 7 * 8:
        raise ValueError(
            "Expected 56 organ features (7 organs x 8 features), "
            f"but found {df_vol.shape[1]}."
        )

    organ_feature_columns = list(df_vol.columns)

    # -------------------------------------------------------------------------
    # Organ labels
    # -------------------------------------------------------------------------
    df_label = df_data.loc[
        :, "RECISTLabel_Bone":"RECISTLabel_Primary"
    ].copy()

    df_label = df_label.apply(
        lambda col: col.map(transform_recist_label)
    )

    label_columns = list(df_label.columns)
    organ_names = [
        col.replace("RECISTLabel_", "", 1)
        for col in label_columns
    ]

    # -------------------------------------------------------------------------
    # Clinical features
    # -------------------------------------------------------------------------
    df_clin = pd.read_excel(clinical_path)
    df_clin_v1 = df_clin.loc[:, "MRN":"RECISTLabel"].copy()

    df_merged = pd.merge(
        df_data,
        df_clin_v1,
        on="MRN",
        how="left",
    )

    df_clin_v2 = df_merged.loc[:, "Stage":"Regimen"].copy()

    non_numeric_cols = df_clin_v2.select_dtypes(
        include=["object", "category"]
    ).columns

    df_clin_v3 = pd.get_dummies(
        df_clin_v2,
        columns=non_numeric_cols,
        drop_first=False,
    )

    # Convert only boolean dummy columns to integer.
    for col in df_clin_v3.columns:
        if pd.api.types.is_bool_dtype(df_clin_v3[col]):
            df_clin_v3[col] = df_clin_v3[col].astype(int)

    df_clin_encoded = pd.concat(
        [
            df_clin_v3.reset_index(drop=True),
            df_merged.loc[:, ["Drug_v3"]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Preserve the original behavior.
    df_clin_encoded = df_clin_encoded.dropna(axis=1)
    df_clin_encoded = df_clin_encoded.apply(
        pd.to_numeric, errors="raise"
    )

    baseline_feature_columns = list(df_clin_encoded.columns)

    # -------------------------------------------------------------------------
    # CV split
    # -------------------------------------------------------------------------
    fold_col = f"Fold{cv + 1}"

    if fold_col not in df_data.columns:
        raise KeyError(f"{fold_col} was not found in the organ feature file.")

    fold_values = (
        df_data[fold_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_idx = np.where(fold_values.eq("train"))[0]
    test_idx = np.where(fold_values.eq("test"))[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"{fold_col} must contain both 'train' and 'test'."
        )

    # -------------------------------------------------------------------------
    # Standardization
    # IMPORTANT: fit only on the predefined training partition.
    # -------------------------------------------------------------------------
    scaler_x = StandardScaler()
    scaler_baseline = StandardScaler()

    x_all = df_vol.to_numpy(dtype=np.float32)
    baseline_all = df_clin_encoded.to_numpy(dtype=np.float32)
    y_all = df_label.to_numpy(dtype=np.int64)

    x_train = scaler_x.fit_transform(x_all[train_idx])
    baseline_train = scaler_baseline.fit_transform(
        baseline_all[train_idx]
    )
    y_train = y_all[train_idx]

    x_test = scaler_x.transform(x_all[test_idx])
    baseline_test = scaler_baseline.transform(
        baseline_all[test_idx]
    )
    y_test = y_all[test_idx]

    train_dataset_full = OrganDataset(
        x_train.reshape(len(train_idx), 7, 8),
        baseline_train,
        y_train,
    )

    test_dataset = OrganDataset(
        x_test.reshape(len(test_idx), 7, 8),
        baseline_test,
        y_test,
    )

    # -------------------------------------------------------------------------
    # Internal validation split from training partition
    # -------------------------------------------------------------------------
    val_size = max(
        1,
        int(round(len(train_dataset_full) * val_ratio)),
    )
    train_size = len(train_dataset_full) - val_size

    if train_size < 1:
        raise ValueError(
            "Training partition is too small for the selected val_ratio."
        )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=generator,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    metadata = {
        "cv": cv,
        "fold": cv + 1,
        "fold_column": fold_col,
        "organ_feature_columns": organ_feature_columns,
        "baseline_feature_columns": baseline_feature_columns,
        "label_columns": label_columns,
        "organ_names": organ_names,
        "organ_dim": 8,
        "num_organs": 7,
        "baseline_dim": len(baseline_feature_columns),
        "num_classes": 3,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "mrn_test": df_data.iloc[test_idx]["MRN"].to_numpy(),
        "scaler_x": scaler_x,
        "scaler_baseline": scaler_baseline,
        "label_mapping": {
            "Class 0": "No metastasis",
            "Class 1": "Metastasis with non-PD response",
            "Class 2": "Metastasis with PD response",
        },
    }

    if preprocessor_save_path is not None:
        preprocessor_save_path = Path(preprocessor_save_path)
        preprocessor_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Remove patient-level indices/MRNs from the portable preprocessing file.
        preprocessor_to_save = {
            k: v
            for k, v in metadata.items()
            if k not in {"train_idx", "test_idx", "mrn_test"}
        }

        joblib.dump(
            preprocessor_to_save,
            preprocessor_save_path,
        )

    return train_loader, val_loader, test_loader, metadata


# =============================================================================
# Focal loss
# =============================================================================

class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(
        self,
        alpha: Optional[Sequence[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be 'mean', 'sum', or 'none'."
            )

        alpha_tensor = (
            None
            if alpha is None
            else torch.tensor(alpha, dtype=torch.float32)
        )
        self.register_buffer("alpha", alpha_tensor)

        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probs = F.softmax(logits, dim=-1)

        targets_one_hot = F.one_hot(
            targets,
            num_classes=logits.shape[-1],
        ).float()

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        pt = (probs * targets_one_hot).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            ce_loss = self.alpha[targets] * ce_loss

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# =============================================================================
# OrganNet
# =============================================================================

class OrganNet(nn.Module):
    """
    OrganNet architecture.

    Input
    -----
    organs:
        [batch, 7, 8]
    baseline:
        [batch, baseline_dim]

    Output
    ------
    logits:
        [batch, 7, 3]
    """

    def __init__(
        self,
        organ_dim: int = 8,
        num_organs: int = 7,
        baseline_dim: int = 16,
        hidden_dim: int = 8,
        num_heads: int = 4,
        num_classes: int = 3,
        memory_slots: int = 5,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_organs = num_organs
        self.num_classes = num_classes

        # Organ feature encoder
        self.organ_encoder = nn.Sequential(
            nn.Linear(organ_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Patient-level baseline feature encoder
        self.baseline_encoder = nn.Sequential(
            nn.Linear(baseline_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Intra-patient organ interaction
        self.intra_patient_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            batch_first=True,
        )

        # Organ-specific global memory
        self.global_organ_memories = nn.Parameter(
            torch.randn(
                num_organs,
                memory_slots,
                hidden_dim * 2,
            )
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        organs: torch.Tensor,
        baseline: torch.Tensor,
    ):
        batch_size, num_organs, _ = organs.shape

        organ_emb = self.organ_encoder(organs)
        baseline_emb = self.baseline_encoder(baseline)

        baseline_emb = (
            baseline_emb
            .unsqueeze(1)
            .repeat(1, num_organs, 1)
        )

        organ_features = torch.cat(
            [organ_emb, baseline_emb],
            dim=-1,
        )

        intra_output, attention_weights = self.intra_patient_attn(
            organ_features,
            organ_features,
            organ_features,
        )

        global_features = []
        memory_all = []

        for organ_idx in range(num_organs):
            current_organ = organ_features[:, organ_idx, :]
            memory = self.global_organ_memories[organ_idx]

            attn_weights = torch.softmax(
                torch.einsum(
                    "bh,mh->bm",
                    current_organ,
                    memory,
                ),
                dim=-1,
            )

            aggregated = torch.einsum(
                "bm,mh->bh",
                attn_weights,
                memory,
            )

            global_features.append(aggregated)
            memory_all.append(memory)

        global_features = torch.stack(
            global_features,
            dim=1,
        )

        combined = torch.cat(
            [
                organ_features,
                intra_output,
                global_features,
            ],
            dim=-1,
        )

        logits = self.classifier(
            combined.reshape(
                -1,
                self.hidden_dim * 6,
            )
        )

        logits = logits.reshape(
            batch_size,
            num_organs,
            self.num_classes,
        )

        return (
            logits,
            attention_weights,
            memory_all,
            combined,
        )


# =============================================================================
# Lightning training wrapper
# =============================================================================

class OrganPrognosisSystem(pl.LightningModule):
    """PyTorch Lightning wrapper used to train OrganNet."""

    def __init__(
        self,
        organ_dim: int = 8,
        num_organs: int = 7,
        baseline_dim: int = 16,
        hidden_dim: int = 8,
        num_heads: int = 4,
        num_classes: int = 3,
        memory_slots: int = 5,
        alpha: Sequence[float] = (0.1, 0.2, 0.7),
        gamma: float = 2.0,
        learning_rate: float = 2e-4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = OrganNet(
            organ_dim=organ_dim,
            num_organs=num_organs,
            baseline_dim=baseline_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            memory_slots=memory_slots,
        )

        self.criterion = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
        )

        self.train_acc = MulticlassAccuracy(
            num_classes=num_classes
        )
        self.val_acc = MulticlassAccuracy(
            num_classes=num_classes
        )

        self.train_auc = MulticlassAUROC(
            num_classes=num_classes,
            average="macro",
        )
        self.val_auc = MulticlassAUROC(
            num_classes=num_classes,
            average="macro",
        )

    def forward(
        self,
        organs: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:

        logits, _, _, _ = self.model(
            organs,
            baseline,
        )
        return logits

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:

        organs = batch["organs"]
        baseline = batch["baseline"]
        labels = batch["labels"]

        logits = self(organs, baseline)

        logits_flat = logits.reshape(
            -1,
            self.hparams.num_classes,
        )
        labels_flat = labels.reshape(-1)

        loss = self.criterion(
            logits_flat,
            labels_flat,
        )

        probs = F.softmax(
            logits_flat,
            dim=-1,
        )
        preds = probs.argmax(dim=-1)

        if stage == "train":
            acc_metric = self.train_acc
            auc_metric = self.train_auc
        else:
            acc_metric = self.val_acc
            auc_metric = self.val_auc

        acc_metric.update(
            preds,
            labels_flat,
        )
        auc_metric.update(
            probs,
            labels_flat,
        )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=organs.shape[0],
        )

        self.log(
            f"{stage}_acc",
            acc_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=organs.shape[0],
        )

        self.log(
            f"{stage}_auc",
            auc_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=organs.shape[0],
        )

        return loss

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        return self._shared_step(
            batch,
            stage="train",
        )

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        return self._shared_step(
            batch,
            stage="val",
        )

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# =============================================================================
# Test / inference
# =============================================================================

@torch.inference_mode()
def evaluate_model(
    model: OrganPrognosisSystem,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    temperature: float = 2.0,
):
    """
    Evaluate OrganNet on a test loader.

    The original evaluation code used temperature=2.0 before softmax.
    This behavior is retained here.

    Returns
    -------
    metrics : dict
    all_probs : np.ndarray
        Shape [N, 7, 3].
    all_preds : np.ndarray
        Shape [N, 7].
    all_labels : np.ndarray
        Shape [N, 7].
    all_attention : np.ndarray
        Shape [N, 7, 7].
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_attention = []

    for batch in test_loader:

        organs = batch["organs"].to(device)
        baseline = batch["baseline"].to(device)
        labels = batch["labels"].to(device)

        logits, attention, _, _ = model.model(
            organs,
            baseline,
        )

        probs = torch.softmax(
            logits / temperature,
            dim=-1,
        )

        preds = probs.argmax(dim=-1)

        all_probs.append(
            probs.cpu().numpy()
        )
        all_preds.append(
            preds.cpu().numpy()
        )
        all_labels.append(
            labels.cpu().numpy()
        )
        all_attention.append(
            attention.cpu().numpy()
        )

    all_probs = np.concatenate(
        all_probs,
        axis=0,
    )
    all_preds = np.concatenate(
        all_preds,
        axis=0,
    )
    all_labels = np.concatenate(
        all_labels,
        axis=0,
    )
    all_attention = np.concatenate(
        all_attention,
        axis=0,
    )

    total_acc = float(
        (all_preds == all_labels).mean()
    )

    organ_acc = [
        float(
            (
                all_preds[:, i]
                == all_labels[:, i]
            ).mean()
        )
        for i in range(all_labels.shape[1])
    ]

    try:
        global_auc = float(
            roc_auc_score(
                all_labels.ravel(),
                all_probs.reshape(-1, 3),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        global_auc = np.nan

    organ_aucs = []

    for i in range(all_labels.shape[1]):

        y_true = all_labels[:, i]
        y_score = all_probs[:, i, :]

        unique_classes = np.unique(y_true)

        if len(unique_classes) < 2:
            organ_aucs.append(np.nan)

        elif len(unique_classes) == 3:
            organ_aucs.append(
                float(
                    roc_auc_score(
                        y_true,
                        y_score,
                        multi_class="ovr",
                        average="macro",
                    )
                )
            )

        else:
            positive_class = unique_classes[1]

            y_true_binary = (
                y_true == positive_class
            ).astype(int)

            organ_aucs.append(
                float(
                    roc_auc_score(
                        y_true_binary,
                        y_score[:, positive_class],
                    )
                )
            )

    metrics = {
        "total_accuracy": total_acc,
        "global_macro_auc": global_auc,
        "organ_accuracy": organ_acc,
        "organ_auc": organ_aucs,
    }

    return (
        metrics,
        all_probs,
        all_preds,
        all_labels,
        all_attention,
    )


def predictions_to_dataframe(
    mrns,
    organ_names: List[str],
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    all_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Convert OrganNet predictions into a long-format table."""

    rows = []

    for patient_idx, mrn in enumerate(mrns):

        for organ_idx, organ_name in enumerate(organ_names):

            row = {
                "MRN": mrn,
                "organ": organ_name,
                "predicted_class": int(
                    all_preds[
                        patient_idx,
                        organ_idx,
                    ]
                ),
                "prob_class_0": float(
                    all_probs[
                        patient_idx,
                        organ_idx,
                        0,
                    ]
                ),
                "prob_class_1": float(
                    all_probs[
                        patient_idx,
                        organ_idx,
                        1,
                    ]
                ),
                "prob_class_2": float(
                    all_probs[
                        patient_idx,
                        organ_idx,
                        2,
                    ]
                ),
            }

            if all_labels is not None:
                row["true_class"] = int(
                    all_labels[
                        patient_idx,
                        organ_idx,
                    ]
                )

            rows.append(row)

    return pd.DataFrame(rows)
