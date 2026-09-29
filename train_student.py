import argparse
import os
import glob
import random
import warnings

# Suppress a compatibility warning emitted by torch.load.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data.dataset import HybridDataset25D
from src.data.transforms import Robust25DTransform
from src.models.models import HepaLiteStudent25D
from src.losses.combined import HepaLiteStudentLoss25D
from src.engine.trainer import Trainer

PARAMS = {
    "model_identifier": "20260503_HepaLite_Student_SemiSup_V1",
    "train_data_dir": "dataset/processed/training_set",
    "val_data_dir": "dataset/processed/validation_set",
    "vendors": ("Vendor_A", "Vendor_B1", "Vendor_B2"),
    "n_splits": 5,
    "max_epochs": 60,
    "batch_size": 8,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "init_features": 16,
    "in_channels": 5,
    "out_classes": 1,
    "pseudo_ratio": 20,
    "val_pseudo_weight_scale": 0.35,
    "num_workers": 4,
    # Set to True for the manually labeled baseline.
    "pure_real_baseline": False,
    "use_se": True,
    "seed": 42,
}


class SimpleConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_real_ids(data_dir, vendors):
    ids = []
    for vendor in vendors:
        mask_files = glob.glob(os.path.join(data_dir, vendor, "*_mask.npy"))
        for f in mask_files:
            fname = os.path.basename(f)
            if "_pseudo_mask" not in fname:
                ids.append(fname.replace("_mask.npy", ""))
    return sorted(list(set(ids)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(use_se=None):
    """Train the student model with cross-validation."""
    cfg = SimpleConfig(**PARAMS)
    if use_se is not None:
        cfg.use_se = use_se
    if not cfg.use_se:
        cfg.model_identifier = f"{cfg.model_identifier}_NoSE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_sup_ids = get_real_ids(cfg.train_data_dir, cfg.vendors)
    print(f"=> [DATA] Manual 3D cases: {len(real_sup_ids)}")

    if cfg.pure_real_baseline:
        print(
            "[CONFIG] Manual-label supervised baseline; pseudo-labels are disabled."
        )
        p_ratio = 0
    else:
        print(
            f"[CONFIG] Semi-supervised transductive training; pseudo-label limit: {cfg.pseudo_ratio}"
        )
        p_ratio = cfg.pseudo_ratio

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.path.join(
        "artifacts", "training_runs", f"{timestamp}_{cfg.model_identifier}"
    )
    os.makedirs(exp_root, exist_ok=True)

    kfold = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(real_sup_ids)):
        fold_seed = cfg.seed + fold_idx + 1
        seed_everything(fold_seed)
        print(f"\n{'='*40}\n Starting fold {fold_idx + 1}/{cfg.n_splits} \n{'='*40}")

        train_real_names = [real_sup_ids[i] for i in train_idx]
        val_real_names = [real_sup_ids[i] for i in val_idx]

        train_dataset = HybridDataset25D(
            train_dir=cfg.train_data_dir,
            vendors=cfg.vendors,
            val_dir=(
                None if cfg.pure_real_baseline else cfg.val_data_dir
            ),  # The baseline excludes the external validation set.
            transform=Robust25DTransform(),
            case_ids=train_real_names,
            is_train=True,
            val_pseudo_weight_scale=cfg.val_pseudo_weight_scale,
            sampling_seed=fold_seed,
        )

        val_dataset = HybridDataset25D(
            train_dir=cfg.train_data_dir,
            vendors=cfg.vendors,
            val_dir=None,
            transform=None,
            case_ids=val_real_names,
            is_train=False,
            sampling_seed=fold_seed,
        )

        loader_generator = torch.Generator()
        loader_generator.manual_seed(fold_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            generator=loader_generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        model = HepaLiteStudent25D(
            in_channels=cfg.in_channels,
            out_classes=cfg.out_classes,
            init_features=cfg.init_features,
            deep_supervision=True,
            use_se=cfg.use_se,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        criteria = {"seg": HepaLiteStudentLoss25D().to(device)}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=10, factor=0.5
        )

        trainer = Trainer(
            model,
            optimizer,
            criteria,
            device,
            exp_root,
            fold_idx + 1,
            val_threshold=0.5,
            pseudo_ratio=cfg.pseudo_ratio,
            pure_real_baseline=cfg.pure_real_baseline,
        )

        best_dice = trainer.train_one_fold(
            train_loader, val_loader, num_epochs=cfg.max_epochs, scheduler=scheduler
        )
        fold_results.append(best_dice)

        del model, optimizer, trainer
        torch.cuda.empty_cache()

    print(f"\n{'='*40}\n Training complete. Mean best Dice: {np.mean(fold_results):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-se",
        action="store_true",
        help="Disable every SE block for the controlled semi-supervised experiment.",
    )
    args = parser.parse_args()
    main(use_se=not args.no_se)
