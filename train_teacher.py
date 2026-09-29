import os
import sys
import glob
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data.dataset import HybridDataset25D
from src.data.transforms import Robust25DTransform
from src.models.models import HepaLite25D
from src.losses.combined import HepaLiteCombinedLoss25D
from src.engine.teacher_trainer import TeacherTrainer

PARAMS = {
    "model_name": "Teacher_GED4_Base_Clean_5Fold",
    "data_dir": "dataset/processed/training_set",
    "vendors": ["Vendor_A", "Vendor_B1", "Vendor_B2"],
    "n_splits": 5,
    "epochs": 60,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "init_features": 16,
    "in_channels": 5,
    "out_classes": 1,
    "num_workers": 4,
    "seed": 42,
}


class SimpleConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_real_ids(data_dir, vendors):
    """Return unique case identifiers that have manual masks."""
    ids = []
    for v in vendors:
        mask_paths = glob.glob(os.path.join(data_dir, v, "*_mask.npy"))
        for p in mask_paths:
            fname = os.path.basename(p)
            if "_pseudo" not in fname:
                ids.append(fname.replace("_mask.npy", ""))
    return sorted(list(set(ids)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SimpleConfig(**PARAMS)

    real_ids = get_real_ids(cfg.data_dir, cfg.vendors)
    print("=> Starting teacher-model training.")
    print(f"=> Manual cases in the target domain: {len(real_ids)}")
    print("=> Target modality: single-modality GED4 with five-channel 2.5D context")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.path.join(
        "artifacts", "training_runs", f"{timestamp}_{cfg.model_name}"
    )

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (t_idx, v_idx) in enumerate(kf.split(real_ids)):
        fold_seed = cfg.seed + fold_idx + 1
        seed_everything(fold_seed)
        print(f"\n{'#'*30}\nTraining fold {fold_idx + 1}/{cfg.n_splits}\n{'#'*30}")

        train_names = [real_ids[i] for i in t_idx]
        val_names = [real_ids[i] for i in v_idx]

        train_ds = HybridDataset25D(
            train_dir=cfg.data_dir,
            vendors=cfg.vendors,
            transform=Robust25DTransform(),
            case_ids=train_names,
            is_train=True,
            sampling_seed=fold_seed,
        )
        train_ds.all_pseudo_samples = []
        train_ds.set_epoch(0, pseudo_ratio=0)

        val_ds = HybridDataset25D(
            train_dir=cfg.data_dir,
            vendors=cfg.vendors,
            transform=None,
            case_ids=val_names,
            is_train=False,
            sampling_seed=fold_seed,
        )

        loader_generator = torch.Generator()
        loader_generator.manual_seed(fold_seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            generator=loader_generator,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = HepaLite25D(
            in_channels=cfg.in_channels,
            out_classes=cfg.out_classes,
            init_features=cfg.init_features,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5, verbose=False
        )

        criteria = {
            "seg": HepaLiteCombinedLoss25D().to(device),
            "staging": nn.CrossEntropyLoss().to(device),
        }

        fold_dir = os.path.join(exp_root, f"fold_{fold_idx+1}")
        trainer = TeacherTrainer(model, optimizer, criteria, device, fold_dir)

        best_dice = trainer.train(
            train_loader, val_loader, num_epochs=cfg.epochs, scheduler=scheduler
        )
        fold_results.append(best_dice)

        del model, optimizer, trainer
        torch.cuda.empty_cache()

    print(f"\n{'='*40}")
    print(" Teacher five-fold training complete.")
    print(f" Mean validation Dice: {np.mean(fold_results):.4f}")
    print(f" Checkpoints saved to: {exp_root}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
