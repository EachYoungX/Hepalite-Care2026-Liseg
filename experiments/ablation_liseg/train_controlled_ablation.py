import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import HybridDataset25D
from src.data.transforms import Robust25DTransform
from src.engine.trainer import Trainer
from src.losses.combined import HepaLiteStudentLoss25D

from experiments.ablation_liseg.variants import (
    ContextDatasetView,
    ControlledStudent,
    SymmetricDiceBCELoss25D,
)


DATA_DIR = ROOT / "dataset/processed/training_set"
OUTPUT_DIR = ROOT / "artifacts/experiments/ablation_liseg/training"
VENDORS = ("Vendor_A", "Vendor_B1", "Vendor_B2")
SEED = 42

VARIANTS = {
    "real_1slice": {"context_slices": 1, "use_coordconv": True, "use_se": True},
    "real_3slice": {"context_slices": 3, "use_coordconv": True, "use_se": True},
    "real_5slice": {"context_slices": 5, "use_coordconv": True, "use_se": True},
    "real_5slice_no_coord": {"context_slices": 5, "use_coordconv": False, "use_se": True},
    "real_5slice_no_se": {"context_slices": 5, "use_coordconv": True, "use_se": False},
    "real_5slice_dice_bce": {
        "context_slices": 5,
        "use_coordconv": True,
        "use_se": True,
        "loss": "dice_bce",
    },
}


def real_ids():
    cases = []
    for vendor in VENDORS:
        for path in sorted((DATA_DIR / vendor).glob("*_mask.npy")):
            if "_pseudo_mask" not in path.name:
                cases.append(path.name.replace("_mask.npy", ""))
    return sorted(set(cases))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_variant(name, config, epochs):
    variant_dir = OUTPUT_DIR / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "variant": name,
        "config": config,
        "epochs": epochs,
        "seed": SEED,
        "supervision": "CARE labeled cases only",
        "pseudo_ratio": 0,
    }
    with open(variant_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    cases = real_ids()
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_zero, (train_indices, val_indices) in enumerate(splitter.split(cases)):
        fold = fold_zero + 1
        checkpoint = variant_dir / f"fold_{fold}" / "checkpoints/best_model.pth"
        if checkpoint.exists():
            print(f"[{name}] Fold {fold}: checkpoint exists, skipping")
            continue
        seed_everything(SEED + fold)
        train_names = [cases[index] for index in train_indices]
        val_names = [cases[index] for index in val_indices]

        base_train = HybridDataset25D(
            train_dir=str(DATA_DIR),
            vendors=VENDORS,
            val_dir=None,
            transform=Robust25DTransform(),
            case_ids=train_names,
            is_train=True,
        )
        base_train.all_pseudo_samples = []
        base_train.set_epoch(0, pseudo_ratio=0)
        train_dataset = ContextDatasetView(base_train, config["context_slices"])
        val_dataset = HybridDataset25D(
            train_dir=str(DATA_DIR),
            vendors=VENDORS,
            val_dir=None,
            transform=None,
            case_ids=val_names,
            is_train=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        model = ControlledStudent(
            context_slices=config["context_slices"],
            use_coordconv=config["use_coordconv"],
            use_se=config["use_se"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5
        )
        criterion = (
            SymmetricDiceBCELoss25D()
            if config.get("loss") == "dice_bce"
            else HepaLiteStudentLoss25D()
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion={"seg": criterion.to(device)},
            device=device,
            exp_root=str(variant_dir),
            fold_num=fold,
            val_threshold=0.5,
            pseudo_ratio=0,
            pure_real_baseline=True,
        )
        score = trainer.train_one_fold(
            train_loader,
            val_loader,
            num_epochs=epochs,
            scheduler=scheduler,
        )
        fold_scores.append(score)
        del model, optimizer, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if fold_scores:
        print(f"[{name}] newly trained fold mean best Dice: {np.mean(fold_scores):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["real_5slice"],
    )
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    os.chdir(ROOT)
    for name in args.variants:
        train_variant(name, VARIANTS[name], args.epochs)


if __name__ == "__main__":
    main()
