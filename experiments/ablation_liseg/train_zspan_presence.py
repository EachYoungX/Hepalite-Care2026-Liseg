import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.ablation_liseg import run_ablation as core


OUTPUT_DIR = core.LOCAL_ARTIFACT_DIR / "zspan_presence"
OOF_DIR = OUTPUT_DIR / "oof_probabilities"
SEED = 42
EPOCHS = 35


class SlicePresenceNet25D(nn.Module):
    """Small position-agnostic classifier for liver presence in the center slice."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, 1)

    def forward(self, image):
        return self.classifier(self.features(image).flatten(1)).squeeze(1)


class PresenceDataset(Dataset):
    def __init__(self, records, augment=False):
        self.augment = augment
        self.cases = []
        self.samples = []
        for case_index, record in enumerate(records):
            image = core.load_volume(record["data_path"], ged4=True).astype(np.float32)
            mask = core.load_volume(record["mask_path"]).astype(np.uint8)
            labels = np.any(mask > 0, axis=(1, 2)).astype(np.float32)
            self.cases.append((record["case"], image, labels))
            self.samples.extend((case_index, z) for z in range(image.shape[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        case_index, z = self.samples[index]
        _, image, labels = self.cases[case_index]
        depth = image.shape[0]
        indices = [
            max(0, z - 2),
            max(0, z - 1),
            z,
            min(depth - 1, z + 1),
            min(depth - 1, z + 2),
        ]
        context = image[indices].copy()
        if self.augment:
            if random.random() < 0.5:
                context = context[:, :, ::-1].copy()
            if random.random() < 0.4:
                context *= random.uniform(0.9, 1.1)
            if random.random() < 0.3:
                context += np.random.normal(0.0, 0.01, context.shape).astype(np.float32)
            context = np.clip(context, 0.0, 1.0)
        return torch.from_numpy(context), torch.tensor(labels[z]), case_index, z


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    locations = []
    for images, targets, case_indices, z_indices in loader:
        logits = model(images.float().to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(targets.numpy().tolist())
        locations.extend(zip(case_indices.numpy().tolist(), z_indices.numpy().tolist()))
    return np.asarray(probabilities), np.asarray(labels), locations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    records = core.real_case_records()
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []

    for fold_zero, (train_indices, val_indices) in enumerate(splitter.split(records)):
        fold = fold_zero + 1
        fold_dir = OUTPUT_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = fold_dir / "best_model.pth"
        train_records = [records[index] for index in train_indices]
        val_records = [records[index] for index in val_indices]
        train_dataset = PresenceDataset(train_records, augment=True)
        val_dataset = PresenceDataset(val_records, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

        seed_everything(SEED + fold)
        model = SlicePresenceNet25D().to(device)
        positive = sum(label for _, _, labels in train_dataset.cases for label in labels)
        total = sum(len(labels) for _, _, labels in train_dataset.cases)
        pos_weight = torch.tensor([(total - positive) / max(positive, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best_auc = -1.0
        history = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            for images, targets, _, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.float().to(device))
                loss = criterion(logits, targets.float().to(device))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            probabilities, labels, _ = predict_loader(model, val_loader, device)
            auc = roc_auc_score(labels, probabilities)
            balanced = balanced_accuracy_score(labels, probabilities >= 0.5)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_auc": float(auc),
                    "val_balanced_accuracy": float(balanced),
                }
            )
            if auc > best_auc:
                best_auc = float(auc)
                torch.save(model.state_dict(), checkpoint)

        with open(fold_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        probabilities, labels, locations = predict_loader(model, val_loader, device)
        case_probabilities = {
            case: np.zeros_like(case_labels, dtype=np.float32)
            for case, _, case_labels in val_dataset.cases
        }
        for probability, (case_index, z) in zip(probabilities, locations):
            case = val_dataset.cases[case_index][0]
            case_probabilities[case][z] = probability
        for case, probability in case_probabilities.items():
            np.save(OOF_DIR / f"{case}_prob.npy", probability)
        summary.append(
            {
                "fold": fold,
                "best_auc": best_auc,
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, probabilities >= 0.5)
                ),
                "checkpoint": str(checkpoint.relative_to(ROOT)),
            }
        )
        print(
            f"Fold {fold}: AUC={summary[-1]['best_auc']:.4f}, "
            f"balanced_acc={summary[-1]['balanced_accuracy']:.4f}"
        )

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Mean OOF AUC: {np.mean([row['best_auc'] for row in summary]):.4f}")


if __name__ == "__main__":
    main()
