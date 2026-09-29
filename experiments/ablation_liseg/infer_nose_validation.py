import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.models import HepaLiteStudent25D
from src.postprocess.topology import postprocess_prob_volume


EXPERIMENT = ROOT / "artifacts/training_runs/20260620_223719_20260503_HepaLite_Student_SemiSup_V1_NoSE"
PROCESSED_VAL_DIR = ROOT / "dataset/processed/validation_set"
OUTPUT_DIR = ROOT / "artifacts/evaluation/nose_validation/temp_predictions"
THRESHOLD = 0.375
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def load_ensemble():
    checkpoint_paths = sorted(EXPERIMENT.glob("fold_*/checkpoints/best_model.pth"))
    if len(checkpoint_paths) != 5:
        raise RuntimeError(f"Expected 5 NoSE checkpoints, found {len(checkpoint_paths)}")
    models = []
    for checkpoint_path in checkpoint_paths:
        model = HepaLiteStudent25D(
            in_channels=5,
            out_classes=1,
            init_features=16,
            deep_supervision=True,
            use_se=False,
        )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE).eval()
        models.append(model)
    return models


@torch.no_grad()
def predict_case(case_path, models):
    image = np.load(case_path)
    if image.ndim == 4:
        image = image[6] if image.shape[0] > 6 else image[0]
    depth = image.shape[0]
    probability = np.zeros_like(image, dtype=np.float32)
    for z in range(depth):
        indices = [
            max(0, z - 2),
            max(0, z - 1),
            z,
            min(depth - 1, z + 1),
            min(depth - 1, z + 2),
        ]
        tensor = torch.from_numpy(image[indices].copy()).unsqueeze(0).float().to(DEVICE)
        probabilities = [torch.sigmoid(model(tensor))[0, 0].cpu().numpy() for model in models]
        probability[z] = np.mean(probabilities, axis=0)
    return postprocess_prob_volume(
        probability,
        threshold=THRESHOLD,
        opening=True,
        fill_holes=True,
        min_size_filter=50,
        min_overlap_iou=0.03,
        centroid_jump_limit=40.0,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_ensemble()
    cases = sorted(PROCESSED_VAL_DIR.glob("*/*_data.npy"))
    if len(cases) != 60:
        raise RuntimeError(f"Expected 60 validation cases, found {len(cases)}")
    for case_path in tqdm(cases, desc="NoSE validation inference"):
        case = case_path.name.replace("_data.npy", "")
        prediction = predict_case(case_path, models)
        np.save(OUTPUT_DIR / f"{case}_mask.npy", prediction.astype(np.uint8))
    print(f"Saved {len(cases)} generic NoSE predictions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
