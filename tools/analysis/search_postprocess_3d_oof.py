import csv
import glob
import itertools
import os

from _project import PROJECT_ROOT  # noqa: F401

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter1d, label
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.models.models import HepaLiteStudent25D


STUDENT_EXP_PATH = "artifacts/training_runs/20260612_104443_20260503_HepaLite_Student_SemiSup_V1"
DATA_DIR = "dataset/processed/training_set"
VENDORS = ("Vendor_A", "Vendor_B1", "Vendor_B2")
CACHE_DIR = "artifacts/analysis/postprocess_oof_cache"
OUT_CSV = "artifacts/analysis/audits/postprocess_3d_oof_grid.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS = [0.35, 0.40, 0.45, 0.50]
MIN_SIZE_FILTERS = [30, 50, 80, 100]
SPAN_MIN_AREAS = [50, 100]
SPAN_CONSECUTIVE = [2, 3]
MIN_OVERLAP_IOUS = [0.03, 0.05, 0.08]
OPENING_FLAGS = [False, True]
FILL_HOLES_FLAGS = [True]

CURRENT_CONFIG = {
    "threshold": 0.50,
    "opening": True,
    "fill_holes": True,
    "min_size_filter": 100,
    "span_min_area": 100,
    "span_consecutive": 3,
    "min_overlap_iou": 0.08,
}


def get_real_ids():
    ids = []
    for vendor in VENDORS:
        for f in glob.glob(os.path.join(DATA_DIR, vendor, "*_mask.npy")):
            fname = os.path.basename(f)
            if "_pseudo_mask" not in fname:
                ids.append(fname.replace("_mask.npy", ""))
    return sorted(set(ids))


def find_case_path(case_name, suffix):
    matches = glob.glob(os.path.join(DATA_DIR, "*", f"{case_name}_{suffix}.npy"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {case_name}_{suffix}.npy")
    return matches[0]


def load_3d(path):
    arr = np.load(path)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def load_model(fold_idx):
    ckpt = os.path.join(
        STUDENT_EXP_PATH, f"fold_{fold_idx}", "checkpoints", "best_model.pth"
    )
    model = HepaLiteStudent25D(
        in_channels=5, out_classes=1, init_features=16, deep_supervision=True
    )
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_prob_volume(model, img_3d):
    d, h, w = img_3d.shape
    prob_3d = np.zeros((d, h, w), dtype=np.float32)
    for z in range(d):
        z_prev2 = max(0, z - 2)
        z_prev1 = max(0, z - 1)
        z_next1 = min(d - 1, z + 1)
        z_next2 = min(d - 1, z + 2)
        slice_5d = img_3d[[z_prev2, z_prev1, z, z_next1, z_next2]].copy()
        input_t = torch.from_numpy(slice_5d).unsqueeze(0).to(DEVICE).float()
        logits = model(input_t)
        prob_3d[z] = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob_3d


def ensure_oof_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    real_ids = get_real_ids()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_zero, (_, val_idx) in enumerate(kfold.split(real_ids)):
        fold_idx = fold_zero + 1
        val_names = [real_ids[i] for i in val_idx]
        missing = [
            name
            for name in val_names
            if not os.path.exists(os.path.join(CACHE_DIR, f"{name}_prob.npy"))
        ]
        if not missing:
            continue

        print(f"=> Caching OOF probabilities for fold {fold_idx}: {missing}")
        model = load_model(fold_idx)
        for case_name in tqdm(missing, desc=f"Fold {fold_idx} prob cache"):
            img = load_3d(find_case_path(case_name, "data"))
            prob = predict_prob_volume(model, img)
            np.save(os.path.join(CACHE_DIR, f"{case_name}_prob.npy"), prob)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def calculate_iou_2d(mask_a, mask_b):
    intersect = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return 0.0
    return intersect / (union + 1e-5)


def postprocess_3d(
    prob_3d,
    threshold,
    opening,
    fill_holes,
    min_size_filter,
    span_min_area,
    span_consecutive,
    min_overlap_iou,
):
    d, h, w = prob_3d.shape
    pred = np.zeros((d, h, w), dtype=np.uint8)
    for z in range(d):
        binary_slice = (prob_3d[z] > threshold).astype(np.uint8)
        if binary_slice.sum() == 0:
            continue
        if opening:
            binary_slice = binary_opening(
                binary_slice, structure=np.ones((3, 3))
            ).astype(np.uint8)
        pred[z] = binary_slice

    labeled_3d, num_features = label(pred)
    if num_features > 0:
        volumes = np.bincount(labeled_3d.ravel())
        volumes[0] = 0
        pred[labeled_3d != volumes.argmax()] = 0

    if fill_holes:
        pred = binary_fill_holes(pred).astype(np.uint8)

    initial_areas = np.asarray([pred[z].sum() for z in range(d)], dtype=np.float32)
    smooth_areas = gaussian_filter1d(initial_areas, sigma=2.0, mode="nearest")

    liver_start, liver_end = 0, d - 1
    found_start = False
    for z in range(0, d - span_consecutive + 1):
        if np.all(initial_areas[z : z + span_consecutive] > span_min_area):
            liver_start = z
            found_start = True
            break
    found_end = False
    for z in range(d - 1, span_consecutive - 2, -1):
        window = initial_areas[z - span_consecutive + 1 : z + 1]
        if np.all(window > span_min_area):
            liver_end = z
            found_end = True
            break

    if found_start and found_end:
        for z in range(d):
            if z < liver_start or z > liver_end or smooth_areas[z] < min_size_filter:
                pred[z] = 0
                continue
            if z > liver_start and pred[z].sum() > 0 and pred[z - 1].sum() > 0:
                if calculate_iou_2d(pred[z], pred[z - 1]) < min_overlap_iou:
                    pred[z] = 0
    else:
        # If no stable span was found, keep CCA/fill output rather than deleting all.
        for z in range(d):
            if smooth_areas[z] < min_size_filter:
                pred[z] = 0

    return pred.astype(np.uint8)


def dice_3d(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    intersect = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0
    return (2.0 * intersect + 1e-5) / (union + 1e-5)


def evaluate_config(config, cases):
    dices = []
    pred_volumes = []
    gt_volumes = []
    pred_slices = []
    for case_name in cases:
        prob = np.load(os.path.join(CACHE_DIR, f"{case_name}_prob.npy"))
        gt = load_3d(find_case_path(case_name, "mask")).astype(np.uint8)
        pred = postprocess_3d(prob, **config)
        dices.append(dice_3d(pred, gt))
        pred_volumes.append(int(pred.sum()))
        gt_volumes.append(int(gt.sum()))
        pred_slices.append(int((pred.reshape(pred.shape[0], -1).sum(axis=1) > 0).sum()))
    volume_ratio = np.sum(pred_volumes) / max(np.sum(gt_volumes), 1)
    return {
        "mean_dice": float(np.mean(dices)),
        "median_dice": float(np.median(dices)),
        "min_dice": float(np.min(dices)),
        "volume_ratio": float(volume_ratio),
        "median_pred_slices": float(np.median(pred_slices)),
    }


def main():
    ensure_oof_cache()
    cases = get_real_ids()
    rows = []

    current_metrics = evaluate_config(CURRENT_CONFIG, cases)
    print("\n=== Current Online-like Config ===")
    print({**CURRENT_CONFIG, **current_metrics})

    grid_iter = itertools.product(
        THRESHOLDS,
        OPENING_FLAGS,
        FILL_HOLES_FLAGS,
        MIN_SIZE_FILTERS,
        SPAN_MIN_AREAS,
        SPAN_CONSECUTIVE,
        MIN_OVERLAP_IOUS,
    )
    total = (
        len(THRESHOLDS)
        * len(OPENING_FLAGS)
        * len(FILL_HOLES_FLAGS)
        * len(MIN_SIZE_FILTERS)
        * len(SPAN_MIN_AREAS)
        * len(SPAN_CONSECUTIVE)
        * len(MIN_OVERLAP_IOUS)
    )

    for (
        threshold,
        opening,
        fill_holes,
        min_size_filter,
        span_min_area,
        span_consecutive,
        min_overlap_iou,
    ) in tqdm(grid_iter, total=total, desc="3D OOF grid"):
        config = {
            "threshold": threshold,
            "opening": opening,
            "fill_holes": fill_holes,
            "min_size_filter": min_size_filter,
            "span_min_area": span_min_area,
            "span_consecutive": span_consecutive,
            "min_overlap_iou": min_overlap_iou,
        }
        metrics = evaluate_config(config, cases)
        rows.append({**config, **metrics})

    rows = sorted(rows, key=lambda r: r["mean_dice"], reverse=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved grid CSV: {OUT_CSV}")
    print("\n=== Top 10 3D OOF Postprocess Configs ===")
    for i, row in enumerate(rows[:10], 1):
        print(
            f"{i:02d}. dice={row['mean_dice']:.5f} | med={row['median_dice']:.5f} | "
            f"min={row['min_dice']:.5f} | vol_ratio={row['volume_ratio']:.3f} | "
            f"thr={row['threshold']} opening={row['opening']} min_size={row['min_size_filter']} "
            f"span_area={row['span_min_area']} span_n={row['span_consecutive']} "
            f"iou={row['min_overlap_iou']}"
        )

    best = rows[0]
    print("\n=== Suggested Config ===")
    print(best)


if __name__ == "__main__":
    main()
