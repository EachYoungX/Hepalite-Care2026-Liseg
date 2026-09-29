import os
import glob
import csv

from _project import PROJECT_ROOT  # noqa: F401

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter1d, label
from tqdm import tqdm

from src.models.models import HepaLiteStudent25D


CONFIG = {
    "student_exp_path": "artifacts/training_runs/20260612_104443_20260503_HepaLite_Student_SemiSup_V1",
    "processed_val_dir": "dataset/processed/validation_set",
    "out_csv": "artifacts/analysis/audits/tta_flip_consistency_audit.csv",
    "ged4_idx": 6,
    "max_cases": 12,
    "threshold": 0.5,
    "min_size_filter": 100,
    "min_overlap_iou": 0.08,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ged4(path):
    arr = np.load(path)
    if arr.ndim == 4:
        return arr[CONFIG["ged4_idx"]] if arr.shape[0] > CONFIG["ged4_idx"] else arr[0]
    return arr


def load_ensemble():
    models = []
    paths = sorted(
        glob.glob(
            os.path.join(CONFIG["student_exp_path"], "fold_*", "checkpoints", "best_model.pth")
        )
    )
    for path in paths:
        model = HepaLiteStudent25D(
            in_channels=5, out_classes=1, init_features=16, deep_supervision=True
        )
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError(f"No fold checkpoints found in {CONFIG['student_exp_path']}")
    return models


def calculate_iou_2d(mask_a, mask_b):
    inter = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    return float(inter / (union + 1e-5)) if union > 0 else 0.0


def infer_probs(img_3d, models, use_lr_flip=False):
    d, h, w = img_3d.shape
    probs_3d = np.zeros((d, h, w), dtype=np.float32)

    for z in range(d):
        z_prev2 = max(0, z - 2)
        z_prev1 = max(0, z - 1)
        z_next1 = min(d - 1, z + 1)
        z_next2 = min(d - 1, z + 2)
        slice_5d = img_3d[[z_prev2, z_prev1, z, z_next1, z_next2]].copy()
        if use_lr_flip:
            slice_5d = slice_5d[:, :, ::-1].copy()

        input_tensor = torch.from_numpy(slice_5d).unsqueeze(0).to(DEVICE).float()
        fold_probs = []
        with torch.no_grad():
            for model in models:
                logits = model(input_tensor)
                if isinstance(logits, list):
                    logits = logits[0]
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
                if use_lr_flip:
                    prob = prob[:, ::-1]
                fold_probs.append(prob)
        probs_3d[z] = np.mean(fold_probs, axis=0)

    return probs_3d


def postprocess(prob_3d):
    pred = np.zeros_like(prob_3d, dtype=np.uint8)
    for z in range(prob_3d.shape[0]):
        binary_slice = (prob_3d[z] > CONFIG["threshold"]).astype(np.uint8)
        if np.sum(binary_slice) > 0:
            pred[z] = binary_opening(binary_slice, structure=np.ones((3, 3)))

    labeled, num_features = label(pred)
    if num_features > 0:
        volumes = np.bincount(labeled.ravel())
        volumes[0] = 0
        pred[labeled != volumes.argmax()] = 0

    pred = binary_fill_holes(pred).astype(np.uint8)
    areas = np.array([np.sum(pred[z]) for z in range(pred.shape[0])])
    smooth_areas = gaussian_filter1d(areas, sigma=2.0, mode="nearest")

    liver_start, liver_end = 0, pred.shape[0] - 1
    for z in range(pred.shape[0] - 2):
        if areas[z] > 100 and areas[z + 1] > 100 and areas[z + 2] > 100:
            liver_start = z
            break
    for z in range(pred.shape[0] - 1, 1, -1):
        if areas[z] > 100 and areas[z - 1] > 100 and areas[z - 2] > 100:
            liver_end = z
            break

    for z in range(pred.shape[0]):
        if z < liver_start or z > liver_end or smooth_areas[z] < CONFIG["min_size_filter"]:
            pred[z] = 0
            continue
        if z > liver_start and np.sum(pred[z]) > 0 and np.sum(pred[z - 1]) > 0:
            if calculate_iou_2d(pred[z], pred[z - 1]) < CONFIG["min_overlap_iou"]:
                pred[z] = 0

    return pred


def dice(a, b):
    inter = np.sum((a > 0) & (b > 0))
    denom = np.sum(a > 0) + np.sum(b > 0)
    return float((2 * inter + 1e-5) / (denom + 1e-5))


def summarize_mask(mask):
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    fg_z = np.where(areas > 0)[0]
    if len(fg_z) == 0:
        return 0, 0, -1, -1
    return int(mask.sum()), int(len(fg_z)), int(fg_z.min()), int(fg_z.max())


def main():
    models = load_ensemble()
    data_files = sorted(glob.glob(os.path.join(CONFIG["processed_val_dir"], "*", "*_data.npy")))

    existing_masks = {
        os.path.basename(path).replace("_mask.npy", ""): np.load(path)
        for path in glob.glob("artifacts/predictions/temp_predictions/*_mask.npy")
    }

    ranked = []
    for path in data_files:
        case = os.path.basename(path).replace("_data.npy", "")
        if case in existing_masks:
            vol, zc, _, _ = summarize_mask(existing_masks[case])
            ranked.append((abs(vol - 79000) + abs(zc - 22) * 5000, path))
        else:
            ranked.append((0, path))
    selected = [path for _, path in sorted(ranked, reverse=True)[: CONFIG["max_cases"]]]

    rows = []
    for path in tqdm(selected, desc="TTA LR flip audit"):
        case = os.path.basename(path).replace("_data.npy", "")
        img = load_ged4(path)
        base_prob = infer_probs(img, models, use_lr_flip=False)
        flip_prob = infer_probs(img, models, use_lr_flip=True)
        tta_prob = 0.5 * (base_prob + flip_prob)

        base_mask = postprocess(base_prob)
        flip_mask = postprocess(flip_prob)
        tta_mask = postprocess(tta_prob)

        base_vol, base_zc, base_zs, base_ze = summarize_mask(base_mask)
        tta_vol, tta_zc, tta_zs, tta_ze = summarize_mask(tta_mask)
        rows.append(
            {
                "case": case,
                "base_vs_flip_dice": round(dice(base_mask, flip_mask), 5),
                "base_vs_tta_dice": round(dice(base_mask, tta_mask), 5),
                "base_volume": base_vol,
                "tta_volume": tta_vol,
                "volume_delta_pct": round((tta_vol - base_vol) / max(base_vol, 1) * 100.0, 2),
                "base_z_count": base_zc,
                "tta_z_count": tta_zc,
                "base_z_span": f"{base_zs}-{base_ze}",
                "tta_z_span": f"{tta_zs}-{tta_ze}",
            }
        )

    os.makedirs(os.path.dirname(CONFIG["out_csv"]), exist_ok=True)
    with open(CONFIG["out_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved TTA audit: {CONFIG['out_csv']}")
    print("Lowest base-vs-flip agreement cases:")
    for row in sorted(rows, key=lambda r: r["base_vs_flip_dice"])[:8]:
        print(
            f"{row['case']} | flip_dice={row['base_vs_flip_dice']} | "
            f"tta_delta={row['volume_delta_pct']}% | z {row['base_z_span']} -> {row['tta_z_span']}"
        )


if __name__ == "__main__":
    main()
