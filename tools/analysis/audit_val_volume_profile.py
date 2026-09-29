import csv
import glob
import os

from _project import PROJECT_ROOT  # noqa: F401

import numpy as np


PROCESSED_VAL_DIR = "dataset/processed/validation_set"
TEMP_PRED_DIR = "artifacts/predictions/temp_predictions"
PROCESSED_TRAIN_DIR = "dataset/processed/training_set"
OUT_CSV = "artifacts/analysis/audits/val_volume_profile_audit.csv"


def load_3d(path):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 4:
        arr = arr[0]
    return np.asarray(arr)


def summarize_mask(mask):
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    nonzero = areas > 0
    if nonzero.any():
        start = int(np.argmax(nonzero))
        end = int(len(nonzero) - 1 - np.argmax(nonzero[::-1]))
        max_area = int(areas.max())
        mean_nonzero_area = float(areas[nonzero].mean())
    else:
        start, end, max_area, mean_nonzero_area = -1, -1, 0, 0.0
    return {
        "volume": int(areas.sum()),
        "nonzero_slices": int(nonzero.sum()),
        "start_z": start,
        "end_z": end,
        "max_area": max_area,
        "mean_nonzero_area": mean_nonzero_area,
    }


def percentile_line(values):
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"n={len(arr)} | mean={arr.mean():.2f} | "
        f"p10={np.percentile(arr, 10):.2f} | p50={np.percentile(arr, 50):.2f} | "
        f"p90={np.percentile(arr, 90):.2f}"
    )


def main():
    rows = []
    pred_volumes = []
    pred_slices = []
    pred_max_areas = []
    fg_ratios = []

    val_data_paths = sorted(
        glob.glob(os.path.join(PROCESSED_VAL_DIR, "*", "*_data.npy"))
    )
    for data_path in val_data_paths:
        case_name = os.path.basename(data_path).replace("_data.npy", "")
        pred_path = os.path.join(TEMP_PRED_DIR, f"{case_name}_mask.npy")
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing prediction for {case_name}")
            continue

        img = load_3d(data_path)
        pred = load_3d(pred_path).astype(np.uint8)
        pred_stats = summarize_mask(pred)
        img_nonzero = int((img > 1e-6).sum())
        fg_ratio = pred_stats["volume"] / max(img_nonzero, 1)

        row = {
            "case": case_name,
            "pred_volume": pred_stats["volume"],
            "pred_nonzero_slices": pred_stats["nonzero_slices"],
            "pred_start_z": pred_stats["start_z"],
            "pred_end_z": pred_stats["end_z"],
            "pred_max_area": pred_stats["max_area"],
            "pred_mean_nonzero_area": f"{pred_stats['mean_nonzero_area']:.2f}",
            "image_nonzero_voxels": img_nonzero,
            "pred_to_image_nonzero_ratio": f"{fg_ratio:.6f}",
        }
        rows.append(row)
        pred_volumes.append(pred_stats["volume"])
        pred_slices.append(pred_stats["nonzero_slices"])
        pred_max_areas.append(pred_stats["max_area"])
        fg_ratios.append(fg_ratio)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    gt_volumes = []
    gt_slices = []
    gt_max_areas = []
    for mask_path in sorted(
        p
        for p in glob.glob(os.path.join(PROCESSED_TRAIN_DIR, "*", "*_mask.npy"))
        if "_pseudo_mask" not in os.path.basename(p)
    ):
        mask = load_3d(mask_path).astype(np.uint8)
        stats = summarize_mask(mask)
        gt_volumes.append(stats["volume"])
        gt_slices.append(stats["nonzero_slices"])
        gt_max_areas.append(stats["max_area"])

    print("\n=== Validation Prediction Volume Audit ===")
    print(f"Saved per-case CSV: {OUT_CSV}")
    print(f"Pred volume:       {percentile_line(pred_volumes)}")
    print(f"Pred nonzero z:    {percentile_line(pred_slices)}")
    print(f"Pred max area:     {percentile_line(pred_max_areas)}")
    print(f"Pred/img fg ratio: {percentile_line(fg_ratios)}")
    print("\n=== Labeled Training GT Reference ===")
    print(f"GT volume:         {percentile_line(gt_volumes)}")
    print(f"GT nonzero z:      {percentile_line(gt_slices)}")
    print(f"GT max area:       {percentile_line(gt_max_areas)}")

    if pred_slices and gt_slices:
        print("\n=== Quick Read ===")
        print(
            f"Pred median nonzero z / GT median nonzero z: "
            f"{np.median(pred_slices):.1f} / {np.median(gt_slices):.1f}"
        )
        print(
            f"Pred median volume / GT median volume: "
            f"{np.median(pred_volumes):.0f} / {np.median(gt_volumes):.0f}"
        )


if __name__ == "__main__":
    main()
