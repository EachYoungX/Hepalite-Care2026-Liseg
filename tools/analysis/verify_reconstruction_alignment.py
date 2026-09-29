import glob
import os
import sys

from _project import PROJECT_ROOT  # noqa: F401

import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import KFold

from reconstruct_and_package import reverse_resample

DATA_DIR = "dataset/processed/training_set"
RAW_DIR = "dataset/raw/training_set"
TEMP_PRED_DIR = "artifacts/predictions/temp_predictions"
VENDORS = ("Vendor_A", "Vendor_B1", "Vendor_B2")


def get_real_ids(data_dir, vendors):
    ids = []
    for vendor in vendors:
        mask_files = glob.glob(os.path.join(data_dir, vendor, "*_mask.npy"))
        for f in mask_files:
            fname = os.path.basename(f)
            if "_pseudo_mask" not in fname:
                ids.append(fname.replace("_mask.npy", ""))
    return sorted(list(set(ids)))


def find_processed_mask(case_name):
    matches = glob.glob(os.path.join(DATA_DIR, "*", f"{case_name}_mask.npy"))
    return matches[0] if matches else None


def find_raw_case_dir(case_name):
    matches = glob.glob(os.path.join(RAW_DIR, "Vendor*", f"{case_name}*"))
    return matches[0] if matches else None


def calculate_dice(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    intersect = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0
    return (2.0 * intersect + 1e-5) / (union + 1e-5)


def load_temp_or_processed_mask(case_name):
    temp_mask_path = os.path.join(TEMP_PRED_DIR, f"{case_name}_mask.npy")
    source = "temp_prediction"
    if not os.path.exists(temp_mask_path):
        matched = glob.glob(os.path.join(TEMP_PRED_DIR, f"{case_name}*_mask.npy"))
        temp_mask_path = matched[0] if matched else None

    if temp_mask_path is None:
        temp_mask_path = find_processed_mask(case_name)
        source = "processed_ground_truth"

    if temp_mask_path is None:
        return None, None

    arr = np.load(temp_mask_path)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8), source


def main():
    print("\n=== Spatial reconstruction and affine consistency check ===")
    real_sup_ids = get_real_ids(DATA_DIR, VENDORS)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    _, val_idx = next(iter(kfold.split(real_sup_ids)))
    val_real_names = [real_sup_ids[i] for i in val_idx]
    print(f"Fold 1 local validation cases: {val_real_names}")

    dices = []
    for case_name in val_real_names:
        pred_processed, source = load_temp_or_processed_mask(case_name)
        if pred_processed is None:
            print(f"Warning: Missing processed mask for case {case_name}")
            continue

        raw_case_dir = find_raw_case_dir(case_name)
        if raw_case_dir is None:
            print(f"Warning: Missing raw case dir for case {case_name}")
            continue

        raw_image_path = os.path.join(raw_case_dir, "GED4.nii.gz")
        raw_mask_path = os.path.join(raw_case_dir, "mask_GED4.nii.gz")
        if not os.path.exists(raw_image_path) or not os.path.exists(raw_mask_path):
            print(f"Warning: Missing raw GED4 or mask for case {case_name}")
            continue

        restored_img = reverse_resample(pred_processed, raw_image_path)
        pred_final = sitk.GetArrayFromImage(restored_img)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(raw_mask_path))

        dice = calculate_dice(pred_final, gt)
        dices.append(dice)
        print(
            f"   Case: {case_name:<15} | Source: {source:<22} | "
            f"Restored versus reference 3D Dice: {dice:.5f}"
        )

    print("\n=== Reconstruction audit summary ===")
    if not dices:
        print("Conclusion: no cases were audited; check the configured paths.")
        return

    mean_dice = float(np.mean(dices))
    print(f"Mean restored 3D Dice: {mean_dice:.5f}")
    if mean_dice > 0.90:
        print("Conclusion: reverse_resample is approximately inverse to preprocessing geometry.")
    else:
        print("Conclusion: substantial affine, orientation, crop, or dimension mismatches remain.")


if __name__ == "__main__":
    main()
