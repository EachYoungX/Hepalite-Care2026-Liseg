import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = str(PROJECT_ROOT / "artifacts" / "smoke_tests" / "output")
RAW_DIR = str(PROJECT_ROOT / "dataset" / "raw" / "validation_set")
TEST_CASES = [
    ("1034-B2", "Vendor_B2"),
    ("1044-B2", "Vendor_B2")
]

def check_alignment(pred_path, raw_path):
    """
    Verify physical dimensions, affine matrices, and voxel values between prediction and raw reference.
    """
    try:
        pred_img = nib.load(pred_path)
        raw_img = nib.load(raw_path)
    except Exception as e:
        return False, f"Failed to load NIfTI files: {str(e)}"

    pred_shape = pred_img.shape
    raw_shape = raw_img.shape
    pred_affine = pred_img.affine
    raw_affine = raw_img.affine

    if pred_shape != raw_shape:
        return False, f"Dimension mismatch: expected {raw_shape}, got {pred_shape}"

    if not np.allclose(pred_affine, raw_affine, atol=1e-5):
        return False, "Affine matrix mismatch (spatial shift detected)"

    try:
        pred_data = pred_img.get_fdata()
    except Exception as e:
        return False, f"Failed to read image array: {str(e)}"

    if np.isnan(pred_data).any() or np.isinf(pred_data).any():
        return False, "Image array contains invalid values (NaN or Inf)"

    unique_vals = np.unique(pred_data)
    for val in unique_vals:
        if val not in [0.0, 1.0]:
            return False, f"Non-binary voxel value detected: {unique_vals}"

    volume_pixels = int(np.sum(pred_data))
    if volume_pixels == 0:
        return False, "Empty mask detected (volume is 0 voxels)"

    return True, f"Success (Volume: {volume_pixels} voxels)"

def main():
    print("\n=== CARE 2026 Docker Inference Local Verification ===")
    liseg_pred_root = os.path.join(PRED_DIR, "LiSeg_pred")

    if not os.path.exists(liseg_pred_root):
        print(f"Error: LiSeg_pred folder not found in output directory: {PRED_DIR}")
        return

    total_cases = len(TEST_CASES)
    passed_cases = 0

    for case_id, vendor in TEST_CASES:
        print(f"\nAnalyzing Case: {case_id} ({vendor})")

        pred_nii_path = os.path.join(liseg_pred_root, case_id, "GED4_pred.nii.gz")
        raw_nii_path = os.path.join(RAW_DIR, vendor, case_id, "GED4.nii.gz")

        if not os.path.exists(pred_nii_path):
            print(f"   [FAIL] Expected output file not found: {pred_nii_path}")
            continue

        if not os.path.exists(raw_nii_path):
            print(f"   [SKIP] Original raw reference file not found: {raw_nii_path}")
            continue

        success, msg = check_alignment(pred_nii_path, raw_nii_path)
        if success:
            print(f"   [PASS] {msg}")
            passed_cases += 1
        else:
            print(f"   [FAIL] {msg}")

    print("\n=== Verification Summary ===")
    print(f"Passed Cases: {passed_cases} / {total_cases}")
    if passed_cases == total_cases:
        print("Status: All test cases passed successfully. Physical dimensions, affine orientation, and binary metrics are aligned with original raw files.")
    else:
        print("Status: Some test cases failed verification. Please inspect the details above.")

if __name__ == "__main__":
    main()
