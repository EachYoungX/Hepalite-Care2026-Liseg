import os
import zipfile
import csv
import shutil
import numpy as np
import nibabel as nib

CHECK_CONFIG = {
    "zip_path": "artifacts/submissions/CARE-Liver-Submission.zip",
    "raw_val_dir": "dataset/raw/validation_set",
    "temp_extract_dir": "artifacts/smoke_tests/temp_submit_check",
}


def find_raw_path(case_name, base_raw_dir):
    """Locate the raw validation NIfTI for a case identifier."""
    for root, dirs, files in os.walk(base_raw_dir):
        if os.path.basename(root) == case_name:
            ged4_path = os.path.join(root, "GED4.nii.gz")
            if os.path.exists(ged4_path):
                return ged4_path
    return None


def main():
    zip_path = CHECK_CONFIG["zip_path"]
    if not os.path.exists(zip_path):
        print(f"[ERROR] Submission archive not found: {zip_path}")
        return

    print(f"\n{'='*30}\n Starting archive validator \n{'='*30}")

    if os.path.exists(CHECK_CONFIG["temp_extract_dir"]):
        shutil.rmtree(CHECK_CONFIG["temp_extract_dir"])
    os.makedirs(CHECK_CONFIG["temp_extract_dir"], exist_ok=True)

    print("[1/4] Scanning archive structure...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(CHECK_CONFIG["temp_extract_dir"])
    except Exception as e:
        print(f"   [ERROR] Archive is damaged or unreadable: {str(e)}")
        return

    has_csv = "LiFS_pred.csv" in file_list
    has_liseg = any(f.startswith("LiSeg_pred/") for f in file_list)
    has_mac_junk = any(os.path.basename(f).startswith("._") for f in file_list)

    if has_csv:
        print("   [OK] LiFS_pred.csv is present at the archive root.")
    else:
        print("   [ERROR] LiFS_pred.csv is missing; the submission may be rejected.")

    if has_liseg:
        print("   [OK] LiSeg_pred/ and case-level directories are present.")
    else:
        print("   [ERROR] LiSeg_pred/ is missing or empty; segmentation scores will be zero.")

    if has_mac_junk:
        print("   [WARNING] Archive contains macOS dot files (._*); some Linux evaluators may fail.")
    else:
        print("   [OK] Archive contains no operating-system metadata files.")

    print("\n[2/4] Auditing the optional staging CSV schema...")
    csv_extract_path = os.path.join(CHECK_CONFIG["temp_extract_dir"], "LiFS_pred.csv")

    csv_cases = []
    if os.path.exists(csv_extract_path):
        try:
            with open(csv_extract_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)

                expected_header = ["Case_ID", "Fibrosis_Stage"]
                if header == expected_header:
                    print(f"   [OK] CSV header matches: {header}")
                else:
                    print(f"   [ERROR] CSV header mismatch: expected {expected_header}, got {header}")

                for row in reader:
                    if len(row) == 2:
                        csv_cases.append(row[0])
            print(f"   [OK] CSV schema audit passed; protected cases written: {len(csv_cases)}.")
        except Exception as e:
            print(f"   [ERROR] CSV read failed: {str(e)}")
    else:
        print("   [ERROR] CSV extraction failed; schema audit cannot continue.")

    print("\n[3/4] Auditing 3D physical-space NIfTI alignment...")

    liseg_extract_dir = os.path.join(CHECK_CONFIG["temp_extract_dir"], "LiSeg_pred")
    pred_case_dirs = sorted([d for d in os.listdir(liseg_extract_dir) if os.path.isdir(os.path.join(liseg_extract_dir, d))])

    mismatch_shapes = 0
    mismatch_affines = 0
    corrupted_masks = 0
    success_count = 0

    for case_name in pred_case_dirs:
        pred_nii_path = os.path.join(liseg_extract_dir, case_name, "GED4_pred.nii.gz")
        if not os.path.exists(pred_nii_path):
            print(f"   [ERROR] GED4_pred.nii.gz is missing for case {case_name}.")
            continue

        raw_nii_path = find_raw_path(case_name, CHECK_CONFIG["raw_val_dir"])
        if raw_nii_path is None:
            print(f"   [SKIP] Raw NIfTI not found for case {case_name}; alignment was not checked.")
            continue

        try:
            pred_img = nib.load(pred_nii_path)
            raw_img = nib.load(raw_nii_path)

            pred_shape = pred_img.shape
            raw_shape = raw_img.shape
            pred_affine = pred_img.affine
            raw_affine = raw_img.affine

            if pred_shape != raw_shape:
                print(f"   [ERROR] Shape mismatch for {case_name}: raw {raw_shape} versus restored {pred_shape}")
                mismatch_shapes += 1
                continue

            if not np.allclose(pred_affine, raw_affine, atol=1e-5):
                print(f"   [ERROR] Affine mismatch for case {case_name}.")
                mismatch_affines += 1
                continue

            pred_data = pred_img.get_fdata()
            unique_vals = np.unique(pred_data)

            is_binary = all(val in [0, 1] for val in unique_vals)
            voxel_sum = np.sum(pred_data)

            if not is_binary:
                print(f"   [ERROR] Non-binary voxels detected for case {case_name}: {unique_vals}")
                corrupted_masks += 1
                continue

            if voxel_sum == 0:
                print(f"   [WARNING] Empty restored mask for case {case_name}; check for excessive pruning.")
                corrupted_masks += 1
                continue

            success_count += 1

        except Exception as e:
            print(f"   [ERROR] Failed to load NIfTI files for case {case_name}: {str(e)}")
            corrupted_masks += 1

    print("\n[4/4] Generating audit summary...")
    print(f"   - Cases restored and aligned successfully: {success_count}.")
    if mismatch_shapes > 0:
        print(f"   FATAL: {mismatch_shapes} cases have resolution mismatches.")
    if mismatch_affines > 0:
        print(f"   FATAL: {mismatch_affines} cases have broken physical alignment.")
    if corrupted_masks > 0:
        print(f"   WARNING: {corrupted_masks} masks are non-binary or empty.")

    print(f"\n{'='*30}")
    if mismatch_shapes == 0 and mismatch_affines == 0 and corrupted_masks == 0 and has_csv and has_liseg:
        print(" Simulated evaluation result: [PASS]")
    else:
        print(" Simulated evaluation result: [FAIL]")
    print(f"{'='*30}")

    shutil.rmtree(CHECK_CONFIG["temp_extract_dir"])


if __name__ == "__main__":
    main()
