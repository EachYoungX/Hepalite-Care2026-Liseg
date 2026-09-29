import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

BASE_RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
BASE_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
DATASET_LIST = ["training_set", "validation_set"]

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "data_audit")
os.makedirs(REPORT_DIR, exist_ok=True)
LOG_FILE = os.path.join(REPORT_DIR, "preprocessing_errors.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

TARGET_SPACING = (1.5, 1.5, 7.0)
TARGET_SIZE = (32, 256, 256)
MODALITY_TO_USE = "GED4"
MASK_NAME = "mask_GED4"

def safe_read_and_orient(filepath):
    """Read an image, orient it consistently, and sanitize invalid values."""
    try:
        if not os.path.exists(filepath):
            return None

        img = sitk.ReadImage(filepath)
        if img.GetDimension() == 4:
            img = sitk.Extract(
                img, (img.GetWidth(), img.GetHeight(), img.GetDepth(), 0), (0, 0, 0, 0)
            )

        img = sitk.DICOMOrient(img, "LPS")

        np_arr = sitk.GetArrayFromImage(img)
        if np.any(np.isnan(np_arr)) or np.any(np.isinf(np_arr)):
            logging.warning(f"File contains NaN/Inf values: {filepath}")
            np_arr = np.nan_to_num(np_arr, nan=0.0, posinf=0.0, neginf=0.0)
            img = sitk.GetImageFromArray(np_arr)

        return img
    except Exception as e:
        logging.error(f"Failed to read {filepath}: {str(e)}")
        return None


def normalize_robust(img_array, case_id):
    """Normalize intensities using percentiles computed on the non-background region."""
    mask = img_array > 0
    if not np.any(mask):
        logging.warning(f"Case {case_id} is empty or contains only non-positive values.")
        return np.zeros_like(img_array)

    valid_vals = img_array[mask]
    p_low = np.percentile(valid_vals, 0.5)
    p_high = np.percentile(valid_vals, 99.5)

    img_array = np.clip(img_array, p_low, p_high)

    min_val = img_array[mask].min()
    max_val = img_array[mask].max()

    if max_val - min_val > 1e-5:
        img_array[mask] = (img_array[mask] - min_val) / (max_val - min_val)
        img_array[~mask] = 0
    else:
        img_array[:] = 0

    return img_array.astype(np.float32)


def center_crop_or_pad_3d(img, target_size=(32, 256, 256)):
    """Center-crop or zero-pad a volume to the target shape."""
    z, y, x = img.shape
    tz, ty, tx = target_size

    start_z = max(0, (z - tz) // 2)
    start_y = max(0, (y - ty) // 2)
    start_x = max(0, (x - tx) // 2)

    end_z = min(z, start_z + tz)
    end_y = min(y, start_y + ty)
    end_x = min(x, start_x + tx)

    res = np.zeros(target_size, dtype=img.dtype)

    dest_start_z = max(0, (tz - z) // 2)
    dest_start_y = max(0, (ty - y) // 2)
    dest_start_x = max(0, (tx - x) // 2)

    dest_end_z = dest_start_z + (end_z - start_z)
    dest_end_y = dest_start_y + (end_y - start_y)
    dest_end_x = dest_start_x + (end_x - start_x)

    res[dest_start_z:dest_end_z, dest_start_y:dest_end_y, dest_start_x:dest_end_x] = (
        img[start_z:end_z, start_y:end_y, start_x:end_x]
    )

    return res


def process_case(case_dir, save_dir):
    case_name = os.path.basename(case_dir)
    ged4_path = os.path.join(case_dir, f"{MODALITY_TO_USE}.nii.gz")
    mask_path = os.path.join(case_dir, f"{MASK_NAME}.nii.gz")

    img_sitk = safe_read_and_orient(ged4_path)
    if img_sitk is None:
        return False, False

    try:
        orig_spacing = img_sitk.GetSpacing()
        orig_size = img_sitk.GetSize()
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(orig_size, orig_spacing, TARGET_SPACING)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img_sitk.GetDirection())
        resampler.SetOutputOrigin(img_sitk.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        img_res = resampler.Execute(img_sitk)

        img_np = sitk.GetArrayFromImage(img_res)
        img_np = normalize_robust(img_np, case_name)
        img_final = center_crop_or_pad_3d(img_np, TARGET_SIZE)

        has_label = False
        mask_final = None
        if os.path.exists(mask_path):
            mask_sitk = safe_read_and_orient(mask_path)
            if mask_sitk:
                mask_res = sitk.Resample(
                    mask_sitk, img_res, sitk.Transform(), sitk.sitkNearestNeighbor, 0
                )
                mask_np = sitk.GetArrayFromImage(mask_res)
                mask_final = center_crop_or_pad_3d(mask_np, TARGET_SIZE).astype(
                    np.uint8
                )
                if mask_final.max() > 0:
                    has_label = True

        os.makedirs(save_dir, exist_ok=True)
        np.save(
            os.path.join(save_dir, f"{case_name}_data.npy"), img_final[np.newaxis, ...]
        )
        if has_label and mask_final is not None:
            np.save(os.path.join(save_dir, f"{case_name}_mask.npy"), mask_final)

        return True, has_label
    except Exception as e:
        logging.error(f"Case {case_name} processing failed: {str(e)}")
        return False, False


def main():
    vendors = ["Vendor_A", "Vendor_B1", "Vendor_B2"]

    for ds_name in DATASET_LIST:
        print(f"\n>>> Processing dataset: {ds_name}")
        ds_raw_root = os.path.join(BASE_RAW_DIR, ds_name)
        ds_processed_root = os.path.join(BASE_PROCESSED_DIR, ds_name)

        if not os.path.exists(ds_raw_root):
            print(f"Skipping {ds_name}; raw directory does not exist.")
            continue

        ds_summary = []

        for v in vendors:
            v_raw_path = os.path.join(ds_raw_root, v)
            v_save_path = os.path.join(ds_processed_root, v)

            if not os.path.exists(v_raw_path):
                continue

            case_dirs = [
                d for d in glob.glob(os.path.join(v_raw_path, "*")) if os.path.isdir(d)
            ]

            for c_dir in tqdm(case_dirs, desc=f"{ds_name} - {v}"):
                success, labeled = process_case(c_dir, v_save_path)
                if success:
                    ds_summary.append(
                        {
                            "case_id": os.path.basename(c_dir),
                            "vendor": v,
                            "has_label": int(labeled),
                        }
                    )

        os.makedirs(ds_processed_root, exist_ok=True)
        csv_path = os.path.join(ds_processed_root, "dataset_split.csv")
        with open(csv_path, "w") as f:
            f.write("case_id,vendor,has_label\n")
            for item in ds_summary:
                f.write(f"{item['case_id']},{item['vendor']},{item['has_label']}\n")

        print(f"--- Finished {ds_name}; manifest saved to: {csv_path}")


if __name__ == "__main__":
    main()
