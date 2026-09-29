import os
import glob
import shutil
import zipfile
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

PACK_CONFIG = {
    "raw_val_dir": "dataset/raw/validation_set",
    "temp_pred_dir": "artifacts/predictions/temp_predictions",
    "output_submit_dir": "artifacts/submissions/CARE-Liver-Submission",
    "zip_output_path": "artifacts/submissions/CARE-Liver-Submission.zip",
    # Keep a schema-compatible placeholder for evaluators that parse LiFS output.
    "include_lifs_dummy": True,
}

os.makedirs(PACK_CONFIG["output_submit_dir"], exist_ok=True)
liseg_pred_dir = os.path.join(PACK_CONFIG["output_submit_dir"], "LiSeg_pred")
os.makedirs(liseg_pred_dir, exist_ok=True)

TARGET_SPACING = (1.5, 1.5, 7.0)


def _read_3d_image(path):
    img = sitk.ReadImage(path)
    if img.GetDimension() == 4:
        img = sitk.Extract(
            img, (img.GetWidth(), img.GetHeight(), img.GetDepth(), 0), (0, 0, 0, 0)
        )
    return img


def inverse_center_crop_or_pad_3d(img_arr, target_shape):
    res = np.zeros(target_shape, dtype=img_arr.dtype)
    td, th, tw = target_shape
    d, h, w = img_arr.shape

    start_z = max(0, (td - d) // 2)
    start_y = max(0, (th - h) // 2)
    start_x = max(0, (tw - w) // 2)

    end_z = start_z + min(d, td)
    end_y = start_y + min(h, th)
    end_x = start_x + min(w, tw)

    src_start_z = max(0, (d - td) // 2)
    src_start_y = max(0, (h - th) // 2)
    src_start_x = max(0, (w - tw) // 2)

    src_end_z = src_start_z + (end_z - start_z)
    src_end_y = src_start_y + (end_y - start_y)
    src_end_x = src_start_x + (end_x - start_x)

    res[start_z:end_z, start_y:end_y, start_x:end_x] = img_arr[
        src_start_z:src_end_z, src_start_y:src_end_y, src_start_x:src_end_x
    ]
    return res


def reverse_resample(pred_processed, raw_nii_path):
    """
    Invert the exact preprocessing geometry:
    raw GED4 -> DICOMOrient(LPS) -> resample to TARGET_SPACING -> center crop/pad.

    The output is finally resampled back to the original raw GED4 grid, preserving
    the organizer-facing size/origin/spacing/direction.
    """
    raw_img = _read_3d_image(raw_nii_path)
    lps_img = sitk.DICOMOrient(raw_img, "LPS")

    orig_spacing = lps_img.GetSpacing()
    orig_size = lps_img.GetSize()
    resampled_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(orig_size, orig_spacing, TARGET_SPACING)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(resampled_size)
    resampler.SetOutputDirection(lps_img.GetDirection())
    resampler.SetOutputOrigin(lps_img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_ref = resampler.Execute(lps_img)

    resampled_shape_np = sitk.GetArrayFromImage(resampled_ref).shape
    pred_uncropped = inverse_center_crop_or_pad_3d(
        pred_processed.astype(np.uint8), resampled_shape_np
    )

    pred_img = sitk.GetImageFromArray(pred_uncropped)
    pred_img.CopyInformation(resampled_ref)

    back_resampler = sitk.ResampleImageFilter()
    back_resampler.SetReferenceImage(raw_img)
    back_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    back_resampler.SetDefaultPixelValue(0)
    return back_resampler.Execute(pred_img)


def main():
    # Scan vendor folders dynamically to support both underscore and hyphen naming.
    raw_cases = sorted(glob.glob(os.path.join(PACK_CONFIG["raw_val_dir"], "Vendor*", "*")))
    print(
        f"=> Detected {len(raw_cases)} raw cases in validation directory. Starting physical resampling and affine alignment..."
    )

    case_names_list = []

    for raw_case_path in tqdm(raw_cases, desc="NIfTI Reconstruct"):
        case_name = os.path.basename(raw_case_path)

        temp_mask_path = os.path.join(
            PACK_CONFIG["temp_pred_dir"], f"{case_name}_mask.npy"
        )

        if not os.path.exists(temp_mask_path):
            # Resolve the base subject identifier when sequence suffixes differ.
            base_name_clean = case_name.split("-S")[0]
            matched = glob.glob(
                os.path.join(PACK_CONFIG["temp_pred_dir"], f"{base_name_clean}*_mask.npy")
            )
            if matched:
                temp_mask_path = matched[0]
            else:
                print(f"Warning: Missing temporary mask for case: {case_name}")
                continue

        raw_nii_path = os.path.join(raw_case_path, "GED4.nii.gz")
        if not os.path.exists(raw_nii_path):
            print(f"Warning: Missing GED4.nii.gz in {raw_case_path}")
            continue

        pred_processed = np.load(temp_mask_path)
        out_nii_img = reverse_resample(pred_processed, raw_nii_path)

        case_out_dir = os.path.join(liseg_pred_dir, case_name)
        os.makedirs(case_out_dir, exist_ok=True)

        sitk.WriteImage(out_nii_img, os.path.join(case_out_dir, "GED4_pred.nii.gz"))
        case_names_list.append(case_name)

    csv_path = os.path.join(PACK_CONFIG["output_submit_dir"], "LiFS_pred.csv")
    if PACK_CONFIG["include_lifs_dummy"]:
        print(f"=> Generating optional dummy LiFS_pred.csv at {csv_path}...")
        with open(csv_path, "w", newline="") as f:
            f.write("Case,Modality Input,S1,S2,S3,S4\n")
            for name in sorted(case_names_list):
                f.write(f'{name},"GED4",0.25,0.25,0.25,0.25\n')
        print("=> Optional dummy classification schema generated.")

    print(f"=> Archiving outputs to {PACK_CONFIG['zip_output_path']}...")
    if os.path.exists(PACK_CONFIG["zip_output_path"]):
        os.remove(PACK_CONFIG["zip_output_path"])

    with zipfile.ZipFile(
        PACK_CONFIG["zip_output_path"], "w", zipfile.ZIP_DEFLATED
    ) as zipf:
        if PACK_CONFIG["include_lifs_dummy"] and os.path.exists(csv_path):
            zipf.write(csv_path, arcname="LiFS_pred.csv")
        for root, dirs, files in os.walk(liseg_pred_dir):
            for file in files:
                file_full_path = os.path.join(root, file)
                relative_path = os.path.join(
                    "LiSeg_pred", os.path.relpath(file_full_path, liseg_pred_dir)
                )
                zipf.write(file_full_path, arcname=relative_path)

    shutil.rmtree(PACK_CONFIG["output_submit_dir"])
    print(f"=> Packaging process finished. Output stored at {PACK_CONFIG['zip_output_path']}")


if __name__ == "__main__":
    main()
