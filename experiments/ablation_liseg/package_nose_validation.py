import csv
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reconstruct_and_package import reverse_resample


RAW_VAL_DIR = ROOT / "dataset/raw/validation_set"
TEMP_PRED_DIR = ROOT / "artifacts/evaluation/nose_validation/temp_predictions"
STAGING_DIR = ROOT / "artifacts/evaluation/nose_validation/CARE-Liver-Submission-NoSE"
ZIP_PATH = ROOT / "artifacts/submissions/CARE-Liver-Submission-NoSE.zip"


def main():
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    prediction_root = STAGING_DIR / "LiSeg_pred"
    prediction_root.mkdir(parents=True)
    raw_cases = sorted(path for path in RAW_VAL_DIR.glob("Vendor*/*") if path.is_dir())
    if len(raw_cases) != 60:
        raise RuntimeError(f"Expected 60 raw validation cases, found {len(raw_cases)}")
    case_names = []
    for raw_case in tqdm(raw_cases, desc="NoSE physical reconstruction"):
        case = raw_case.name
        mask_path = TEMP_PRED_DIR / f"{case}_mask.npy"
        if not mask_path.exists():
            matches = sorted(TEMP_PRED_DIR.glob(f"{case.split('-S')[0]}*_mask.npy"))
            if len(matches) != 1:
                raise FileNotFoundError(f"No unique processed mask for {case}")
            mask_path = matches[0]
        ged4_path = raw_case / "GED4.nii.gz"
        if not ged4_path.exists():
            raise FileNotFoundError(ged4_path)
        output_case = prediction_root / case
        output_case.mkdir(parents=True)
        output_image = reverse_resample(np.load(mask_path), str(ged4_path))
        sitk.WriteImage(output_image, str(output_case / "GED4_pred.nii.gz"))
        case_names.append(case)

    csv_path = STAGING_DIR / "LiFS_pred.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Case", "Modality Input", "S1", "S2", "S3", "S4"])
        for case in sorted(case_names):
            writer.writerow([case, "GED4", 0.25, 0.25, 0.25, 0.25])

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, "LiFS_pred.csv")
        for path in sorted(prediction_root.glob("*/*.nii.gz")):
            archive.write(path, path.relative_to(STAGING_DIR).as_posix())
    print(f"Packaged {len(case_names)} cases: {ZIP_PATH}")


if __name__ == "__main__":
    main()
