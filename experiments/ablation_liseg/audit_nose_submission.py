import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
RAW_VAL_DIR = ROOT / "dataset/raw/validation_set"
STAGING_DIR = ROOT / "artifacts/evaluation/nose_validation/CARE-Liver-Submission-NoSE"
ZIP_PATH = ROOT / "artifacts/submissions/CARE-Liver-Submission-NoSE.zip"
REPORT_PATH = ROOT / "artifacts/evaluation/nose_validation/submission_audit.json"


def main():
    failures = []
    records = []
    raw_cases = sorted(path for path in RAW_VAL_DIR.glob("Vendor*/*") if path.is_dir())
    for raw_case in raw_cases:
        case = raw_case.name
        raw = sitk.ReadImage(str(raw_case / "GED4.nii.gz"))
        pred_path = STAGING_DIR / "LiSeg_pred" / case / "GED4_pred.nii.gz"
        if not pred_path.exists():
            failures.append(f"missing prediction: {case}")
            continue
        pred = sitk.ReadImage(str(pred_path))
        array = sitk.GetArrayFromImage(pred)
        checks = {
            "size": pred.GetSize() == raw.GetSize(),
            "spacing": bool(np.allclose(pred.GetSpacing(), raw.GetSpacing())),
            "origin": bool(np.allclose(pred.GetOrigin(), raw.GetOrigin())),
            "direction": bool(np.allclose(pred.GetDirection(), raw.GetDirection())),
            "binary": set(np.unique(array).tolist()).issubset({0, 1}),
            "nonempty": bool(np.any(array)),
        }
        if not all(checks.values()):
            failures.append(f"{case}: {checks}")
        records.append({"case": case, "voxels": int(np.count_nonzero(array)), **checks})

    csv_path = STAGING_DIR / "LiFS_pred.csv"
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        csv_columns = reader.fieldnames
    expected_columns = ["Case", "Modality Input", "S1", "S2", "S3", "S4"]
    if csv_columns != expected_columns or len(csv_rows) != 60:
        failures.append(f"CSV schema/count: {csv_columns}, rows={len(csv_rows)}")

    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        nifti_count = sum(name.endswith("GED4_pred.nii.gz") for name in names)
        if nifti_count != 60 or "LiFS_pred.csv" not in names:
            failures.append(f"ZIP contents: nifti={nifti_count}, csv={'LiFS_pred.csv' in names}")
        bad_roots = [name for name in names if not (name == "LiFS_pred.csv" or name.startswith("LiSeg_pred/"))]
        if bad_roots:
            failures.append(f"Unexpected ZIP roots: {bad_roots[:5]}")

    report = {
        "passed": not failures,
        "failures": failures,
        "raw_cases": len(raw_cases),
        "prediction_records": len(records),
        "zip_nifti_count": nifti_count,
        "zip_path": str(ZIP_PATH),
        "records": records,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
