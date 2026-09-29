import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.ablation_liseg import run_ablation as core
from experiments.ablation_liseg.train_zspan_presence import SlicePresenceNet25D


PRESENCE_DIR = core.LOCAL_ARTIFACT_DIR / "zspan_presence"
OOF_PRESENCE_DIR = PRESENCE_DIR / "oof_probabilities"
SEGMENTATION_CACHE = core.CACHE_DIR / "student_transductive_nose"
VALIDATION_DATA = ROOT / "dataset/processed/validation_set"
VALIDATION_MASKS = ROOT / "artifacts/evaluation/nose_validation/temp_predictions"
VALIDATION_OUTPUT = ROOT / "artifacts/evaluation/nose_validation/auto_zspan_predictions"
VALIDATION_PRESENCE_CACHE = PRESENCE_DIR / "validation_probabilities"

PRESENCE_THRESHOLDS = [0.35, 0.45, 0.55, 0.65]
RESCUE_THRESHOLDS = [0.15, 0.25, 0.35]
SIGMAS = [0.0, 0.75, 1.25]
MARGINS = [0, 1, 2]


def choose_interval(probability, anchor, threshold, rescue_threshold, sigma, margin):
    smoothed = (
        gaussian_filter1d(probability, sigma=sigma, mode="nearest")
        if sigma > 0
        else probability.copy()
    )
    active = smoothed >= threshold
    runs = []
    start = None
    for z, value in enumerate(active.tolist() + [False]):
        if value and start is None:
            start = z
        elif not value and start is not None:
            runs.append((start, z - 1))
            start = None
    if not runs:
        return 0, len(probability) - 1
    containing = [run for run in runs if run[0] <= anchor <= run[1]]
    if containing:
        start, end = containing[0]
    else:
        start, end = min(runs, key=lambda run: min(abs(anchor - run[0]), abs(anchor - run[1])))
    while start > 0 and smoothed[start - 1] >= rescue_threshold:
        start -= 1
    while end + 1 < len(smoothed) and smoothed[end + 1] >= rescue_threshold:
        end += 1
    return max(0, start - margin), min(len(smoothed) - 1, end + margin)


def gate_mask(mask, probability, config):
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    if not np.any(areas):
        return mask.copy(), (0, mask.shape[0] - 1)
    anchor = int(np.argmax(areas))
    start, end = choose_interval(probability, anchor=anchor, **config)
    output = mask.copy()
    output[:start] = 0
    output[end + 1 :] = 0
    return output, (start, end)


def search_oof(records):
    baseline_rows = []
    case_data = {}
    for record in records:
        probability = np.load(SEGMENTATION_CACHE / f"{record['case']}_prob.npy")
        mask = core.stage_predictions(probability)["final_topology"]
        target = core.load_volume(record["mask_path"]).astype(np.uint8)
        presence = np.load(OOF_PRESENCE_DIR / f"{record['case']}_prob.npy")
        case_data[record["case"]] = (mask, target, presence)
        baseline_rows.append(core.segmentation_metrics(mask, target))
    baseline_dice = float(np.mean([row["dice"] for row in baseline_rows]))
    baseline_hd = float(np.mean([row["hd95_mm"] for row in baseline_rows]))
    def fast_metrics(prediction, target):
        prediction = prediction.astype(bool)
        target = target.astype(bool)
        intersection = np.count_nonzero(prediction & target)
        denominator = np.count_nonzero(prediction) + np.count_nonzero(target)
        dice = 2.0 * intersection / max(denominator, 1)
        pred_z = np.flatnonzero(np.any(prediction, axis=(1, 2)))
        target_z = np.flatnonzero(np.any(target, axis=(1, 2)))
        z_error = (
            abs(int(pred_z[0]) - int(target_z[0]))
            + abs(int(pred_z[-1]) - int(target_z[-1]))
            if len(pred_z) and len(target_z)
            else prediction.shape[0]
        )
        return float(dice), int(z_error)

    rows = []
    for threshold, rescue, sigma, margin in itertools.product(
        PRESENCE_THRESHOLDS, RESCUE_THRESHOLDS, SIGMAS, MARGINS
    ):
        config = {
            "threshold": threshold,
            "rescue_threshold": rescue,
            "sigma": sigma,
            "margin": margin,
        }
        dice_values = []
        z_errors = []
        for mask, target, presence in case_data.values():
            gated, _ = gate_mask(mask, presence, config)
            dice, z_error = fast_metrics(gated, target)
            dice_values.append(dice)
            z_errors.append(z_error)
        rows.append(
            {
                **config,
                "dice_mean": float(np.mean(dice_values)),
                "hd95_mean_mm": None,
                "z_span_error_mean": float(np.mean(z_errors)),
                "dice_delta": float(np.mean(dice_values) - baseline_dice),
                "hd95_delta_mm": None,
            }
        )
    safe = [row for row in rows if row["dice_delta"] >= -0.0005]
    candidate_pool = sorted(
        safe,
        key=lambda row: (row["z_span_error_mean"], -row["dice_mean"]),
    )[:20]
    if not candidate_pool:
        candidate_pool = sorted(rows, key=lambda row: -row["dice_mean"])[:10]
    for row in candidate_pool:
        config = {
            key: row[key]
            for key in ("threshold", "rescue_threshold", "sigma", "margin")
        }
        hd_values = []
        for mask, target, presence in case_data.values():
            gated, _ = gate_mask(mask, presence, config)
            hd_values.append(core.hd95_mm(gated, target))
        row["hd95_mean_mm"] = float(np.mean(hd_values))
        row["hd95_delta_mm"] = row["hd95_mean_mm"] - baseline_hd
    best = min(
        candidate_pool,
        key=lambda row: (row["hd95_mean_mm"], -row["dice_mean"]),
    )
    return rows, best, {"dice_mean": baseline_dice, "hd95_mean_mm": baseline_hd}


@torch.no_grad()
def validation_presence_probabilities(device):
    VALIDATION_PRESENCE_CACHE.mkdir(parents=True, exist_ok=True)
    models = []
    for fold in range(1, 6):
        model = SlicePresenceNet25D().to(device)
        model.load_state_dict(
            torch.load(
                PRESENCE_DIR / f"fold_{fold}/best_model.pth",
                map_location=device,
                weights_only=True,
            )
        )
        model.eval()
        models.append(model)
    output = {}
    for path in sorted(VALIDATION_DATA.glob("*/*_data.npy")):
        case = path.name.replace("_data.npy", "")
        cache_path = VALIDATION_PRESENCE_CACHE / f"{case}_prob.npy"
        if cache_path.exists():
            output[case] = np.load(cache_path)
            continue
        image = core.load_volume(path, ged4=True).astype(np.float32)
        probabilities = np.zeros(image.shape[0], dtype=np.float32)
        for z in range(image.shape[0]):
            indices = [
                max(0, z - 2),
                max(0, z - 1),
                z,
                min(image.shape[0] - 1, z + 1),
                min(image.shape[0] - 1, z + 2),
            ]
            tensor = torch.from_numpy(image[indices].copy()).unsqueeze(0).float().to(device)
            probabilities[z] = float(
                np.mean([torch.sigmoid(model(tensor))[0].cpu().item() for model in models])
            )
        output[case] = probabilities
        np.save(cache_path, probabilities)
    return output


def apply_validation(probabilities, config):
    VALIDATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    gate_config = {
        key: config[key]
        for key in ("threshold", "rescue_threshold", "sigma", "margin")
    }
    for case, presence in probabilities.items():
        mask = np.load(VALIDATION_MASKS / f"{case}_mask.npy").astype(np.uint8)
        gated, interval = gate_mask(mask, presence, gate_config)
        np.save(VALIDATION_OUTPUT / f"{case}_mask.npy", gated)
        original_z = np.flatnonzero(mask.reshape(mask.shape[0], -1).sum(axis=1))
        gated_z = np.flatnonzero(gated.reshape(gated.shape[0], -1).sum(axis=1))
        rows.append(
            {
                "case": case,
                "presence_start": interval[0],
                "presence_end": interval[1],
                "original_start": int(original_z[0]) if len(original_z) else -1,
                "original_end": int(original_z[-1]) if len(original_z) else -1,
                "gated_start": int(gated_z[0]) if len(gated_z) else -1,
                "gated_end": int(gated_z[-1]) if len(gated_z) else -1,
                "removed_voxels": int(mask.sum() - gated.sum()),
            }
        )
    return rows


def main():
    records = core.real_case_records()
    grid, best, baseline = search_oof(records)
    core.write_csv(core.METRIC_DIR / "zspan_presence_grid.csv", grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = validation_presence_probabilities(device)
    validation_rows = apply_validation(probabilities, best)
    core.write_csv(core.METRIC_DIR / "zspan_presence_validation.csv", validation_rows)
    report = {
        "baseline_oof": baseline,
        "best_config": best,
        "known_risk_cases": [
            row
            for row in validation_rows
            if row["case"] in {"1096-B2", "1110-B2", "1146-B1", "1186-B1", "2016-B2", "2022-B2"}
        ],
        "validation_cases_changed": sum(row["removed_voxels"] > 0 for row in validation_rows),
    }
    (PRESENCE_DIR / "gating_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
