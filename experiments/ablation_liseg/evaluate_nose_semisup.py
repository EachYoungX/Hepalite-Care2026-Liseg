import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.ablation_liseg import run_ablation as core
from src.models.models import HepaLiteStudent25D
from src.postprocess.topology import (
    keep_dominant_temporal_chain,
    keep_largest_3d_component,
)


NOSE_EXPERIMENT = ROOT / "artifacts/training_runs/20260620_223719_20260503_HepaLite_Student_SemiSup_V1_NoSE"
NOSE_CACHE = core.CACHE_DIR / "student_transductive_nose"
SE_CACHE = core.CACHE_DIR / "student_transductive"
THRESHOLDS = np.arange(0.30, 0.5001, 0.025).round(3).tolist()


def checkpoint_path(fold):
    return NOSE_EXPERIMENT / f"fold_{fold}" / "checkpoints/best_model.pth"


def ensure_nose_cache(records, assignments, device):
    NOSE_CACHE.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        fold_records = [record for record in records if assignments[record["case"]] == fold]
        missing = [
            record
            for record in fold_records
            if not (NOSE_CACHE / f"{record['case']}_prob.npy").exists()
        ]
        if not missing:
            continue
        model = HepaLiteStudent25D(
            in_channels=5,
            out_classes=1,
            init_features=16,
            deep_supervision=True,
            use_se=False,
        )
        model.load_state_dict(
            torch.load(checkpoint_path(fold), map_location=device, weights_only=True)
        )
        model.to(device).eval()
        for record in missing:
            image = core.load_volume(record["data_path"], ged4=True).astype(np.float32)
            probability = core.predict_probability(model, "student", image, device)
            np.save(NOSE_CACHE / f"{record['case']}_prob.npy", probability)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generic_topology(probability, threshold):
    opened = core.opening_volume(probability, threshold=threshold)
    cca = keep_largest_3d_component(opened)
    holes = binary_fill_holes(cca).astype(np.uint8)
    prediction = keep_dominant_temporal_chain(
        holes,
        min_slice_area=50,
        min_overlap_iou=0.03,
        centroid_jump_limit=40.0,
        rescue_slices=2,
    )
    return core.remove_low_area_slices(prediction, min_size=50)


def evaluate_thresholds(records):
    rows = []
    caches = {"SE": SE_CACHE, "NoSE": NOSE_CACHE}
    for model_name, cache_dir in caches.items():
        for record in records:
            probability = np.load(cache_dir / f"{record['case']}_prob.npy")
            target = core.load_volume(record["mask_path"]).astype(np.uint8)
            for threshold in THRESHOLDS:
                prediction = generic_topology(probability, threshold)
                rows.append(
                    {
                        "case": record["case"],
                        "vendor": record["vendor"],
                        "model": model_name,
                        "threshold": threshold,
                        **core.segmentation_metrics(prediction, target),
                    }
                )
    return rows


def summarize(rows):
    summary = []
    for model_name in ("SE", "NoSE"):
        for threshold in THRESHOLDS:
            group = [
                row
                for row in rows
                if row["model"] == model_name and row["threshold"] == threshold
            ]
            summary.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "dice_mean": float(np.mean([row["dice"] for row in group])),
                    "dice_std": float(np.std([row["dice"] for row in group], ddof=1)),
                    "hd95_mean_mm": float(np.mean([row["hd95_mm"] for row in group])),
                    "hd95_std_mm": float(np.std([row["hd95_mm"] for row in group], ddof=1)),
                    "precision_mean": float(np.mean([row["precision"] for row in group])),
                    "recall_mean": float(np.mean([row["recall"] for row in group])),
                }
            )
    return summary


def best_row(summary, model_name):
    candidates = [row for row in summary if row["model"] == model_name]
    return max(candidates, key=lambda row: (row["dice_mean"], -row["hd95_mean_mm"]))


def paired_comparison(rows, se_threshold, nose_threshold):
    selected = {
        (row["model"], row["case"]): row
        for row in rows
        if (row["model"] == "SE" and row["threshold"] == se_threshold)
        or (row["model"] == "NoSE" and row["threshold"] == nose_threshold)
    }
    cases = sorted({case for model, case in selected if model == "SE"})
    output = []
    for metric in ("dice", "hd95_mm", "precision", "recall", "z_span_error_slices"):
        se_values = np.asarray([selected[("SE", case)][metric] for case in cases])
        nose_values = np.asarray([selected[("NoSE", case)][metric] for case in cases])
        try:
            statistic, p_value = wilcoxon(se_values, nose_values, zero_method="wilcox")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        output.append(
            {
                "metric": metric,
                "se_mean": float(se_values.mean()),
                "nose_mean": float(nose_values.mean()),
                "nose_minus_se": float((nose_values - se_values).mean()),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    return output


def plot_thresholds(summary, se_best, nose_best):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for model_name, color in (("SE", "#2563eb"), ("NoSE", "#059669")):
        group = [row for row in summary if row["model"] == model_name]
        axes[0].plot(
            [row["threshold"] for row in group],
            [row["dice_mean"] for row in group],
            marker="o",
            label=model_name,
            color=color,
        )
        axes[1].plot(
            [row["threshold"] for row in group],
            [row["hd95_mean_mm"] for row in group],
            marker="o",
            label=model_name,
            color=color,
        )
    axes[0].scatter(se_best["threshold"], se_best["dice_mean"], color="#2563eb", s=70)
    axes[0].scatter(nose_best["threshold"], nose_best["dice_mean"], color="#059669", s=70)
    axes[0].set_ylabel("OOF 3D Dice")
    axes[1].set_ylabel("OOF HD95 (mm)")
    for axis in axes:
        axis.set_xlabel("Probability threshold")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("SE vs NoSE semi-supervised threshold sensitivity")
    fig.tight_layout()
    core.save_figure(fig, "figure_08_nose_semisup_thresholds")


def write_table(se_best, nose_best, paired):
    path = core.TABLE_DIR / "table_06_nose_semisup.md"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| Model | Threshold | Dice | HD95 mm | Precision | Recall |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in (se_best, nose_best):
            handle.write(
                f"| {row['model']} | {row['threshold']:.3f} | {row['dice_mean']:.4f} | "
                f"{row['hd95_mean_mm']:.2f} | {row['precision_mean']:.4f} | "
                f"{row['recall_mean']:.4f} |\n"
            )
        handle.write("\n| Paired metric | NoSE - SE | Wilcoxon p |\n")
        handle.write("|---|---:|---:|\n")
        for row in paired:
            handle.write(
                f"| {row['metric']} | {row['nose_minus_se']:.5f} | {row['p_value']:.5g} |\n"
            )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = core.real_case_records()
    assignments = core.fold_assignments(records)
    ensure_nose_cache(records, assignments, device)
    rows = evaluate_thresholds(records)
    summary = summarize(rows)
    se_best = best_row(summary, "SE")
    nose_best = best_row(summary, "NoSE")
    paired = paired_comparison(rows, se_best["threshold"], nose_best["threshold"])
    dice_delta = next(row["nose_minus_se"] for row in paired if row["metric"] == "dice")
    hd_delta = next(row["nose_minus_se"] for row in paired if row["metric"] == "hd95_mm")
    decision = {
        "se_best": se_best,
        "nose_best": nose_best,
        "dice_delta": dice_delta,
        "hd95_delta_mm": hd_delta,
        "submit_recommended": bool(dice_delta >= 0.003 and hd_delta <= 0.0),
        "criteria": "NoSE Dice gain >= 0.003 and HD95 does not worsen",
        "manual_case_overrides": False,
    }
    core.write_csv(core.METRIC_DIR / "nose_semisup_per_case_thresholds.csv", rows)
    core.write_csv(core.METRIC_DIR / "nose_semisup_threshold_summary.csv", summary)
    core.write_csv(core.METRIC_DIR / "nose_semisup_paired.csv", paired)
    with open(core.ARTIFACT_DIR / "nose_semisup_decision.json", "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    plot_thresholds(summary, se_best, nose_best)
    write_table(se_best, nose_best, paired)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
