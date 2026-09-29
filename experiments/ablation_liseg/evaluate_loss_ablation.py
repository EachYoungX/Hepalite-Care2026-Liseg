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
from experiments.ablation_liseg.variants import ControlledStudent
from src.postprocess.topology import keep_dominant_temporal_chain, keep_largest_3d_component


TRAINING_DIR = core.LOCAL_ARTIFACT_DIR / "training"
CACHE_ROOT = core.CACHE_DIR / "loss_ablation"
THRESHOLDS = np.arange(0.30, 0.5001, 0.025).round(3).tolist()
SPECS = {
    "Tversky+BCE": {
        "training": TRAINING_DIR / "real_5slice",
        "cache": core.CACHE_DIR / "structure/real_5slice",
    },
    "Dice+BCE": {
        "training": TRAINING_DIR / "real_5slice_dice_bce",
        "cache": CACHE_ROOT / "dice_bce",
    },
}


def topology(probability, threshold):
    opened = core.opening_volume(probability, threshold)
    cca = keep_largest_3d_component(opened)
    holes = binary_fill_holes(cca).astype(np.uint8)
    mask = keep_dominant_temporal_chain(
        holes,
        min_slice_area=50,
        min_overlap_iou=0.03,
        centroid_jump_limit=40.0,
        rescue_slices=2,
    )
    return core.remove_low_area_slices(mask, 50)


def ensure_dice_cache(records, assignments, device):
    cache = SPECS["Dice+BCE"]["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        fold_records = [record for record in records if assignments[record["case"]] == fold]
        missing = [record for record in fold_records if not (cache / f"{record['case']}_prob.npy").exists()]
        if not missing:
            continue
        model = ControlledStudent(context_slices=5, use_coordconv=True, use_se=True)
        model.load_state_dict(
            torch.load(
                SPECS["Dice+BCE"]["training"] / f"fold_{fold}/checkpoints/best_model.pth",
                map_location=device,
                weights_only=True,
            )
        )
        model.to(device).eval()
        for record in missing:
            image = core.load_volume(record["data_path"], ged4=True).astype(np.float32)
            probability = core.predict_probability(model, "controlled", image, device)
            np.save(cache / f"{record['case']}_prob.npy", probability)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(records):
    rows = []
    for loss_name, spec in SPECS.items():
        for record in records:
            probability = np.load(spec["cache"] / f"{record['case']}_prob.npy")
            target = core.load_volume(record["mask_path"]).astype(np.uint8)
            for threshold in THRESHOLDS:
                rows.append(
                    {
                        "case": record["case"],
                        "loss": loss_name,
                        "threshold": threshold,
                        **core.segmentation_metrics(topology(probability, threshold), target),
                    }
                )
    return rows


def summarize(rows):
    output = []
    for loss_name in SPECS:
        for threshold in THRESHOLDS:
            group = [row for row in rows if row["loss"] == loss_name and row["threshold"] == threshold]
            output.append(
                {
                    "loss": loss_name,
                    "threshold": threshold,
                    "dice_mean": float(np.mean([row["dice"] for row in group])),
                    "hd95_mean_mm": float(np.mean([row["hd95_mm"] for row in group])),
                    "precision_mean": float(np.mean([row["precision"] for row in group])),
                    "recall_mean": float(np.mean([row["recall"] for row in group])),
                }
            )
    return output


def best(summary, loss_name):
    return max(
        [row for row in summary if row["loss"] == loss_name],
        key=lambda row: (row["dice_mean"], -row["hd95_mean_mm"]),
    )


def compare(rows, tversky_best, dice_best):
    selected = {
        (row["loss"], row["case"]): row
        for row in rows
        if (row["loss"] == "Tversky+BCE" and row["threshold"] == tversky_best["threshold"])
        or (row["loss"] == "Dice+BCE" and row["threshold"] == dice_best["threshold"])
    }
    cases = sorted({case for loss, case in selected if loss == "Tversky+BCE"})
    output = []
    for metric in ("dice", "hd95_mm", "precision", "recall"):
        tversky = np.asarray([selected[("Tversky+BCE", case)][metric] for case in cases])
        dice = np.asarray([selected[("Dice+BCE", case)][metric] for case in cases])
        statistic, p_value = wilcoxon(dice, tversky, zero_method="wilcox")
        output.append(
            {
                "metric": metric,
                "tversky_mean": float(tversky.mean()),
                "dice_bce_mean": float(dice.mean()),
                "tversky_minus_dice_bce": float((tversky - dice).mean()),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    return output


def write_outputs(rows, summary, tversky_best, dice_best, paired):
    core.write_csv(core.METRIC_DIR / "loss_per_case_thresholds.csv", rows)
    core.write_csv(core.METRIC_DIR / "loss_threshold_summary.csv", summary)
    core.write_csv(core.METRIC_DIR / "loss_paired_significance.csv", paired)
    with open(core.TABLE_DIR / "table_07_loss_ablation.md", "w", encoding="utf-8") as handle:
        handle.write("| Loss | Threshold | Dice | HD95 mm | Precision | Recall |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in (dice_best, tversky_best):
            handle.write(
                f"| {row['loss']} | {row['threshold']:.3f} | {row['dice_mean']:.4f} | "
                f"{row['hd95_mean_mm']:.2f} | {row['precision_mean']:.4f} | {row['recall_mean']:.4f} |\n"
            )
        handle.write("\n| Metric | Tversky - Dice+BCE | Wilcoxon p |\n|---|---:|---:|\n")
        for row in paired:
            handle.write(
                f"| {row['metric']} | {row['tversky_minus_dice_bce']:.5f} | {row['p_value']:.5g} |\n"
            )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for loss_name, color in (("Dice+BCE", "#6b7280"), ("Tversky+BCE", "#2563eb")):
        group = [row for row in summary if row["loss"] == loss_name]
        axes[0].plot([row["threshold"] for row in group], [row["dice_mean"] for row in group], marker="o", label=loss_name, color=color)
        axes[1].plot([row["threshold"] for row in group], [row["hd95_mean_mm"] for row in group], marker="o", label=loss_name, color=color)
    axes[0].set_ylabel("OOF 3D Dice")
    axes[1].set_ylabel("OOF HD95 (mm)")
    for axis in axes:
        axis.set_xlabel("Probability threshold")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Asymmetric Tversky loss ablation")
    fig.tight_layout()
    core.save_figure(fig, "figure_09_loss_ablation")
    report = {"tversky_best": tversky_best, "dice_bce_best": dice_best, "paired": paired}
    (core.ARTIFACT_DIR / "loss_ablation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    records = core.real_case_records()
    assignments = core.fold_assignments(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dice_cache(records, assignments, device)
    rows = evaluate(records)
    summary = summarize(rows)
    tversky_best = best(summary, "Tversky+BCE")
    dice_best = best(summary, "Dice+BCE")
    paired = compare(rows, tversky_best, dice_best)
    write_outputs(rows, summary, tversky_best, dice_best, paired)


if __name__ == "__main__":
    main()
