import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    label,
)
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.models import HepaLite25D, HepaLiteStudent25D
from src.postprocess.topology import (
    keep_dominant_temporal_chain,
    keep_largest_3d_component,
)
from experiments.ablation_liseg.variants import ControlledStudent


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULT_DIR = EXPERIMENT_DIR / "results"
LOCAL_ARTIFACT_DIR = ROOT / "artifacts/experiments/ablation_liseg"
ARTIFACT_DIR = RESULT_DIR
CACHE_DIR = LOCAL_ARTIFACT_DIR / "cache"
METRIC_DIR = RESULT_DIR / "metrics"
FIGURE_DIR = ROOT / "publication/camera_ready/CARE_12_source/figures"
TABLE_DIR = RESULT_DIR / "tables"

DATA_DIR = ROOT / "dataset/processed/training_set"
VENDORS = ("Vendor_A", "Vendor_B1", "Vendor_B2")
SPACING_ZYX_MM = (7.0, 1.5, 1.5)
SEED = 42

MODEL_SPECS = {
    "teacher_supervised": {
        "label": "Supervised teacher",
        "experiment": ROOT / "artifacts/training_runs/20260611_212742_Teacher_GED4_Base_Clean_5Fold",
        "checkpoint": "teacher_best.pth",
        "kind": "teacher",
    },
    "student_real_same_arch": {
        "label": "Student architecture (real labels only)",
        "experiment": LOCAL_ARTIFACT_DIR / "training/real_5slice",
        "checkpoint": "best_model.pth",
        "kind": "controlled",
    },
    "student_topology_pseudo": {
        "label": "Student + topology pseudo labels",
        "experiment": ROOT / "artifacts/training_runs/20260612_104443_20260503_HepaLite_Student_SemiSup_V1",
        "checkpoint": "best_model.pth",
        "kind": "student",
    },
    "student_transductive": {
        "label": "Student + validation pseudo labels (0.35)",
        "experiment": ROOT / "artifacts/training_runs/20260617_155036_20260503_HepaLite_Student_SemiSup_V1",
        "checkpoint": "best_model.pth",
        "kind": "student",
    },
}

STAGE_ORDER = [
    "raw_050",
    "raw_035",
    "opening",
    "cca",
    "fill_holes",
    "z_chain",
    "final_topology",
]

STAGE_LABELS = {
    "raw_050": "Raw 0.50",
    "raw_035": "Raw 0.35",
    "opening": "+ Opening",
    "cca": "+ 3D CCA",
    "fill_holes": "+ Fill holes",
    "z_chain": "+ Z-chain",
    "final_topology": "+ Tip rescue",
}


def setup_logging():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(ARTIFACT_DIR / "run.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def real_case_records():
    records = []
    for vendor in VENDORS:
        for mask_path in sorted((DATA_DIR / vendor).glob("*_mask.npy")):
            if "_pseudo_mask" in mask_path.name:
                continue
            case = mask_path.name.replace("_mask.npy", "")
            records.append(
                {
                    "case": case,
                    "vendor": vendor.replace("Vendor_", ""),
                    "mask_path": mask_path,
                    "data_path": mask_path.with_name(f"{case}_data.npy"),
                }
            )
    return sorted(records, key=lambda item: item["case"])


def load_volume(path, ged4=False):
    array = np.load(path, mmap_mode="r")
    if array.ndim == 4:
        if ged4:
            index = 6 if array.shape[0] > 6 else 0
            array = array[index]
        else:
            array = array[0]
    return np.asarray(array)


def fold_assignments(records):
    assignments = {}
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_zero, (_, val_indices) in enumerate(splitter.split(records)):
        for index in val_indices:
            assignments[records[index]["case"]] = fold_zero + 1
    return assignments


def checkpoint_path(spec, fold):
    return spec["experiment"] / f"fold_{fold}" / "checkpoints" / spec["checkpoint"]


def load_model(spec, fold, device):
    if spec["kind"] == "teacher":
        model = HepaLite25D(in_channels=5, out_classes=1, init_features=16)
    elif spec["kind"] == "controlled":
        model = ControlledStudent(context_slices=5, use_coordconv=True, use_se=True)
    else:
        model = HepaLiteStudent25D(
            in_channels=5, out_classes=1, init_features=16, deep_supervision=True
        )
    path = checkpoint_path(spec, fold)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_probability(model, model_kind, image, device):
    depth = image.shape[0]
    probability = np.zeros_like(image, dtype=np.float32)
    for z in range(depth):
        indices = [max(0, z - 2), max(0, z - 1), z, min(depth - 1, z + 1), min(depth - 1, z + 2)]
        tensor = torch.from_numpy(image[indices].copy()).unsqueeze(0).float().to(device)
        output = model(tensor)
        logits = output[0] if model_kind == "teacher" else output
        probability[z] = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probability


def ensure_probability_caches(records, assignments, device, rebuild, logger):
    runtimes = {}
    for model_key, spec in MODEL_SPECS.items():
        model_cache = CACHE_DIR / model_key
        model_cache.mkdir(parents=True, exist_ok=True)
        total_start = time.perf_counter()
        for fold in range(1, 6):
            fold_records = [r for r in records if assignments[r["case"]] == fold]
            missing = [
                r
                for r in fold_records
                if rebuild or not (model_cache / f"{r['case']}_prob.npy").exists()
            ]
            if not missing:
                continue
            logger.info("Caching %s fold %d (%d cases)", model_key, fold, len(missing))
            model = load_model(spec, fold, device)
            for record in missing:
                image = load_volume(record["data_path"], ged4=True).astype(np.float32)
                probability = predict_probability(model, spec["kind"], image, device)
                np.save(model_cache / f"{record['case']}_prob.npy", probability)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        runtimes[model_key] = time.perf_counter() - total_start
    return runtimes


def opening_volume(probability, threshold):
    prediction = np.zeros_like(probability, dtype=np.uint8)
    for z in range(probability.shape[0]):
        binary_slice = probability[z] > threshold
        if np.any(binary_slice):
            prediction[z] = binary_opening(binary_slice, structure=np.ones((3, 3))).astype(np.uint8)
    return prediction


def remove_low_area_slices(prediction, min_size=50):
    output = prediction.copy()
    areas = output.reshape(output.shape[0], -1).sum(axis=1)
    output[areas < min_size] = 0
    return output


def stage_predictions(probability):
    raw_050 = (probability > 0.50).astype(np.uint8)
    raw_035 = (probability > 0.35).astype(np.uint8)
    opened = opening_volume(probability, threshold=0.35)
    cca = keep_largest_3d_component(opened)
    holes = binary_fill_holes(cca).astype(np.uint8)
    z_chain = keep_dominant_temporal_chain(
        holes,
        min_slice_area=50,
        min_overlap_iou=0.03,
        centroid_jump_limit=40.0,
        rescue_slices=0,
    )
    z_chain = remove_low_area_slices(z_chain, min_size=50)
    final = keep_dominant_temporal_chain(
        holes,
        min_slice_area=50,
        min_overlap_iou=0.03,
        centroid_jump_limit=40.0,
        rescue_slices=2,
    )
    final = remove_low_area_slices(final, min_size=50)
    return {
        "raw_050": raw_050,
        "raw_035": raw_035,
        "opening": opened,
        "cca": cca,
        "fill_holes": holes,
        "z_chain": z_chain,
        "final_topology": final,
    }


def surface(mask):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return mask.astype(bool) & ~binary_erosion(mask.astype(bool), border_value=0)


def hd95_mm(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not np.any(prediction) and not np.any(target):
        return 0.0
    if not np.any(prediction) or not np.any(target):
        shape = np.asarray(prediction.shape, dtype=np.float64)
        return float(np.linalg.norm((shape - 1.0) * np.asarray(SPACING_ZYX_MM)))
    pred_surface = surface(prediction)
    target_surface = surface(target)
    distance_to_target = distance_transform_edt(~target_surface, sampling=SPACING_ZYX_MM)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=SPACING_ZYX_MM)
    distances = np.concatenate([distance_to_target[pred_surface], distance_to_pred[target_surface]])
    return float(np.percentile(distances, 95))


def segmentation_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int(np.count_nonzero(prediction & target))
    pred_count = int(np.count_nonzero(prediction))
    target_count = int(np.count_nonzero(target))
    dice = (2.0 * intersection) / max(pred_count + target_count, 1)
    precision = intersection / max(pred_count, 1)
    recall = intersection / max(target_count, 1)
    pred_z = np.where(np.any(prediction, axis=(1, 2)))[0]
    target_z = np.where(np.any(target, axis=(1, 2)))[0]
    if len(pred_z) and len(target_z):
        z_span_error = abs(int(pred_z[0]) - int(target_z[0])) + abs(int(pred_z[-1]) - int(target_z[-1]))
    else:
        z_span_error = prediction.shape[0]
    _, component_count = label(prediction)
    return {
        "dice": float(dice),
        "hd95_mm": hd95_mm(prediction, target),
        "precision": float(precision),
        "recall": float(recall),
        "volume_ratio": float(pred_count / max(target_count, 1)),
        "z_span_error_slices": int(z_span_error),
        "component_count": int(component_count),
    }


def evaluate(records, assignments, logger):
    rows = []
    current_key = "student_transductive"
    for index, record in enumerate(records, 1):
        target = load_volume(record["mask_path"]).astype(np.uint8)
        for model_key in MODEL_SPECS:
            probability = np.load(CACHE_DIR / model_key / f"{record['case']}_prob.npy")
            predictions = stage_predictions(probability)
            stages = STAGE_ORDER if model_key == current_key else ["raw_035", "final_topology"]
            for stage in stages:
                metrics = segmentation_metrics(predictions[stage], target)
                rows.append(
                    {
                        "case": record["case"],
                        "vendor": record["vendor"],
                        "fold": assignments[record["case"]],
                        "model": model_key,
                        "model_label": MODEL_SPECS[model_key]["label"],
                        "stage": stage,
                        "stage_label": STAGE_LABELS[stage],
                        **metrics,
                    }
                )
        logger.info("Metrics %02d/%02d: %s", index, len(records), record["case"])
    return rows


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, iterations=5000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return np.percentile(samples, [2.5, 97.5]).tolist()


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = (row["model"], row["stage"])
        groups.setdefault(key, []).append(row)
    aggregate = []
    for (model, stage), group in groups.items():
        dice = np.asarray([row["dice"] for row in group])
        hd95 = np.asarray([row["hd95_mm"] for row in group])
        dice_ci = bootstrap_ci(dice)
        hd_ci = bootstrap_ci(hd95)
        aggregate.append(
            {
                "model": model,
                "model_label": MODEL_SPECS[model]["label"],
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
                "n": len(group),
                "dice_mean": float(dice.mean()),
                "dice_std": float(dice.std(ddof=1)),
                "dice_ci_low": dice_ci[0],
                "dice_ci_high": dice_ci[1],
                "hd95_mean_mm": float(hd95.mean()),
                "hd95_std_mm": float(hd95.std(ddof=1)),
                "hd95_ci_low_mm": hd_ci[0],
                "hd95_ci_high_mm": hd_ci[1],
                "precision_mean": float(np.mean([row["precision"] for row in group])),
                "recall_mean": float(np.mean([row["recall"] for row in group])),
                "z_span_error_mean": float(np.mean([row["z_span_error_slices"] for row in group])),
            }
        )
    return sorted(aggregate, key=lambda row: (row["model"], STAGE_ORDER.index(row["stage"])))


def significance_rows(rows):
    lookup = {}
    for row in rows:
        lookup[(row["model"], row["stage"], row["case"])] = row
    comparisons = [
        ("student_transductive", "raw_035", "student_transductive", "final_topology", "Topology effect"),
        ("teacher_supervised", "final_topology", "student_real_same_arch", "final_topology", "Teacher vs student architecture"),
        ("student_real_same_arch", "final_topology", "student_topology_pseudo", "final_topology", "Topology pseudo-label effect"),
        ("student_topology_pseudo", "final_topology", "student_transductive", "final_topology", "Validation pseudo-label effect"),
    ]
    output = []
    cases = sorted({row["case"] for row in rows})
    for model_a, stage_a, model_b, stage_b, label_text in comparisons:
        a_dice = np.asarray([lookup[(model_a, stage_a, case)]["dice"] for case in cases])
        b_dice = np.asarray([lookup[(model_b, stage_b, case)]["dice"] for case in cases])
        a_hd = np.asarray([lookup[(model_a, stage_a, case)]["hd95_mm"] for case in cases])
        b_hd = np.asarray([lookup[(model_b, stage_b, case)]["hd95_mm"] for case in cases])
        for metric, a_values, b_values in [("dice", a_dice, b_dice), ("hd95_mm", a_hd, b_hd)]:
            try:
                statistic, p_value = wilcoxon(a_values, b_values, zero_method="wilcox")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            output.append(
                {
                    "comparison": label_text,
                    "metric": metric,
                    "model_a": model_a,
                    "stage_a": stage_a,
                    "model_b": model_b,
                    "stage_b": stage_b,
                    "mean_a": float(a_values.mean()),
                    "mean_b": float(b_values.mean()),
                    "mean_delta_b_minus_a": float((b_values - a_values).mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return output


def subset(rows, model=None, stage=None):
    return [
        row
        for row in rows
        if (model is None or row["model"] == model) and (stage is None or row["stage"] == stage)
    ]


def save_figure(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stage_ablation(rows):
    stage_rows = subset(rows, model="student_transductive")
    plot_stages = ["raw_035", "opening", "cca", "fill_holes", "z_chain"]
    labels = ["Raw", "Opening", "CCA", "Fill holes", "Z-chain"]
    colors = ["#9ca3af", "#cbd5e1", "#1f4e79", "#0f766e", "#166534"]
    dice = [[row["dice"] for row in stage_rows if row["stage"] == stage] for stage in plot_stages]
    hd = [[row["hd95_mm"] for row in stage_rows if row["stage"] == stage] for stage in plot_stages]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    positions = np.arange(len(plot_stages))
    for axis, values, title in zip(axes, [dice, hd], ["Dice", "HD95 (mm)"]):
        box = axis.boxplot(
            values,
            positions=positions,
            widths=0.56,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#111827", "linewidth": 1.2},
            whiskerprops={"color": "#475569", "linewidth": 1.0},
            capprops={"color": "#475569", "linewidth": 1.0},
        )
        for patch, color, stage in zip(box["boxes"], colors, plot_stages):
            patch.set_facecolor(color)
            patch.set_edgecolor("#1f2937")
            patch.set_alpha(0.92 if stage == "cca" else 0.78)
            patch.set_linewidth(1.2 if stage == "cca" else 0.9)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=11)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.grid(axis="y", alpha=0.16, color="#94a3b8")
        axis.tick_params(axis="y", labelsize=11)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)
    axes[0].set_ylim(0.65, 1.0)
    fig.tight_layout()
    save_figure(fig, "figure_01_topology_ablation")


def plot_model_comparison(rows):
    model_order = [
        "teacher_supervised",
        "student_real_same_arch",
        "student_topology_pseudo",
        "student_transductive",
    ]
    final_rows = {model: subset(rows, model=model, stage="final_topology") for model in model_order}
    dice_means = [np.mean([row["dice"] for row in final_rows[m]]) for m in model_order]
    dice_stds = [np.std([row["dice"] for row in final_rows[m]], ddof=1) for m in model_order]
    hd_means = [np.mean([row["hd95_mm"] for row in final_rows[m]]) for m in model_order]
    hd_stds = [np.std([row["hd95_mm"] for row in final_rows[m]], ddof=1) for m in model_order]
    labels = [
        "Teacher\n(real only)",
        "Student\n(real only)",
        "Student\n(topology pseudo)",
        "Student\n(+ val pseudo 0.35)",
    ]
    colors = ["#6b7280", "#d97706", "#2563eb", "#059669"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(model_order))
    axes[0].bar(x, dice_means, yerr=dice_stds, capsize=4, color=colors)
    axes[1].bar(x, hd_means, yerr=hd_stds, capsize=4, color=colors)
    for axis, ylabel in zip(axes, ["3D Dice", "HD95 (mm)"]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.75, 1.0)
    fig.suptitle("OOF model comparison with identical topology post-processing")
    fig.tight_layout()
    save_figure(fig, "figure_02_model_comparison")


def plot_vendor_distribution(rows):
    final_rows = subset(rows, model="student_transductive", stage="final_topology")
    vendors = ["A", "B1", "B2"]
    dice = [[row["dice"] for row in final_rows if row["vendor"] == vendor] for vendor in vendors]
    hd = [[row["hd95_mm"] for row in final_rows if row["vendor"] == vendor] for vendor in vendors]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].boxplot(dice, labels=vendors, showfliers=True)
    axes[1].boxplot(hd, labels=vendors, showfliers=True)
    axes[0].set_ylabel("3D Dice")
    axes[1].set_ylabel("HD95 (mm)")
    for axis in axes:
        axis.set_xlabel("Vendor")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Vendor-stratified performance (current student, OOF)")
    fig.tight_layout()
    save_figure(fig, "figure_03_vendor_distribution")


def plot_profiles_and_qualitative(rows, records):
    raw_lookup = {row["case"]: row for row in subset(rows, model="student_transductive", stage="raw_035")}
    final_lookup = {row["case"]: row for row in subset(rows, model="student_transductive", stage="final_topology")}
    gains = sorted(
        ((case, final_lookup[case]["dice"] - raw_lookup[case]["dice"]) for case in raw_lookup),
        key=lambda item: item[1],
        reverse=True,
    )
    worst = min(final_lookup, key=lambda case: final_lookup[case]["dice"])
    selected = []
    for case in [gains[0][0], worst, gains[len(gains) // 2][0]]:
        if case not in selected:
            selected.append(case)
    record_lookup = {record["case"]: record for record in records}

    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 2.8 * len(selected)), squeeze=False)
    for axis, case in zip(axes[:, 0], selected):
        target = load_volume(record_lookup[case]["mask_path"]).astype(bool)
        probability = np.load(CACHE_DIR / "student_transductive" / f"{case}_prob.npy")
        predictions = stage_predictions(probability)
        axis.plot(target.reshape(target.shape[0], -1).sum(axis=1), label="Ground truth", linewidth=2)
        axis.plot(predictions["raw_035"].reshape(target.shape[0], -1).sum(axis=1), label="Raw 0.35")
        axis.plot(predictions["final_topology"].reshape(target.shape[0], -1).sum(axis=1), label="Final topology")
        axis.set_title(f"{case}: axial area profile")
        axis.set_xlabel("Processed axial slice")
        axis.set_ylabel("Area (pixels)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=3, frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure_04_z_area_profiles")

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)), squeeze=False)
    for row_index, case in enumerate(selected):
        record = record_lookup[case]
        image = load_volume(record["data_path"], ged4=True)
        target = load_volume(record["mask_path"]).astype(bool)
        probability = np.load(CACHE_DIR / "student_transductive" / f"{case}_prob.npy")
        predictions = stage_predictions(probability)
        z = int(np.argmax(target.reshape(target.shape[0], -1).sum(axis=1)))
        panels = [target[z], predictions["raw_035"][z], predictions["final_topology"][z]]
        axes[row_index, 0].imshow(image[z], cmap="gray")
        axes[row_index, 0].set_title(f"{case} | GED4")
        for col, (mask, title, color) in enumerate(
            zip(panels, ["Ground truth", "Raw 0.35", "Final topology"], ["#22c55e", "#ef4444", "#06b6d4"]),
            start=1,
        ):
            axes[row_index, col].imshow(image[z], cmap="gray")
            if np.any(mask):
                axes[row_index, col].contour(mask, levels=[0.5], colors=[color], linewidths=1.2)
            axes[row_index, col].set_title(title)
        for axis in axes[row_index]:
            axis.axis("off")
    fig.tight_layout()
    save_figure(fig, "figure_05_qualitative_oof")


def write_markdown_tables(aggregate, significance):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stage_rows = [
        row for row in aggregate if row["model"] == "student_transductive" and row["stage"] in STAGE_ORDER
    ]
    with open(TABLE_DIR / "table_01_topology_ablation.md", "w", encoding="utf-8") as handle:
        handle.write("| Stage | Dice mean +/- SD | HD95 mm mean +/- SD | Precision | Recall |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in stage_rows:
            handle.write(
                f"| {row['stage_label']} | {row['dice_mean']:.4f} +/- {row['dice_std']:.4f} | "
                f"{row['hd95_mean_mm']:.2f} +/- {row['hd95_std_mm']:.2f} | "
                f"{row['precision_mean']:.4f} | {row['recall_mean']:.4f} |\n"
            )
    model_order = [
        "teacher_supervised",
        "student_real_same_arch",
        "student_topology_pseudo",
        "student_transductive",
    ]
    model_rows = sorted(
        [row for row in aggregate if row["stage"] == "final_topology"],
        key=lambda row: model_order.index(row["model"]),
    )
    with open(TABLE_DIR / "table_02_model_ablation.md", "w", encoding="utf-8") as handle:
        handle.write("| Model | Dice mean (95% CI) | HD95 mm mean (95% CI) |\n")
        handle.write("|---|---:|---:|\n")
        for row in model_rows:
            handle.write(
                f"| {row['model_label']} | {row['dice_mean']:.4f} "
                f"({row['dice_ci_low']:.4f}-{row['dice_ci_high']:.4f}) | "
                f"{row['hd95_mean_mm']:.2f} ({row['hd95_ci_low_mm']:.2f}-{row['hd95_ci_high_mm']:.2f}) |\n"
            )
    with open(TABLE_DIR / "table_03_significance.md", "w", encoding="utf-8") as handle:
        handle.write("| Comparison | Metric | Delta (B-A) | Wilcoxon p |\n")
        handle.write("|---|---|---:|---:|\n")
        for row in significance:
            handle.write(
                f"| {row['comparison']} | {row['metric']} | "
                f"{row['mean_delta_b_minus_a']:.5f} | {row['p_value']:.5g} |\n"
            )


def write_manifest(records, assignments, cache_runtimes, device):
    checkpoints = {}
    for model_key, spec in MODEL_SPECS.items():
        checkpoints[model_key] = []
        for fold in range(1, 6):
            path = checkpoint_path(spec, fold)
            checkpoints[model_key].append(
                {"fold": fold, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            )
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": SEED,
        "fold_split": "sklearn KFold(n_splits=5, shuffle=True, random_state=42)",
        "case_count": len(records),
        "case_fold_assignments": assignments,
        "spacing_zyx_mm": SPACING_ZYX_MM,
        "manual_case_overrides": False,
        "oof_ensemble": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cache_runtime_seconds": cache_runtimes,
        "checkpoints": checkpoints,
    }
    with open(ARTIFACT_DIR / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    for directory in [CACHE_DIR, METRIC_DIR, FIGURE_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = real_case_records()
    if len(records) != 30:
        raise RuntimeError(f"Expected 30 labeled cases, found {len(records)}")
    assignments = fold_assignments(records)
    logger.info("Starting OOF ablation on %d cases using %s", len(records), device)
    cache_runtimes = ensure_probability_caches(records, assignments, device, args.rebuild_cache, logger)
    rows = evaluate(records, assignments, logger)
    aggregate = aggregate_rows(rows)
    significance = significance_rows(rows)
    write_csv(METRIC_DIR / "per_case_metrics.csv", rows)
    write_csv(METRIC_DIR / "aggregate_metrics.csv", aggregate)
    write_csv(METRIC_DIR / "paired_significance.csv", significance)
    plot_stage_ablation(rows)
    plot_model_comparison(rows)
    plot_vendor_distribution(rows)
    plot_profiles_and_qualitative(rows, records)
    write_markdown_tables(aggregate, significance)
    write_manifest(records, assignments, cache_runtimes, device)
    logger.info("Ablation complete. Artifacts: %s", ARTIFACT_DIR)
    for row in aggregate:
        if row["stage"] == "final_topology":
            logger.info(
                "%s | Dice %.4f | HD95 %.2f mm",
                row["model_label"],
                row["dice_mean"],
                row["hd95_mean_mm"],
            )


if __name__ == "__main__":
    main()
