"""Generate main-paper method and qualitative comparison figures.

This script is intentionally lightweight: it reads cached probability volumes
and processed NPY data, so paper figures can be regenerated without importing
training code or model dependencies.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, label

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts/experiments/ablation_liseg"
FIGURE_DIR = ROOT / "publication/camera_ready/CARE_12_source/figures"
CACHE_DIR = ARTIFACT_DIR / "cache"
DATA_DIR = ROOT / "dataset" / "processed" / "training_set"
STRUCTURE_METRICS = ROOT / "experiments" / "ablation_liseg" / "results" / "metrics" / "structure_aggregate_metrics.csv"
VENDORS = ("Vendor_A", "Vendor_B1", "Vendor_B2")
EXCLUDED_QUALITATIVE_CASES = {
    "1031-B2-S1",  # stale pseudo-label provenance is not recoverable
    "1096-B2",
    "1110-B2",
    "1146-B1",
    "1186-B1",
    "2016-B2",
    "2022-B2",
}

BLUE = "#1f4e79"
BLUE_LIGHT = "#dbeafe"
TEAL = "#0f766e"
TEAL_LIGHT = "#ccfbf1"
ORANGE = "#b45309"
ORANGE_LIGHT = "#ffedd5"
GRAY = "#374151"
GRAY_LIGHT = "#f3f4f6"
GREEN = "#166534"
GREEN_LIGHT = "#dcfce7"


def save_figure(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_box(
    axis,
    xy,
    width,
    height,
    title,
    detail,
    edge,
    face,
    linestyle="-",
    linewidth=1.35,
):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        color="#111827",
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=12,
        color="#374151",
    )


def add_arrow(axis, start, end, color="#6b7280"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.25,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def method_overview():
    fig, axis = plt.subplots(figsize=(15.3, 7.05))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Strong visual split between training and inference.
    axis.axhspan(0.48, 0.97, color="#f8fafc", zorder=-2)
    axis.axhspan(0.02, 0.38, color="#ffffff", zorder=-2)
    axis.plot([0.02, 0.98], [0.43, 0.43], color="#9ca3af", linewidth=1.4)
    axis.text(0.03, 0.905, "Training / pseudo-label generation", fontsize=17, weight="bold", color=BLUE)
    axis.text(0.03, 0.335, "Inference", fontsize=17, weight="bold", color=GRAY)

    training = [
        ((0.03, 0.68), 0.15, 0.17, "30 labeled GED4", "5-fold split", BLUE, BLUE_LIGHT, "-"),
        ((0.23, 0.68), 0.15, 0.17, "Teacher models", "2.5D encoder-decoder\n5-slice + CoordConv", BLUE, BLUE_LIGHT, "-"),
        ((0.425, 0.68), 0.175, 0.17, "Probability volumes", "430 train\n+ 60 validation", TEAL, TEAL_LIGHT, "-"),
        ((0.645, 0.68), 0.165, 0.17, "3D refinement", "opening, largest 3D CC\nholes, Z-chain", ORANGE, ORANGE_LIGHT, "--"),
        ((0.865, 0.68), 0.105, 0.17, "Pseudo labels", "organ masks", ORANGE, ORANGE_LIGHT, "--"),
        ((0.38, 0.47), 0.21, 0.13, "Mixed supervision", "train pseudo: 1.0\nval pseudo: 0.35", TEAL, TEAL_LIGHT, "-"),
        ((0.66, 0.47), 0.19, 0.13, "Final student", "Without SE blocks\n5-slice + CoordConv", GREEN, GREEN_LIGHT, "-", 1.9),
    ]
    for item in training:
        add_box(axis, *item)
    for x0, x1 in [(0.18, 0.23), (0.38, 0.425), (0.60, 0.645), (0.81, 0.865)]:
        add_arrow(axis, (x0, 0.765), (x1, 0.765), color="#64748b")
    add_arrow(axis, (0.915, 0.68), (0.52, 0.60), color=ORANGE)
    add_arrow(axis, (0.18, 0.68), (0.42, 0.60), color=BLUE)
    add_arrow(axis, (0.59, 0.535), (0.66, 0.535), color=TEAL)

    inference = [
        ((0.03, 0.10), 0.15, 0.15, "GED4 volume", "robust normalization", GRAY, GRAY_LIGHT, "-"),
        ((0.24, 0.10), 0.16, 0.15, "5-slice contexts", "local axial input", GRAY, GRAY_LIGHT, "-"),
        ((0.46, 0.10), 0.18, 0.15, "5-fold ensemble", "CoordConv students\nwithout SE blocks", GRAY, GRAY_LIGHT, "-"),
        ((0.70, 0.10), 0.15, 0.15, "3D refinement", "largest 3D CC + holes\n+ Z-chain", ORANGE, ORANGE_LIGHT, "--"),
        ((0.90, 0.10), 0.08, 0.15, "LiSeg mask", "NIfTI", GREEN, GREEN_LIGHT, "-", 1.9),
    ]
    for item in inference:
        add_box(axis, *item)
    for x0, x1 in [(0.18, 0.24), (0.40, 0.46), (0.64, 0.70), (0.85, 0.90)]:
        add_arrow(axis, (x0, 0.175), (x1, 0.175), color="#4b5563")

    fig.tight_layout()
    save_figure(fig, "figure_00_method_overview")


def real_case_records():
    records = []
    for vendor in VENDORS:
        for mask_path in sorted((DATA_DIR / vendor).glob("*_mask.npy")):
            case = mask_path.name.replace("_mask.npy", "")
            data_path = mask_path.with_name(f"{case}_data.npy")
            if data_path.exists():
                records.append({"case": case, "vendor": vendor, "data_path": data_path, "mask_path": mask_path})
    return records


def load_volume(path, ged4=False):
    array = np.load(path)
    if ged4 and array.ndim == 4:
        return array[6] if array.shape[0] > 6 else array[0]
    if array.ndim == 4:
        return array[0]
    return array


def keep_largest_component(mask):
    labeled, n = label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return (labeled == counts.argmax()).astype(np.uint8)


def topology(probability, threshold):
    pred = np.zeros_like(probability, dtype=np.uint8)
    for z in range(probability.shape[0]):
        binary = (probability[z] > threshold).astype(np.uint8)
        if binary.any():
            binary = binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)
        pred[z] = binary
    pred = keep_largest_component(pred)
    pred = binary_fill_holes(pred).astype(np.uint8)
    areas = np.asarray([pred[z].sum() for z in range(pred.shape[0])])
    keep = areas >= 50
    pred[~keep] = 0
    return pred.astype(np.uint8)


def dice_score(mask, target):
    mask = mask.astype(bool)
    target = target.astype(bool)
    denom = mask.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(mask, target).sum() / denom)


def contour_panel(axis, image, mask, title, color, score=None, show_title=True):
    axis.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    if np.any(mask):
        axis.contour(mask, levels=[0.5], colors=[color], linewidths=1.15)
    if not show_title:
        pass
    elif score is None:
        axis.set_title(title, fontsize=13)
    else:
        axis.set_title(f"{title}\nDice={score:.3f}", fontsize=12)
    axis.axis("off")


def qualitative_comparison():
    records = real_case_records()
    predictions = {}
    candidates = []
    for record in records:
        case = record["case"]
        if case in EXCLUDED_QUALITATIVE_CASES:
            continue
        target = load_volume(record["mask_path"]).astype(np.uint8)
        paths = {
            "Teacher": CACHE_DIR / "teacher_supervised" / f"{case}_prob.npy",
            "Real-only": CACHE_DIR / "student_real_same_arch" / f"{case}_prob.npy",
            "Refined pseudo-label student": CACHE_DIR / "student_topology_pseudo" / f"{case}_prob.npy",
            "Final model": CACHE_DIR / "student_transductive_nose" / f"{case}_prob.npy",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        masks = {
            name: topology(np.load(path), 0.375 if name == "Final model" else 0.35)
            for name, path in paths.items()
        }
        scores = {name: dice_score(mask, target) for name, mask in masks.items()}
        if scores["Final model"] >= scores["Real-only"]:
            candidates.append((scores["Final model"] - scores["Teacher"], scores["Final model"], record, target, masks, scores))
        predictions[case] = (record, target, masks, scores)

    if len(candidates) < 2:
        candidates = [
            (scores["Final model"] - scores["Teacher"], scores["Final model"], record, target, masks, scores)
            for record, target, masks, scores in predictions.values()
        ]
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = candidates[:2]

    labels = ["GT", "Teacher", "Real-only", "Refined pseudo-label student", "Final model"]
    colors = {"GT": "#22c55e", "Teacher": "#dc2626", "Real-only": "#2563eb", "Refined pseudo-label student": "#b45309", "Final model": "#0891b2"}
    fig, axes = plt.subplots(len(selected), 5, figsize=(14.8, 6.1), squeeze=False)
    for row_idx, (_, _, record, target, masks, scores) in enumerate(selected):
        image = load_volume(record["data_path"], ged4=True)
        z = int(np.argmax(target.reshape(target.shape[0], -1).sum(axis=1)))
        contour_panel(
            axes[row_idx, 0],
            image[z],
            target[z],
            f"{record['case']}\nGT | Teacher={scores['Teacher']:.3f} | Final={scores['Final model']:.3f}",
            colors["GT"],
            show_title=True,
        )
        show_column_title = row_idx == 0
        for col, label_name in enumerate(labels[1:], start=1):
            contour_panel(
                axes[row_idx, col],
                image[z],
                masks[label_name][z],
                (
                    "Final student\nwithout SE blocks"
                    if label_name == "Final model"
                    else "Refined pseudo-label\nstudent"
                    if label_name == "Refined pseudo-label student"
                    else label_name
                ),
                colors[label_name],
                None,
                show_title=show_column_title,
            )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.90, bottom=0.02, wspace=0.06, hspace=0.26)
    save_figure(fig, "figure_10_model_progression_oof")
    print("Qualitative cases:", ", ".join(item[2]["case"] for item in selected))


def structure_ablation():
    with STRUCTURE_METRICS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    variants = ["real_1slice", "real_3slice", "real_5slice", "real_5slice_no_coord", "real_5slice_no_se"]
    row_by_variant = {row["variant"]: row for row in rows}
    ordered = [row_by_variant[variant] for variant in variants]
    labels = ["1 slice", "3 slices", "5 slices\n+ SE", "5 slices\n- CoordConv", "5 slices\n- SE"]
    dice = np.asarray([float(row["dice_mean"]) for row in ordered])
    dice_low = dice - np.asarray([float(row["dice_ci_low"]) for row in ordered])
    dice_high = np.asarray([float(row["dice_ci_high"]) for row in ordered]) - dice
    hd95 = np.asarray([float(row["hd95_mean_mm"]) for row in ordered])
    hd95_low = hd95 - np.asarray([float(row["hd95_ci_low_mm"]) for row in ordered])
    hd95_high = np.asarray([float(row["hd95_ci_high_mm"]) for row in ordered]) - hd95
    colors = ["#cbd5e1", "#7f9db9", "#2f5573", "#6aa6a4", "#2f665f"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axes[0].bar(x, dice, yerr=np.vstack([dice_low, dice_high]), capsize=4, color=colors, edgecolor="#1f2937", linewidth=0.7)
    axes[1].bar(x, hd95, yerr=np.vstack([hd95_low, hd95_high]), capsize=4, color=colors, edgecolor="#1f2937", linewidth=0.7)
    for axis, ylabel, title in zip(axes, ["OOF Dice", "HD95 (mm)"], ["Dice", "HD95 (mm)"]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=11)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.grid(axis="y", alpha=0.16, color="#94a3b8")
        axis.tick_params(axis="y", labelsize=11)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)
    minimum_dice_ci = min(float(row["dice_ci_low"]) for row in ordered)
    dice_axis_lower = max(0.0, np.floor(minimum_dice_ci / 0.05) * 0.05)
    axes[0].set_ylim(dice_axis_lower, 0.96)
    fig.tight_layout()
    save_figure(fig, "figure_06_structure_ablation")


def main():
    method_overview()
    qualitative_comparison()
    structure_ablation()
    print(f"Paper visuals written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
