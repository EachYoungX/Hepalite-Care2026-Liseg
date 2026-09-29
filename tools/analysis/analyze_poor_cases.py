import os
import glob
import warnings

from _project import PROJECT_ROOT  # noqa: F401

# Suppress a compatibility warning emitted by torch.load.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label
from src.models.models import HepaLiteStudent25D

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

CHECKPOINT_PATH = "artifacts/training_runs/20260612_104443_20260503_HepaLite_Student_SemiSup_V1/fold_1/checkpoints/best_model.pth"
POOR_CASES_TXT = (
    "artifacts/training_runs/20260612_104443_20260503_HepaLite_Student_SemiSup_V1/fold_1/poor_cases.txt"
)
DATA_DIR = "dataset/processed/training_set"
OUTPUT_DIAG_DIR = "artifacts/analysis/poor_cases"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIAG_DIR, exist_ok=True)


def find_case_path(case_name, base_dir):
    """Search vendor subdirectories for a case volume."""
    for v in ["Vendor_A", "Vendor_B1", "Vendor_B2"]:
        path = os.path.join(base_dir, v, f"{case_name}_data.npy")
        if os.path.exists(path):
            return path
    return None


def calculate_slice_dice(pred_bin, gt_bin):
    """Compute Dice for a binarized 2D slice."""
    intersect = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    if union == 0:
        return 1.0
    return (2.0 * intersect + 1e-5) / (union + 1e-5)


def analyze_intensity_distribution(img_3d, mask_3d):
    liver_voxels = img_3d[mask_3d > 0]
    bg_voxels = img_3d[(mask_3d == 0) & (img_3d > 0.01)]

    if len(liver_voxels) == 0:
        return "reference mask is empty"

    mean_liver = np.mean(liver_voxels)
    std_liver = np.std(liver_voxels)
    mean_bg = np.mean(bg_voxels) if len(bg_voxels) > 0 else 0.5

    contrast_ratio = mean_liver / max(1e-5, mean_bg)
    overlap_area = np.sum(
        (img_3d > (mean_liver - std_liver))
        & (img_3d < (mean_bg + std_liver))
        & (mask_3d == 0)
    )

    reasons = []
    if contrast_ratio < 1.15:
        reasons.append("very low contrast")
    if std_liver > 0.25:
        reasons.append("intensity inhomogeneity")
    if overlap_area > 20000:
        reasons.append("high overlap with surrounding tissue")
    if not reasons:
        reasons.append("incomplete boundary; consider tuning post-processing")
    return " | ".join(reasons)


def run_full_diagnostics():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        return
    if not os.path.exists(POOR_CASES_TXT):
        print(f"[ERROR] Poor-case list not found: {POOR_CASES_TXT}")
        return

    print("=> Loading the topology-aware student network...")
    model = HepaLiteStudent25D(
        in_channels=5, out_classes=1, init_features=16, deep_supervision=True
    )
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    with open(POOR_CASES_TXT, "r") as f:
        poor_lines = [line.strip() for line in f.readlines() if line.strip()]

    print(
        f"=> Found {len(poor_lines)} poor-performing cases; starting full-volume diagnostics..."
    )

    for line in poor_lines:
        case_name, dice_val = line.split(":")
        case_path = find_case_path(case_name, DATA_DIR)

        if case_path is None:
            print(f"[WARNING] Case not found: {case_name}")
            continue

        print(
            f"\n[Full-volume analysis] Case: {case_name} | Local Dice: {float(dice_val):.4f}"
        )

        img_3d = np.load(case_path)
        if img_3d.ndim == 4:
            img_3d = img_3d[0]

        mask_path = case_path.replace("_data.npy", "_mask.npy")
        if not os.path.exists(mask_path):
            mask_path = case_path.replace("_data.npy", "_pseudo_mask.npy")
        mask_3d = np.load(mask_path)
        if mask_3d.ndim == 4:
            mask_3d = mask_3d[0]

        D, H, W = img_3d.shape
        pred_3d = np.zeros_like(img_3d, dtype=np.float32)

        case_full_dir = os.path.join(OUTPUT_DIAG_DIR, f"{case_name}_full_scan")
        os.makedirs(case_full_dir, exist_ok=True)

        for z in range(D):
            z_prev2 = max(0, z - 2)
            z_prev1 = max(0, z - 1)
            z_next1 = min(D - 1, z + 1)
            z_next2 = min(D - 1, z + 2)

            slice_5d = img_3d[[z_prev2, z_prev1, z, z_next1, z_next2]].copy()

            with torch.no_grad():
                input_tensor = (
                    torch.from_numpy(slice_5d).unsqueeze(0).to(DEVICE).float()
                )
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
                pred_3d[z] = probs

        slice_dices = []
        for z in range(D):
            s_dice = calculate_slice_dice(pred_3d[z] > 0.5, mask_3d[z])
            slice_dices.append(s_dice)

        physio_reasons = analyze_intensity_distribution(img_3d, mask_3d)

        plt.figure(figsize=(10, 4))
        plt.plot(range(D), slice_dices, marker="o", color="darkred", label="Slice Dice")
        plt.axhline(y=0.7, color="gray", linestyle="--", label="Reference threshold (0.7)")
        plt.title(f"Case {case_name} - Dice profile across the z-axis")
        plt.xlabel("Slice index along z-axis")
        plt.ylabel("Dice score")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        curve_path = os.path.join(OUTPUT_DIAG_DIR, f"{case_name}_depth_dice_curve.png")
        plt.savefig(curve_path, dpi=120)
        plt.close()
        print(f"Depth-Dice profile saved to: {curve_path}")

        print(f"Exporting full-volume slice frames ({D} frames)...")
        for z in range(D):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                f"Slice {z+1}/{D} | Dice: {slice_dices[z]:.4f}\n"
                f"Intensity findings: {physio_reasons}",
                fontsize=11,
                fontweight="bold",
                color="navy",
            )

            axes[0].imshow(img_3d[z], cmap="gray")
            axes[0].set_title("Input MRI")
            axes[0].axis("off")

            axes[1].imshow(mask_3d[z], cmap="gray")
            axes[1].set_title("Reference mask")
            axes[1].axis("off")

            axes[2].imshow(pred_3d[z], cmap="jet", vmin=0, vmax=1)
            axes[2].set_title("Student prediction")
            axes[2].axis("off")

            plt.tight_layout()
            frame_path = os.path.join(case_full_dir, f"slice_{z:03d}.png")
            plt.savefig(frame_path, dpi=80)
            plt.close()

        print(f"[DONE] Full-volume frames saved for case {case_name}: {case_full_dir}")

    print("\nFull-volume diagnostic analysis completed. Review the diagnostics directory.")


if __name__ == "__main__":
    run_full_diagnostics()
