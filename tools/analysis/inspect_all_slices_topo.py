import os
import sys
import glob
import warnings

from _project import PROJECT_ROOT  # noqa: F401

# Suppress a compatibility warning emitted by torch.load.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# `_project` places the repository root on sys.path.
from src.models.models import HepaLite25D

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

exp_candidates = sorted(
    glob.glob("artifacts/training_runs/*_Teacher_GED4_Base_Clean_5Fold")
)
TEACHER_EXP_PATH = (
    exp_candidates[-1]
    if exp_candidates
    else "artifacts/training_runs/YOUR_CLEAN_TEACHER_FOLDER"
)
CASE_ID = "0196-B1-S1"
OUTPUT_DIR = str(PROJECT_ROOT / "artifacts" / "analysis" / "slice_inspections" / f"{CASE_ID}_topo_slices")


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fold_paths = glob.glob(
        os.path.join(TEACHER_EXP_PATH, "fold_*", "checkpoints", "best_model.pth")
    )
    if not fold_paths:
        fold_paths = glob.glob(
            os.path.join(TEACHER_EXP_PATH, "fold_*", "checkpoints", "teacher_best.pth")
        )

    print(
        f"=> Loading {len(fold_paths)} teacher models from {os.path.basename(TEACHER_EXP_PATH)}..."
    )
    teachers = []
    for p in fold_paths:
        m = HepaLite25D(in_channels=5, out_classes=1, init_features=16).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.eval()
        teachers.append(m)

    data_path = glob.glob(
        os.path.join("dataset/processed", "**", f"{CASE_ID}_data.npy"), recursive=True
    )[0]
    pseudo_path = data_path.replace("_data.npy", "_pseudo_mask.npy")
    weight_path = data_path.replace("_data.npy", "_weight.npy")

    data = np.load(data_path)
    mask = np.load(pseudo_path)
    weight = np.load(weight_path)

    if data.ndim == 4:
        data = data[0]
    if mask.ndim == 4:
        mask = mask[0]
    if weight.ndim == 4:
        weight = weight[0]

    D, H, W = data.shape
    print(f"=> Exporting all {D} slices for case {CASE_ID} to: {OUTPUT_DIR} ...")

    print("=> Extracting the global volume profile...")
    raw_teacher_areas = []
    final_topo_areas = []
    cached_mean_probs = []

    for z in range(D):
        z_prev2 = max(0, z - 2)
        z_prev1 = max(0, z - 1)
        z_next1 = min(D - 1, z + 1)
        z_next2 = min(D - 1, z + 2)

        slice_5d = data[[z_prev2, z_prev1, z, z_next1, z_next2]].copy()
        input_t = torch.from_numpy(slice_5d).unsqueeze(0).to(device).float()

        probs = []
        for m in teachers:
            logits, _ = m(input_t)
            probs.append(torch.sigmoid(logits)[0, 0].cpu().numpy())
        mean_prob = np.mean(probs, axis=0)
        cached_mean_probs.append(mean_prob)

        raw_teacher_areas.append(np.sum(mean_prob > 0.35))
        final_topo_areas.append(np.sum(mask[z]))

    plt.figure(figsize=(11, 5))
    plt.plot(range(D), raw_teacher_areas, label="1. Raw teacher area (prob > 0.35)", color="royalblue", alpha=0.6, linestyle="--", marker="o")
    plt.plot(range(D), final_topo_areas, label="2. Topology-refined area", color="crimson", alpha=0.8, linewidth=2.5, marker="s")
    plt.title(f"Case {CASE_ID} - z-axis projected area comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Slice index along z-axis")
    plt.ylabel("Projected foreground area (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, f"{CASE_ID}_volume_profile_comparison.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Longitudinal area comparison saved to: {curve_path}")

    for z in tqdm(range(D), desc="Exporting Slices"):
        mean_prob = cached_mean_probs[z]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(data[z], cmap="gray")
        prob_overlay = plt.cm.jet(mean_prob)
        prob_overlay[..., 3] = mean_prob * 0.45
        axes[0].imshow(prob_overlay)
        axes[0].set_title(f"1. Raw GED4 (z={z})\nTeacher probability overlay")
        axes[0].axis("off")

        axes[1].imshow(data[z], cmap="gray")
        overlay = np.zeros((*mask[z].shape, 4))
        if np.sum(mask[z]) > 0:
            overlay[mask[z] > 0] = [1, 0, 0, 0.45]
            axes[1].imshow(overlay)
            axes[1].set_title(
                f"2. Topology-refined pseudo-label (area: {np.sum(mask[z])} pixels)"
            )
        else:
            axes[1].set_title("2. Topology-refined pseudo-label")
        axes[1].axis("off")

        im = axes[2].imshow(weight[z], cmap="jet", vmin=0, vmax=1)
        axes[2].set_title(f"3. Confidence weight (slice mean: {np.mean(weight[z]):.2f})")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"slice_{z:02d}_inspect.png"), dpi=100)
        plt.close()

    print(f"\n=> Slice-level topology inspection figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
