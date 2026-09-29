import os
import sys
import glob
import warnings

# Suppress a compatibility warning emitted by torch.load.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

import numpy as np
import torch
from tqdm import tqdm

from src.models.models import HepaLite25D
from src.postprocess.topology import postprocess_prob_volume

exp_candidates = sorted(
    glob.glob("artifacts/training_runs/*_Teacher_GED4_Base_Clean_5Fold")
)
DEFAULT_TEACHER_PATH = (
    exp_candidates[-1]
    if exp_candidates
    else "artifacts/training_runs/YOUR_CLEAN_TEACHER_FOLDER"
)

PARAMS = {
    "teacher_exp_path": DEFAULT_TEACHER_PATH,
    "data_dir": "dataset/processed",
    "vendors": ["Vendor_A", "Vendor_B1", "Vendor_B2"],
    "ged4_idx": 6,
    "target_sets": ["validation_set"],
    "target_case": None,

    "prob_threshold": 0.35,       # Probability threshold for binary refinement.
    "min_size_filter": 50,        # Minimum in-plane component area.
    "max_hole_area": 200,         # Maximum area for local hole filling.
    "centroid_jump_limit": 40.0,  # Maximum centroid displacement in pixels.
    "min_overlap_iou": 0.03,      # Minimum overlap between adjacent slices.
    "sigma_area": 2.0,
    "case_z_start_overrides": {
        "1096-B2": 24,
        "1110-B2": 20,
        "1146-B1": 21,
        "1186-B1": 19,
        "2016-B2": 20,
        "2022-B2": 11,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def load_teacher_ensemble(exp_path, device):
    models = []
    fold_paths = glob.glob(os.path.join(exp_path, "fold_*", "checkpoints", "best_model.pth"))
    if not fold_paths:
        fold_paths = glob.glob(os.path.join(exp_path, "fold_*", "checkpoints", "teacher_best.pth"))

    print(f"=> Loading {len(fold_paths)} teacher folds from {os.path.basename(exp_path)}...")
    for p in fold_paths:
        m = HepaLite25D(in_channels=5, out_classes=1, init_features=16).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    return models


def generate_topo_pseudos():
    device = DEVICE

    try:
        teachers = load_teacher_ensemble(PARAMS["teacher_exp_path"], device)
        print("Teacher ensemble loaded.")
    except Exception as e:
        print(f"Failed to load teacher models: {e}")
        return

    unlabeled_cases = []
    if PARAMS["target_case"] is not None:
        print(f"=> Targeted single-case refinement: {PARAMS['target_case']}")
        for target_set in PARAMS["target_sets"]:
            for v in PARAMS["vendors"]:
                data_path = os.path.join(PARAMS["data_dir"], target_set, v, f"{PARAMS['target_case']}_data.npy")
                if os.path.exists(data_path):
                    unlabeled_cases.append(data_path)
                    break
            if unlabeled_cases:
                break
    else:
        for target_set in PARAMS["target_sets"]:
            for v in PARAMS["vendors"]:
                set_path = os.path.join(PARAMS["data_dir"], target_set, v)
                if not os.path.exists(set_path):
                    continue
                all_data = sorted(glob.glob(os.path.join(set_path, "*_data.npy")))
                for d in all_data:
                    # Generate pseudo-labels only for cases without manual masks.
                    if not os.path.exists(d.replace("_data.npy", "_mask.npy")):
                        unlabeled_cases.append(d)

    if len(unlabeled_cases) == 0:
        print("=> No unlabeled cases require refinement.")
        return

    for data_path in tqdm(unlabeled_cases, desc="Generating 3D topological pseudo-labels"):
        real_mask_path = data_path.replace("_data.npy", "_mask.npy")
        pseudo_path = data_path.replace("_data.npy", "_pseudo_mask.npy")
        weight_path = data_path.replace("_data.npy", "_weight.npy")

        # Apply the real-label guard in both bulk and targeted modes. Targeted
        # refinement must never create or overwrite pseudo artifacts for a
        # manually labeled case.
        if os.path.exists(real_mask_path):
            print(
                f"[SKIP] {os.path.basename(data_path)} has a manual mask; pseudo-label generation is disabled."
            )
            continue

        # Targeted refinement may replace artifacts only for unlabeled cases.
        if PARAMS["target_case"] is not None:
            for p in [pseudo_path, weight_path]:
                if os.path.exists(p):
                    os.remove(p)

        try:
            img_3d = np.load(data_path, mmap_mode="r")
            if img_3d.ndim == 4:
                img_3d = (
                    img_3d[PARAMS["ged4_idx"]]
                    if img_3d.shape[0] > PARAMS["ged4_idx"]
                    else img_3d[0]
                )

            D, H, W = img_3d.shape
            raw_prob_3d = np.zeros((D, H, W), dtype=np.float32)
            variance_3d = np.zeros((D, H, W), dtype=np.float32)

            for z in range(D):
                z_prev2 = max(0, z - 2)
                z_prev1 = max(0, z - 1)
                z_next1 = min(D - 1, z + 1)
                z_next2 = min(D - 1, z + 2)
                slice_5d = img_3d[[z_prev2, z_prev1, z, z_next1, z_next2]].copy()
                slice_tensor = torch.from_numpy(slice_5d).unsqueeze(0).to(device).float()

                fold_probs = []
                for m in teachers:
                    logits, _ = m(slice_tensor)
                    prob = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]
                    fold_probs.append(prob)

                raw_prob_3d[z] = np.mean(fold_probs, axis=0)
                variance_3d[z] = np.var(fold_probs, axis=0)

            fused_mask_3d = postprocess_prob_volume(
                raw_prob_3d,
                threshold=PARAMS["prob_threshold"],
                opening=True,
                fill_holes=False,
                fill_small_holes_only=True,
                max_hole_area=PARAMS["max_hole_area"],
                min_size_filter=PARAMS["min_size_filter"],
                min_overlap_iou=PARAMS["min_overlap_iou"],
                centroid_jump_limit=PARAMS["centroid_jump_limit"],
                sigma_area=PARAMS["sigma_area"],
            )
            case_name = os.path.basename(data_path).replace("_data.npy", "")
            if case_name in PARAMS["case_z_start_overrides"]:
                fused_mask_3d[: PARAMS["case_z_start_overrides"][case_name]] = 0

            final_weight_map = np.zeros((D, H, W), dtype=np.float32)
            for z in range(D):
                if np.sum(fused_mask_3d[z]) == 0:
                    continue

                # Combine teacher disagreement with ensemble variance as confidence.
                disagreement = np.abs(raw_prob_3d[z] - fused_mask_3d[z])
                uncertainty_weight = np.exp(-variance_3d[z] * 10.0)
                raw_weight = (1.0 - disagreement) * uncertainty_weight
                final_weight_map[z] = 0.15 + 0.85 * raw_weight

            np.save(pseudo_path, fused_mask_3d)
            np.save(weight_path, final_weight_map)
            print(f"=> 3D topological refinement completed for {os.path.basename(data_path)}")

        except Exception as e:
            print(f"=> Refinement failed for {os.path.basename(data_path)}: {str(e)}")
            for p in [pseudo_path, weight_path]:
                if os.path.exists(p):
                    os.remove(p)
            continue


if __name__ == "__main__":
    generate_topo_pseudos()
