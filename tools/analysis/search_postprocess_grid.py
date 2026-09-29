import os
import warnings

from _project import PROJECT_ROOT  # noqa: F401

# Suppress a compatibility warning emitted by torch.load.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

import torch
import numpy as np
from tqdm import tqdm
import glob
from torch.utils.data import DataLoader
from scipy.ndimage import label, binary_fill_holes, binary_opening
from skimage.morphology import convex_hull_image

from src.models.models import HepaLiteStudent25D
from src.data.dataset import HybridDataset25D

CHECKPOINT_PATH = "artifacts/training_runs/20260612_104443_20260503_HepaLite_Student_SemiSup_V1/fold_1/checkpoints/best_model.pth"
DATA_DIR = "dataset/processed/training_set"
VENDORS = ("Vendor_A", "Vendor_B1", "Vendor_B2")

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MIN_SIZES = [0, 50, 100, 150]
ENABLE_OPENING = [True, False]
ENABLE_CONVEX_HULL = [True, False]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleConfig:
    def __init__(self):
        self.vendors = VENDORS


def get_real_ids(data_dir, vendors):
    ids = []
    for vendor in vendors:
        mask_files = glob.glob(os.path.join(data_dir, vendor, "*_mask.npy"))
        for f in mask_files:
            fname = os.path.basename(f)
            if "_pseudo_mask" not in fname:
                ids.append(fname.replace("_mask.npy", ""))
    return sorted(list(set(ids)))


def evaluate_postprocess_slice(prob, thresh, min_size, open_flag, hull_flag):
    """Apply a configurable 2D post-processing pipeline to one slice."""
    mask = prob > thresh
    if np.sum(mask) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    if open_flag:
        mask = binary_opening(mask, structure=np.ones((3, 3)))

    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = np.bincount(labeled_mask.ravel())
    areas[0] = 0
    max_idx = areas.argmax()

    filtered_mask = np.zeros_like(mask, dtype=bool)
    if areas[max_idx] >= min_size:
        filtered_mask[labeled_mask == max_idx] = True

    for label_idx in range(1, len(areas)):
        if (
            label_idx != max_idx
            and areas[label_idx] >= (areas[max_idx] * 0.15)
            and areas[label_idx] >= min_size
        ):
            filtered_mask[labeled_mask == label_idx] = True

    if np.sum(filtered_mask) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    if hull_flag:
        try:
            filtered_mask = convex_hull_image(filtered_mask)
        except:
            pass

    filtered_mask = binary_fill_holes(filtered_mask)
    return filtered_mask.astype(np.uint8)


def calculate_dice(pred, gt):
    intersect = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return (2.0 * intersect + 1e-5) / (union + 1e-5)


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(
            f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}. Update CHECKPOINT_PATH."
        )
        return

    cfg = SimpleConfig()
    real_sup_ids = get_real_ids(DATA_DIR, VENDORS)

    val_split = int(len(real_sup_ids) * 0.2)
    val_real_names = real_sup_ids[:val_split]

    print(f"=> Loading Fold 1 local validation cases: {val_real_names}")
    val_dataset = HybridDataset25D(
        cfg,
        train_dir=DATA_DIR,
        val_dir=None,
        transform=None,
        case_ids=val_real_names,
        is_train=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("=> Loading the student network...")
    model = HepaLiteStudent25D(
        in_channels=5, out_classes=1, init_features=16, deep_supervision=True
    )
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    print("=> Running one GPU inference pass and caching probability maps...")
    cached_probs = []
    cached_gts = []

    with torch.no_grad():
        for images, masks, weights, labels, is_pseudo_batch in tqdm(
            val_loader, desc="Caching probabilities"
        ):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            gt = masks[0, 0].numpy()

            cached_probs.append(probs)
            cached_gts.append(gt)

    num_slices = len(cached_probs)
    print(f"[DONE] Cached {num_slices} slices. Starting offline grid search...")

    results = []
    total_combinations = (
        len(THRESHOLDS) * len(MIN_SIZES) * len(ENABLE_OPENING) * len(ENABLE_CONVEX_HULL)
    )

    pbar_grid = tqdm(total=total_combinations, desc="Grid-search configurations")

    for thresh in THRESHOLDS:
        for min_sz in MIN_SIZES:
            for open_f in ENABLE_OPENING:
                for hull_f in ENABLE_CONVEX_HULL:

                    slice_dices = []
                    for i in range(num_slices):
                        pred_processed = evaluate_postprocess_slice(
                            cached_probs[i], thresh, min_sz, open_f, hull_f
                        )
                        dice = calculate_dice(pred_processed, cached_gts[i])
                        slice_dices.append(dice)

                    mean_dice = np.mean(slice_dices)
                    results.append(
                        {
                            "dice": mean_dice,
                            "thresh": thresh,
                            "min_size": min_sz,
                            "opening": open_f,
                            "convex_hull": hull_f,
                        }
                    )
                    pbar_grid.update(1)

    pbar_grid.close()

    results = sorted(results, key=lambda x: x["dice"], reverse=True)

    print(f"\n{'='*50}\n Offline grid-search results (top 5 configurations) \n{'='*50}")
    for rank, res in enumerate(results[:5]):
        print(
            f"Rank {rank+1} (local Dice: {res['dice']:.5f}) -> "
            f"threshold: {res['thresh']} | "
            f"minimum area: {res['min_size']} | "
            f"opening: {res['opening']} | "
            f"convex hull: {res['convex_hull']}"
        )
    print(f"{'='*50}")

    best_config = results[0]
    print("\nRecommended actions:")
    print(f"1. The estimated local validation Dice ceiling is {best_config['dice']:.5f}.")
    print(
        "2. Apply the best configuration to validation metrics or submission inference:"
    )
    print(f"   - Replace the default binarization threshold with: {best_config['thresh']}")
    print(f"   - Enable opening: {best_config['opening']}")
    print(f"   - Enable convex-hull filling: {best_config['convex_hull']}")


if __name__ == "__main__":
    main()
