import os
import glob
import csv

from _project import PROJECT_ROOT  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass


CONFIG = {
    "processed_val_dir": "dataset/processed/validation_set",
    "temp_pred_dir": "artifacts/predictions/temp_predictions",
    "review_dir": "artifacts/analysis/manual_review_cases",
    "summary_csv": "artifacts/analysis/manual_review_cases/review_summary.csv",
    "ged4_idx": 6,
    "num_slices": 8,
    "top_k": 18,
}


def load_ged4(path):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 4:
        return np.asarray(arr[CONFIG["ged4_idx"]] if arr.shape[0] > CONFIG["ged4_idx"] else arr[0])
    return np.asarray(arr)


def robust_window(img):
    nz = img[np.abs(img) > 1e-6]
    if nz.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(nz, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def mask_bbox(mask_2d):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def case_stats(case_name, data_path, mask_path):
    img = load_ged4(data_path)
    mask = np.load(mask_path)
    if mask.ndim == 4:
        mask = mask[0]

    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    fg_z = np.where(areas > 0)[0]
    img_nonzero = (np.abs(img) > 1e-6).reshape(img.shape[0], -1).sum(axis=1)
    black_pred = int(np.sum((areas > 0) & (img_nonzero <= 16)))

    if len(fg_z) == 0:
        return {
            "case": case_name,
            "volume": 0,
            "z_count": 0,
            "z_start": -1,
            "z_end": -1,
            "max_area": 0,
            "black_pred_slices": black_pred,
            "centroid_x_mean": -1,
            "centroid_y_mean": -1,
            "risk_score": 999999.0,
        }

    centroids = []
    for z in fg_z:
        c = center_of_mass(mask[z] > 0)
        if np.isfinite(c[0]) and np.isfinite(c[1]):
            centroids.append((c[1], c[0]))
    centroid_x = float(np.mean([c[0] for c in centroids])) if centroids else -1.0
    centroid_y = float(np.mean([c[1] for c in centroids])) if centroids else -1.0

    # High score means worth human inspection: extreme z span, black predictions,
    # left/right drift, or unusually large/small volume.
    volume = int(mask.sum())
    z_count = int(len(fg_z))
    max_area = int(areas.max())
    risk = (
        abs(z_count - 22) * 4.0
        + abs(volume - 79000) / 2500.0
        + black_pred * 30.0
        + max(0.0, 118.0 - centroid_x) * 1.5
        + max(0.0, centroid_x - 188.0) * 1.5
    )

    return {
        "case": case_name,
        "volume": volume,
        "z_count": z_count,
        "z_start": int(fg_z.min()),
        "z_end": int(fg_z.max()),
        "max_area": max_area,
        "black_pred_slices": black_pred,
        "centroid_x_mean": round(centroid_x, 2),
        "centroid_y_mean": round(centroid_y, 2),
        "risk_score": round(float(risk), 2),
    }


def pick_slices(mask):
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    fg_z = np.where(areas > 0)[0]
    if len(fg_z) == 0:
        return np.linspace(0, mask.shape[0] - 1, CONFIG["num_slices"], dtype=int)

    candidates = np.unique(
        np.concatenate(
            [
                fg_z[:2],
                fg_z[-2:],
                np.linspace(fg_z.min(), fg_z.max(), CONFIG["num_slices"], dtype=int),
                np.array([int(areas.argmax())]),
            ]
        )
    )
    if len(candidates) >= CONFIG["num_slices"]:
        return candidates[: CONFIG["num_slices"]]
    pad = np.linspace(0, mask.shape[0] - 1, CONFIG["num_slices"], dtype=int)
    return np.unique(np.concatenate([candidates, pad]))[: CONFIG["num_slices"]]


def save_sheet(case_name, data_path, mask_path, out_path):
    img = robust_window(load_ged4(data_path))
    mask = np.load(mask_path)
    if mask.ndim == 4:
        mask = mask[0]

    z_list = pick_slices(mask)
    cols = 4
    rows = int(np.ceil(len(z_list) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).ravel()

    for ax, z in zip(axes, z_list):
        ax.imshow(img[z], cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*mask[z].shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = (mask[z] > 0) * 0.38
        ax.imshow(overlay)
        bbox = mask_bbox(mask[z])
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1 + 1,
                    y2 - y1 + 1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.0,
                )
            )
        ax.set_title(f"{case_name} | z={int(z)} | area={int(mask[z].sum())}")
        ax.axis("off")

    for ax in axes[len(z_list) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(CONFIG["review_dir"], exist_ok=True)
    data_files = glob.glob(os.path.join(CONFIG["processed_val_dir"], "*", "*_data.npy"))
    rows = []

    for data_path in data_files:
        case_name = os.path.basename(data_path).replace("_data.npy", "")
        mask_path = os.path.join(CONFIG["temp_pred_dir"], f"{case_name}_mask.npy")
        if not os.path.exists(mask_path):
            continue
        rows.append(case_stats(case_name, data_path, mask_path))

    rows = sorted(rows, key=lambda r: r["risk_score"], reverse=True)
    with open(CONFIG["summary_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows[: CONFIG["top_k"]]:
        case_name = row["case"]
        data_path = glob.glob(os.path.join(CONFIG["processed_val_dir"], "*", f"{case_name}_data.npy"))[0]
        mask_path = os.path.join(CONFIG["temp_pred_dir"], f"{case_name}_mask.npy")
        out_path = os.path.join(CONFIG["review_dir"], f"{case_name}_review.png")
        save_sheet(case_name, data_path, mask_path, out_path)

    print(f"Saved summary: {CONFIG['summary_csv']}")
    print(f"Saved review sheets: {CONFIG['review_dir']}")
    print("Top cases for manual review:")
    for row in rows[: CONFIG["top_k"]]:
        print(
            f"{row['case']} | risk={row['risk_score']} | vol={row['volume']} | "
            f"z={row['z_start']}-{row['z_end']} ({row['z_count']}) | "
            f"cx={row['centroid_x_mean']} | black_pred={row['black_pred_slices']}"
        )


if __name__ == "__main__":
    main()
