"""Generate a compact qualitative figure for HD95-sensitive visual patterns.

The validation cases have hidden ground truth. This figure therefore shows GED4
images with final pseudo-label contours only and avoids assigning verified
pathology or error-source labels.
"""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.ndimage import binary_erosion

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    }
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "publication/camera_ready/CARE_12_source/figures"
VALIDATION_DIR = ROOT / "dataset" / "processed" / "validation_set"


RISK_CASES = [
    {
        "case": "1055-B2",
        "title": "Low-contrast boundary",
        "note": "A weak inferior margin can change boundary distance while affecting limited area.",
    },
    {
        "case": "1146-B1",
        "title": "Tip truncation",
        "note": "A small missed tip is a minor overlap change but may be distance-sensitive.",
    },
    {
        "case": "1186-B1",
        "title": "Off-target fragmented response",
        "note": "Small separated responses can contribute long-distance surface outliers.",
    },
    {
        "case": "1096-B2",
        "title": "Axial-profile instability",
        "note": "A non-smooth axial extent can be more visible to surface-distance metrics.",
    },
]


def wrap(text, width=48):
    return "\n".join(textwrap.wrap(text, width=width))


def case_paths(case):
    for vendor_dir in sorted(VALIDATION_DIR.glob("Vendor_*")):
        data_path = vendor_dir / f"{case}_data.npy"
        mask_path = vendor_dir / f"{case}_pseudo_mask.npy"
        if data_path.exists() and mask_path.exists():
            return data_path, mask_path
    raise FileNotFoundError(case)


def robust_window(slice_2d):
    low, high = np.percentile(slice_2d, [1, 99])
    if high <= low:
        return slice_2d
    return np.clip((slice_2d - low) / (high - low), 0, 1)


def select_slice(mask):
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    positive = np.flatnonzero(areas > 0)
    if positive.size == 0:
        return mask.shape[0] // 2
    z_min, z_max = int(positive[0]), int(positive[-1])
    case_span = z_max - z_min + 1
    if case_span <= 3:
        return int(positive[len(positive) // 2])
    # Pick a representative non-central slice to make thin tips and fragmented
    # responses visible without using hidden validation annotations.
    if areas[z_min] < 0.18 * areas.max():
        return z_min
    if areas[z_max] < 0.18 * areas.max():
        return z_max
    return int(positive[len(positive) // 4])


def contour(mask_slice):
    if not np.any(mask_slice):
        return np.zeros_like(mask_slice, dtype=bool)
    return mask_slice.astype(bool) ^ binary_erosion(mask_slice.astype(bool))


def add_text_box(axis):
    axis.axis("off")
    box = FancyBboxPatch(
        (0.035, 0.08),
        0.93,
        0.84,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#475569",
        facecolor="#f8fafc",
    )
    axis.add_patch(box)
    axis.text(
        0.07,
        0.76,
        "Why high Dice can coexist with relatively high HD95",
        fontsize=15,
        weight="bold",
        color="#0f172a",
    )
    points = [
        "Dice reflects volumetric overlap and may remain high despite localized boundary deviations.",
        "HD95 emphasizes tail surface distances, so thin tips and fragmented responses can dominate.",
    ]
    for i, point in enumerate(points):
        axis.text(0.075, 0.50 - 0.22 * i, wrap(point, 100), fontsize=12, color="#334155", va="top")


def panel(axis, item):
    data_path, mask_path = case_paths(item["case"])
    image = np.load(data_path)[0]
    mask = np.load(mask_path).astype(bool)
    z = select_slice(mask)
    axis.imshow(robust_window(image[z]), cmap="gray", interpolation="nearest")
    edge = contour(mask[z])
    if np.any(edge):
        axis.contour(edge, levels=[0.5], colors=["#06b6d4"], linewidths=1.2)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"{item['case']} | {item['title']}", fontsize=12, weight="bold", pad=5)
    axis.text(
        0.0,
        -0.085,
        wrap(item["note"], 50),
        transform=axis.transAxes,
        fontsize=10.2,
        color="#334155",
        va="top",
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#cbd5e1")


def hd95_risk_modes():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.8, 8.8))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.62, 1.0, 1.0], hspace=0.46, wspace=0.14)

    add_text_box(fig.add_subplot(grid[0, :]))
    for index, item in enumerate(RISK_CASES):
        panel(fig.add_subplot(grid[1 + index // 2, index % 2]), item)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.97, bottom=0.055)
    fig.savefig(FIGURE_DIR / "figure_11_hd95_risk_modes.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure_11_hd95_risk_modes.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    hd95_risk_modes()
    print(f"HD95 risk-mode figure written to {FIGURE_DIR}")
