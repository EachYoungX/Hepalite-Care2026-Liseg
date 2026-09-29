"""Generate the HepaLite architecture and training overview figure."""

from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.postprocess.topology import postprocess_prob_volume  # noqa: E402


SOURCE_FIGURE_DIR = (
    WORKSPACE_ROOT / "publication/camera_ready/CARE_12_source/figures"
)
SUBMISSION_FIGURE_DIR = (
    WORKSPACE_ROOT / "tmp/pdfs/care12_email_update_20260826/figures"
)
DATA_PATH = (
    WORKSPACE_ROOT
    / "dataset/processed/training_set/Vendor_B2/1041-B2-S3_data.npy"
)
PROBABILITY_PATH = (
    WORKSPACE_ROOT
    / "artifacts/experiments/ablation_liseg/cache/student_transductive_nose"
    / "1041-B2-S3_prob.npy"
)

NAVY = "#173F67"
BLUE = "#3F78B5"
BLUE_LIGHT = "#E8F1FA"
TEAL = "#147D78"
TEAL_LIGHT = "#E2F3F1"
ORANGE = "#C86624"
ORANGE_LIGHT = "#FFF0E4"
GREEN = "#16845B"
GREEN_LIGHT = "#E4F5EC"
INK = "#17212B"
GRAY = "#53606B"
LINE = "#73808C"
PANEL = "#FDFEFE"


def rounded_box(
    axis,
    x,
    y,
    width,
    height,
    text,
    edge=LINE,
    face="white",
    fontsize=8.3,
    weight="normal",
    linestyle="-",
    linewidth=1.0,
    zorder=2,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.005,rounding_size=0.006",
        edgecolor=edge,
        facecolor=face,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=INK,
        zorder=zorder + 1,
    )
    return patch


def arrow(axis, start, end, color=LINE, linewidth=1.0, connectionstyle="arc3"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
            zorder=3,
        )
    )


def panel(axis, y, height, label, title, edge=LINE):
    rounded_box(
        axis,
        0.008,
        y,
        0.984,
        height,
        "",
        edge=edge,
        face=PANEL,
        linewidth=1.15,
        zorder=0,
    )
    axis.text(
        0.016,
        y + height - 0.018,
        f"({label})  {title}",
        ha="left",
        va="top",
        fontsize=13.2,
        weight="bold",
        color=INK,
        zorder=5,
    )


def image_inset(axis, bounds, image, mask=None, contour_color=GREEN, probability=False):
    inset = axis.inset_axes(bounds, transform=axis.transAxes)
    if probability:
        inset.imshow(image, cmap="magma", vmin=0, vmax=1)
    elif np.issubdtype(image.dtype, np.bool_) or np.array_equal(
        np.unique(image), np.asarray([0, 1])
    ):
        inset.imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        nonzero = image[np.abs(image) > 1e-6]
        if nonzero.size:
            vmin, vmax = np.percentile(nonzero, [1, 99])
        else:
            vmin, vmax = 0, 1
        inset.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    if mask is not None and np.any(mask):
        inset.contour(mask, levels=[0.5], colors=[contour_color], linewidths=1.35)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#AAB4BD")
        spine.set_linewidth(0.65)
    return inset


def feature_stack(axis, x, y, width, height, color, label, layers=3):
    for index in range(layers - 1, -1, -1):
        offset = index * 0.004
        axis.add_patch(
            Rectangle(
                (x + offset, y + offset),
                width,
                height,
                facecolor=color if index == 0 else "white",
                edgecolor=NAVY,
                linewidth=0.75,
                alpha=0.95,
                zorder=2 + index,
            )
        )
    axis.text(
        x + width / 2,
        y + height + 0.012,
        label,
        ha="center",
        va="bottom",
        fontsize=8.2,
        weight="bold",
        color=NAVY,
    )


def draw_architecture(axis, slices):
    labels = [r"$z-2$", r"$z-1$", r"$z$", r"$z+1$", r"$z+2$"]
    start_x = 0.022
    tile_w = 0.041
    gap = 0.004
    for index, (slice_image, label) in enumerate(zip(slices, labels)):
        x = start_x + index * (tile_w + gap)
        image_inset(axis, [x, 0.718, tile_w, 0.112], slice_image)
        axis.text(
            x + tile_w / 2,
            0.838,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
            color=INK,
        )
    arrow(axis, (0.250, 0.774), (0.269, 0.774), color=NAVY)
    rounded_box(
        axis,
        0.269,
        0.724,
        0.061,
        0.102,
        "x/y\nCoordConv",
        edge=NAVY,
        face=BLUE_LIGHT,
        fontsize=8.6,
        weight="bold",
    )
    arrow(axis, (0.330, 0.774), (0.348, 0.774), color=NAVY)

    encoder = [
        (0.350, 0.753, 0.019, 0.070, "C16"),
        (0.404, 0.720, 0.023, 0.094, "C32"),
        (0.463, 0.687, 0.027, 0.116, "C64"),
        (0.527, 0.653, 0.031, 0.139, "C128"),
    ]
    decoder = [
        (0.594, 0.687, 0.027, 0.116, "C64"),
        (0.650, 0.720, 0.023, 0.094, "C32"),
        (0.701, 0.753, 0.019, 0.070, "C16"),
    ]
    for x, y, width, height, label in encoder:
        feature_stack(axis, x, y, width, height, BLUE, label)
    for x, y, width, height, label in decoder:
        feature_stack(axis, x, y, width, height, "#B7D2ED", label)
    for left, right in zip(encoder[:-1], encoder[1:]):
        arrow(
            axis,
            (left[0] + left[2] + 0.014, left[1] + left[3] / 2),
            (right[0] - 0.006, right[1] + right[3] / 2),
            color=NAVY,
        )
    arrow(axis, (0.572, 0.721), (0.590, 0.745), color=BLUE)
    for left, right in zip(decoder[:-1], decoder[1:]):
        arrow(
            axis,
            (left[0] + left[2] + 0.014, left[1] + left[3] / 2),
            (right[0] - 0.006, right[1] + right[3] / 2),
            color=BLUE,
        )
    skip_pairs = [(encoder[0], decoder[2]), (encoder[1], decoder[1]), (encoder[2], decoder[0])]
    for source, target in skip_pairs:
        y = source[1] + source[3] + 0.013
        axis.plot(
            [source[0] + source[2] / 2, source[0] + source[2] / 2, target[0] + target[2] / 2],
            [source[1] + source[3], y, y],
            color=GRAY,
            linewidth=0.85,
            zorder=1,
        )
        arrow(axis, (target[0] + target[2] / 2, y), (target[0] + target[2] / 2, target[1] + target[3]), color=GRAY)
    axis.text(
        0.532,
        0.623,
        "stride-2 downsampling     transposed-conv upsampling     skip connections",
        ha="center",
        va="center",
        fontsize=7.5,
        color=GRAY,
    )

    for index, y in enumerate([0.798, 0.766, 0.734, 0.702]):
        rounded_box(
            axis,
            0.736,
            y,
            0.038,
            0.024,
            f"DS {index + 1}",
            edge=TEAL,
            face=TEAL_LIGHT,
            fontsize=7.0,
        )
    axis.text(
        0.755,
        0.667,
        "4-level deep supervision",
        ha="center",
        va="center",
        fontsize=7.4,
        weight="bold",
        color=TEAL,
    )

    rounded_box(
        axis,
        0.793,
        0.634,
        0.186,
        0.205,
        "",
        edge=NAVY,
        face="white",
        linestyle="--",
        linewidth=1.05,
    )
    axis.text(
        0.886,
        0.817,
        "Factorized residual block",
        ha="center",
        va="center",
        fontsize=9.2,
        weight="bold",
        color=INK,
    )
    rounded_box(axis, 0.807, 0.739, 0.043, 0.041, "Conv\n5×1", edge=NAVY, face=BLUE_LIGHT, fontsize=7.2)
    rounded_box(axis, 0.807, 0.682, 0.043, 0.041, "Conv\n1×5", edge=NAVY, face=BLUE_LIGHT, fontsize=7.2)
    axis.text(0.863, 0.731, "+", ha="center", va="center", fontsize=15, color=INK)
    rounded_box(axis, 0.880, 0.711, 0.031, 0.040, "BN", edge=LINE, face="white", fontsize=7.3)
    rounded_box(axis, 0.919, 0.711, 0.038, 0.040, "GELU", edge=LINE, face="white", fontsize=7.0)
    rounded_box(axis, 0.880, 0.655, 0.036, 0.034, "SE*", edge=TEAL, face=TEAL_LIGHT, fontsize=7.2)
    rounded_box(axis, 0.926, 0.655, 0.036, 0.034, "Conv\n1×1", edge=NAVY, face=BLUE_LIGHT, fontsize=6.8)
    axis.text(0.970, 0.672, "⊕", ha="center", va="center", fontsize=12, color=INK)
    arrow(axis, (0.850, 0.759), (0.858, 0.739), color=NAVY)
    arrow(axis, (0.850, 0.702), (0.858, 0.725), color=NAVY)
    arrow(axis, (0.869, 0.731), (0.878, 0.731), color=NAVY)
    arrow(axis, (0.911, 0.731), (0.917, 0.731), color=NAVY)
    arrow(axis, (0.938, 0.709), (0.899, 0.690), color=TEAL)
    arrow(axis, (0.916, 0.672), (0.924, 0.672), color=NAVY)
    arrow(axis, (0.962, 0.672), (0.966, 0.672), color=NAVY)
    axis.plot(
        [0.800, 0.800, 0.970, 0.970],
        [0.731, 0.792, 0.792, 0.681],
        color=GRAY,
        linewidth=0.8,
        zorder=1,
    )
    axis.text(
        0.886,
        0.644,
        "Teacher: SE*     Final student: no SE",
        ha="center",
        va="top",
        fontsize=7.5,
        weight="bold",
        color=TEAL,
    )


def draw_training(axis):
    rounded_box(axis, 0.023, 0.468, 0.090, 0.061, "30 labeled\nGED4 cases", edge=NAVY, face=BLUE_LIGHT, fontsize=8.0, weight="bold")
    rounded_box(axis, 0.023, 0.375, 0.090, 0.061, "430 train + 60 val\nunlabeled volumes", edge=TEAL, face=TEAL_LIGHT, fontsize=7.8, weight="bold")
    arrow(axis, (0.113, 0.498), (0.139, 0.498), color=NAVY)
    axis.text(0.210, 0.535, "5-fold teachers (SE)", ha="center", va="center", fontsize=8.9, weight="bold", color=NAVY)
    for index in range(5):
        rounded_box(axis, 0.142 + index * 0.029, 0.469, 0.025, 0.050, f"T{index + 1}", edge=NAVY, face=BLUE_LIGHT, fontsize=7.6, weight="bold")
    arrow(axis, (0.113, 0.405), (0.312, 0.455), color=TEAL, connectionstyle="arc3,rad=-0.09")
    arrow(axis, (0.289, 0.493), (0.313, 0.493), color=NAVY)
    rounded_box(axis, 0.313, 0.455, 0.092, 0.077, "5 probability\nvolumes", edge=TEAL, face=TEAL_LIGHT, fontsize=8.0, weight="bold")
    arrow(axis, (0.405, 0.493), (0.429, 0.493), color=TEAL)
    rounded_box(axis, 0.429, 0.455, 0.084, 0.077, "Mean $\\bar{p}$\nVariance $v$", edge=TEAL, face=TEAL_LIGHT, fontsize=8.2, weight="bold")
    arrow(axis, (0.513, 0.493), (0.538, 0.493), color=TEAL)
    rounded_box(axis, 0.538, 0.455, 0.096, 0.077, "3D refinement\nthreshold 0.350", edge=ORANGE, face=ORANGE_LIGHT, fontsize=7.8, weight="bold")
    arrow(axis, (0.634, 0.493), (0.656, 0.493), color=ORANGE)
    rounded_box(axis, 0.656, 0.455, 0.082, 0.077, "Pseudo-mask\n$\hat{y}$", edge=ORANGE, face=ORANGE_LIGHT, fontsize=8.2, weight="bold")
    arrow(axis, (0.738, 0.493), (0.760, 0.493), color=TEAL)
    rounded_box(axis, 0.760, 0.455, 0.086, 0.077, "Confidence\nweight $w$", edge=TEAL, face=TEAL_LIGHT, fontsize=8.1, weight="bold")

    rounded_box(axis, 0.314, 0.364, 0.532, 0.057, "Manual masks + weighted pseudo supervision     train ×1.0     validation ×0.35", edge=TEAL, face="#F0FAF8", fontsize=8.3, weight="bold")
    arrow(axis, (0.068, 0.468), (0.314, 0.392), color=NAVY, connectionstyle="arc3,rad=0.12")
    arrow(axis, (0.697, 0.455), (0.697, 0.421), color=ORANGE)
    arrow(axis, (0.803, 0.455), (0.803, 0.421), color=TEAL)
    arrow(axis, (0.846, 0.392), (0.870, 0.392), color=TEAL)
    rounded_box(axis, 0.870, 0.358, 0.103, 0.173, "", edge=TEAL, face="white", linestyle="--", linewidth=1.05)
    axis.text(0.921, 0.514, "fold-specific students", ha="center", va="center", fontsize=8.2, weight="bold", color=TEAL)
    for index in range(5):
        rounded_box(axis, 0.885, 0.472 - index * 0.024, 0.073, 0.018, f"Student {index + 1}", edge=TEAL, face=GREEN_LIGHT, fontsize=6.8)


def draw_inference(axis, slices, probability, final_mask):
    image_inset(axis, [0.025, 0.083, 0.080, 0.142], slices[2])
    axis.text(0.065, 0.235, "GED4 input", ha="center", va="bottom", fontsize=8.2, weight="bold", color=INK)
    arrow(axis, (0.107, 0.154), (0.125, 0.154), color=GRAY)
    for index, slice_image in enumerate(slices):
        image_inset(axis, [0.126 + index * 0.029, 0.103, 0.032, 0.098], slice_image)
    axis.text(0.200, 0.214, "five-slice context", ha="center", va="bottom", fontsize=8.0, weight="bold", color=INK)
    arrow(axis, (0.274, 0.154), (0.294, 0.154), color=TEAL)
    rounded_box(axis, 0.294, 0.079, 0.100, 0.151, "", edge=TEAL, face="white", linestyle="--")
    axis.text(0.344, 0.214, "5 students", ha="center", va="bottom", fontsize=8.1, weight="bold", color=TEAL)
    for index in range(5):
        rounded_box(axis, 0.309, 0.184 - index * 0.024, 0.070, 0.018, f"Fold {index + 1}", edge=TEAL, face=GREEN_LIGHT, fontsize=6.8)
    arrow(axis, (0.394, 0.154), (0.416, 0.154), color=TEAL)
    image_inset(axis, [0.418, 0.103, 0.068, 0.098], probability, probability=True)
    axis.text(0.452, 0.214, r"mean probability $\bar{p}$", ha="center", va="bottom", fontsize=7.8, weight="bold", color=INK)
    arrow(axis, (0.488, 0.154), (0.507, 0.154), color=ORANGE)
    image_inset(axis, [0.509, 0.103, 0.068, 0.098], final_mask)
    axis.text(0.543, 0.214, "threshold 0.375", ha="center", va="bottom", fontsize=7.8, weight="bold", color=INK)
    arrow(axis, (0.579, 0.154), (0.596, 0.154), color=ORANGE)

    steps = ["2D\nopening", "largest\n3D CC", "3D hole\nfilling", "axial\ncontinuity", "end-slice\nrescue"]
    step_x = 0.598
    step_w = 0.054
    gap = 0.008
    for index, text in enumerate(steps):
        x = step_x + index * (step_w + gap)
        rounded_box(axis, x, 0.116, step_w, 0.073, text, edge=ORANGE, face=ORANGE_LIGHT, fontsize=7.0, weight="bold")
        if index < len(steps) - 1:
            arrow(axis, (x + step_w, 0.153), (x + step_w + gap, 0.153), color=ORANGE)
    axis.text(0.750, 0.214, "deterministic 3D refinement", ha="center", va="bottom", fontsize=8.1, weight="bold", color=ORANGE)
    arrow(axis, (0.900, 0.154), (0.918, 0.154), color=GREEN)
    image_inset(axis, [0.920, 0.083, 0.059, 0.142], slices[2], mask=final_mask, contour_color=GREEN)
    axis.text(0.949, 0.235, "restored NIfTI", ha="center", va="bottom", fontsize=8.1, weight="bold", color=GREEN)


def load_example():
    data = np.load(DATA_PATH, mmap_mode="r")
    ged4 = np.asarray(data[6] if data.ndim == 4 and data.shape[0] > 6 else data[0])
    probability = np.asarray(np.load(PROBABILITY_PATH), dtype=np.float32)
    final_mask = postprocess_prob_volume(
        probability,
        threshold=0.375,
        opening=True,
        fill_holes=True,
        min_size_filter=50,
        min_overlap_iou=0.03,
        centroid_jump_limit=40.0,
    )
    areas = final_mask.reshape(final_mask.shape[0], -1).sum(axis=1)
    z = int(np.argmax(areas))
    indices = [max(0, min(ged4.shape[0] - 1, z + offset)) for offset in (-2, -1, 0, 1, 2)]
    return [ged4[index] for index in indices], probability[z], final_mask[z]


def generate_figure():
    slices, probability, final_mask = load_example()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(15.4, 8.25))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    panel(axis, 0.602, 0.386, "a", "Lightweight 2.5D HepaLite architecture", edge=NAVY)
    panel(axis, 0.337, 0.250, "b", "Shared-teacher pseudo-label training", edge=TEAL)
    panel(axis, 0.015, 0.307, "c", "Ensemble inference and deterministic 3D refinement", edge=ORANGE)
    draw_architecture(axis, slices)
    draw_training(axis)
    draw_inference(axis, slices, probability, final_mask)

    figure.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    SOURCE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = SOURCE_FIGURE_DIR / "figure_00_method_overview.pdf"
    output_png = SOURCE_FIGURE_DIR / "figure_00_method_overview.png"
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    SUBMISSION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_pdf, SUBMISSION_FIGURE_DIR / output_pdf.name)
    shutil.copy2(output_png, SUBMISSION_FIGURE_DIR / output_png.name)
    print(output_pdf)
    print(output_png)


if __name__ == "__main__":
    generate_figure()
