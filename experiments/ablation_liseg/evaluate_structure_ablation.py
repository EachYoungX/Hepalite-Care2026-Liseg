import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.ablation_liseg import run_ablation as core
from experiments.ablation_liseg.variants import ControlledStudent


TRAINING_DIR = core.LOCAL_ARTIFACT_DIR / "training"
STRUCTURE_CACHE = core.CACHE_DIR / "structure"

VARIANTS = {
    "real_1slice": {
        "label": "1 slice",
        "context_slices": 1,
        "use_coordconv": True,
        "use_se": True,
    },
    "real_3slice": {
        "label": "3 slices",
        "context_slices": 3,
        "use_coordconv": True,
        "use_se": True,
    },
    "real_5slice": {
        "label": "5 slices + SE",
        "context_slices": 5,
        "use_coordconv": True,
        "use_se": True,
    },
    "real_5slice_no_coord": {
        "label": "5 slices - CoordConv",
        "context_slices": 5,
        "use_coordconv": False,
        "use_se": True,
    },
    "real_5slice_no_se": {
        "label": "5 slices - SE",
        "context_slices": 5,
        "use_coordconv": True,
        "use_se": False,
    },
}


def checkpoint_path(variant, fold):
    return TRAINING_DIR / variant / f"fold_{fold}" / "checkpoints/best_model.pth"


def load_variant(variant, device):
    config = VARIANTS[variant]
    model = ControlledStudent(
        context_slices=config["context_slices"],
        use_coordconv=config["use_coordconv"],
        use_se=config["use_se"],
    )
    return model.to(device)


def ensure_caches(records, assignments, device):
    runtime_rows = []
    for variant in VARIANTS:
        cache_dir = STRUCTURE_CACHE / variant
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        inferred = 0
        for fold in range(1, 6):
            fold_records = [record for record in records if assignments[record["case"]] == fold]
            missing = [
                record
                for record in fold_records
                if not (cache_dir / f"{record['case']}_prob.npy").exists()
            ]
            if not missing:
                continue
            model = load_variant(variant, device)
            model.load_state_dict(
                torch.load(checkpoint_path(variant, fold), map_location=device, weights_only=True)
            )
            model.eval()
            for record in missing:
                image = core.load_volume(record["data_path"], ged4=True).astype(np.float32)
                probability = core.predict_probability(model, "controlled", image, device)
                np.save(cache_dir / f"{record['case']}_prob.npy", probability)
                inferred += 1
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        runtime_rows.append(
            {
                "variant": variant,
                "new_cases_inferred": inferred,
                "cache_runtime_seconds": time.perf_counter() - started,
            }
        )
    return runtime_rows


def evaluate(records, assignments):
    rows = []
    parameter_rows = []
    for variant, config in VARIANTS.items():
        model = ControlledStudent(
            context_slices=config["context_slices"],
            use_coordconv=config["use_coordconv"],
            use_se=config["use_se"],
        )
        parameter_rows.append(
            {
                "variant": variant,
                "label": config["label"],
                "context_slices": config["context_slices"],
                "coordconv": config["use_coordconv"],
                "se": config["use_se"],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
            }
        )
        for record in records:
            probability = np.load(STRUCTURE_CACHE / variant / f"{record['case']}_prob.npy")
            prediction = core.stage_predictions(probability)["final_topology"]
            target = core.load_volume(record["mask_path"]).astype(np.uint8)
            metrics = core.segmentation_metrics(prediction, target)
            rows.append(
                {
                    "case": record["case"],
                    "vendor": record["vendor"],
                    "fold": assignments[record["case"]],
                    "variant": variant,
                    "label": config["label"],
                    **metrics,
                }
            )
    return rows, parameter_rows


def aggregate(rows, parameter_rows):
    parameters = {row["variant"]: row for row in parameter_rows}
    output = []
    for variant, config in VARIANTS.items():
        group = [row for row in rows if row["variant"] == variant]
        dice = np.asarray([row["dice"] for row in group])
        hd95 = np.asarray([row["hd95_mm"] for row in group])
        dice_ci = core.bootstrap_ci(dice)
        hd_ci = core.bootstrap_ci(hd95)
        output.append(
            {
                **parameters[variant],
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
    return output


def paired_significance(rows):
    lookup = {(row["variant"], row["case"]): row for row in rows}
    cases = sorted({row["case"] for row in rows})
    comparisons = [
        ("real_1slice", "real_5slice", "1-slice vs 5-slice"),
        ("real_3slice", "real_5slice", "3-slice vs 5-slice"),
        ("real_5slice_no_coord", "real_5slice", "CoordConv contribution"),
        ("real_5slice_no_se", "real_5slice", "SE contribution"),
    ]
    output = []
    for variant_a, variant_b, comparison in comparisons:
        for metric in ["dice", "hd95_mm", "z_span_error_slices"]:
            a = np.asarray([lookup[(variant_a, case)][metric] for case in cases], dtype=float)
            b = np.asarray([lookup[(variant_b, case)][metric] for case in cases], dtype=float)
            try:
                statistic, p_value = wilcoxon(a, b, zero_method="wilcox")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            output.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "delta_full_minus_ablation": float((b - a).mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return output


def write_markdown(aggregate_rows, significance_rows):
    table_path = core.TABLE_DIR / "table_04_structure_ablation.md"
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(
            "| Variant | Params | Dice mean +/- SD | HD95 mm mean +/- SD | Precision | Recall | Z-span error |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in aggregate_rows:
            handle.write(
                f"| {row['label']} | {row['parameters'] / 1e6:.3f}M | "
                f"{row['dice_mean']:.4f} +/- {row['dice_std']:.4f} | "
                f"{row['hd95_mean_mm']:.2f} +/- {row['hd95_std_mm']:.2f} | "
                f"{row['precision_mean']:.4f} | {row['recall_mean']:.4f} | "
                f"{row['z_span_error_mean']:.2f} |\n"
            )
    with open(core.TABLE_DIR / "table_05_structure_significance.md", "w", encoding="utf-8") as handle:
        handle.write("| Comparison | Metric | Full - ablation | Wilcoxon p |\n")
        handle.write("|---|---|---:|---:|\n")
        for row in significance_rows:
            handle.write(
                f"| {row['comparison']} | {row['metric']} | "
                f"{row['delta_full_minus_ablation']:.5f} | {row['p_value']:.5g} |\n"
            )


def plot_structure(aggregate_rows):
    label_overrides = {
        "real_1slice": "1 slice",
        "real_3slice": "3 slices",
        "real_5slice": "5 slices\n+ SE",
        "real_5slice_no_coord": "5 slices\n- CoordConv",
        "real_5slice_no_se": "5 slices\n- SE",
    }
    labels = [label_overrides.get(row["variant"], row["label"]) for row in aggregate_rows]
    dice = np.asarray([row["dice_mean"] for row in aggregate_rows])
    dice_low = dice - np.asarray([row["dice_ci_low"] for row in aggregate_rows])
    dice_high = np.asarray([row["dice_ci_high"] for row in aggregate_rows]) - dice
    hd = np.asarray([row["hd95_mean_mm"] for row in aggregate_rows])
    hd_low = hd - np.asarray([row["hd95_ci_low_mm"] for row in aggregate_rows])
    hd_high = np.asarray([row["hd95_ci_high_mm"] for row in aggregate_rows]) - hd
    colors = ["#cbd5e1", "#7f9db9", "#2f5573", "#6aa6a4", "#2f665f"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axes[0].bar(x, dice, yerr=np.vstack([dice_low, dice_high]), capsize=4, color=colors, edgecolor="#1f2937", linewidth=0.7)
    axes[1].bar(x, hd, yerr=np.vstack([hd_low, hd_high]), capsize=4, color=colors, edgecolor="#1f2937", linewidth=0.7)
    for axis, ylabel, title in zip(axes, ["OOF Dice", "HD95 (mm)"], ["Dice", "HD95 (mm)"]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=11)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.grid(axis="y", alpha=0.16, color="#94a3b8")
        axis.tick_params(axis="y", labelsize=11)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)
    axes[0].set_ylim(0.70, 0.96)
    fig.tight_layout()
    core.save_figure(fig, "figure_06_structure_ablation")


def plot_convergence():
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for variant, config in VARIANTS.items():
        curves = []
        for fold in range(1, 6):
            path = TRAINING_DIR / variant / f"fold_{fold}" / "metrics_log.csv"
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            curves.append([float(row["val_dice"]) for row in rows])
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        epochs = np.arange(1, len(mean) + 1)
        axis.plot(epochs, mean, label=config["label"], linewidth=1.6)
        axis.fill_between(epochs, mean - std, mean + std, alpha=0.10)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Internal validation Dice")
    axis.set_ylim(0.45, 0.96)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    axis.set_title("Five-fold validation convergence (mean +/- SD)")
    fig.tight_layout()
    core.save_figure(fig, "figure_07_structure_convergence")


def main():
    core.seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = core.real_case_records()
    assignments = core.fold_assignments(records)
    for variant in VARIANTS:
        for fold in range(1, 6):
            path = checkpoint_path(variant, fold)
            if not path.exists():
                raise FileNotFoundError(path)
    runtime_rows = ensure_caches(records, assignments, device)
    rows, parameter_rows = evaluate(records, assignments)
    aggregate_rows = aggregate(rows, parameter_rows)
    significance_rows = paired_significance(rows)
    core.write_csv(core.METRIC_DIR / "structure_per_case_metrics.csv", rows)
    core.write_csv(core.METRIC_DIR / "structure_aggregate_metrics.csv", aggregate_rows)
    core.write_csv(core.METRIC_DIR / "structure_paired_significance.csv", significance_rows)
    core.write_csv(core.METRIC_DIR / "structure_cache_runtime.csv", runtime_rows)
    write_markdown(aggregate_rows, significance_rows)
    plot_structure(aggregate_rows)
    plot_convergence()
    for row in aggregate_rows:
        print(
            f"{row['label']}: Dice={row['dice_mean']:.4f}, "
            f"HD95={row['hd95_mean_mm']:.2f} mm, Params={row['parameters'] / 1e6:.3f}M"
        )


if __name__ == "__main__":
    main()
