import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

METRIC_DIR = ROOT / "experiments" / "ablation_liseg" / "results" / "metrics"
FIGURE_DIR = ROOT / "publication" / "camera_ready" / "CARE_12_source" / "figures"
STAGES = ["raw_035", "opening", "cca", "fill_holes", "z_chain", "final_topology"]
LABELS = {
    "raw_035": "Raw 0.350",
    "opening": "+ Opening",
    "cca": "+ Largest 3D CC",
    "fill_holes": "+ 3D hole filling",
    "z_chain": "+ Z-chain",
    "final_topology": "+ End-slice rescue",
}


def evaluate(records):
    rows = []
    for record in records:
        probability_path = CACHE_DIR / f"{record['case']}_prob.npy"
        if not probability_path.exists():
            raise FileNotFoundError(probability_path)
        probability = np.load(probability_path)
        target = core.load_volume(record["mask_path"]).astype(np.uint8)
        predictions = core.stage_predictions(probability)
        for stage in STAGES:
            rows.append(
                {
                    "case": record["case"],
                    "vendor": record["vendor"],
                    "stage": stage,
                    "label": LABELS[stage],
                    **core.segmentation_metrics(predictions[stage], target),
                }
            )
    return rows


def summarize(rows):
    output = []
    for stage in STAGES:
        group = [row for row in rows if row["stage"] == stage]
        dice = np.asarray([row["dice"] for row in group], dtype=float)
        hd95 = np.asarray([row["hd95_mm"] for row in group], dtype=float)
        output.append(
            {
                "stage": stage,
                "label": LABELS[stage],
                "n": len(group),
                "dice_mean": float(dice.mean()),
                "dice_std": float(dice.std(ddof=1)),
                "hd95_mean_mm": float(hd95.mean()),
                "hd95_std_mm": float(hd95.std(ddof=1)),
            }
        )
    return output


def paired_statistics(rows):
    lookup = {(row["stage"], row["case"]): row for row in rows}
    cases = sorted({row["case"] for row in rows})
    output = []
    for previous, current in zip(STAGES, STAGES[1:]):
        for metric in ("dice", "hd95_mm"):
            a = np.asarray([lookup[(previous, case)][metric] for case in cases])
            b = np.asarray([lookup[(current, case)][metric] for case in cases])
            if np.allclose(a, b):
                statistic, p_value = 0.0, 1.0
            else:
                try:
                    statistic, p_value = wilcoxon(a, b, zero_method="wilcox")
                except ValueError:
                    statistic, p_value = 0.0, 1.0
            output.append(
                {
                    "comparison": f"{LABELS[previous]} -> {LABELS[current]}",
                    "metric": metric,
                    "mean_before": float(a.mean()),
                    "mean_after": float(b.mean()),
                    "mean_delta": float((b - a).mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary):
    labels = [LABELS[row["stage"]] for row in summary]
    x = np.arange(len(labels))
    dice = [float(row["dice_mean"]) for row in summary]
    hd95 = [float(row["hd95_mean_mm"]) for row in summary]
    colors = ["#9ca3af", "#7f9db9", "#4d7895", "#4f8f86", "#2f7468", "#24564f"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    axes[0].plot(x, dice, marker="o", linewidth=2.1, color="#2f665f")
    axes[1].plot(x, hd95, marker="o", linewidth=2.1, color="#9a5b3f")
    for axis, values, ylabel in zip(axes, (dice, hd95), ("Dice", "HD95 (mm)")):
        axis.scatter(x, values, s=58, c=colors, edgecolor="#1f2937", linewidth=0.6, zorder=3)
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(labelsize=10)
        for index, value in enumerate(values):
            axis.annotate(f"{value:.4f}" if ylabel == "Dice" else f"{value:.2f}",
                          (index, value), xytext=(0, 8), textcoords="offset points",
                          ha="center", fontsize=9.5)
    axes[0].set_ylim(min(dice) - 0.004, max(dice) + 0.004)
    axes[1].set_ylim(0, max(hd95) * 1.18)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"figure_12_strict_nose_topology_ablation.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from the existing audited summary CSV without loading training dependencies.",
    )
    args = parser.parse_args()
    if args.plot_only:
        with (METRIC_DIR / "strict_nose_topology_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            plot(list(csv.DictReader(handle)))
        return

    global core, CACHE_DIR
    core = importlib.import_module("experiments.ablation_liseg.run_ablation")
    CACHE_DIR = core.CACHE_DIR / "structure" / "real_5slice_no_se"
    records = core.real_case_records()
    if len(records) != 30:
        raise RuntimeError(f"Expected 30 labeled cases, found {len(records)}")
    rows = evaluate(records)
    summary = summarize(rows)
    paired = paired_statistics(rows)
    write_csv(core.METRIC_DIR / "strict_nose_topology_per_case.csv", rows)
    write_csv(core.METRIC_DIR / "strict_nose_topology_summary.csv", summary)
    write_csv(core.METRIC_DIR / "strict_nose_topology_paired.csv", paired)
    decision = {
        "evaluation": "strict real-label-only NoSE OOF",
        "label_disjoint": True,
        "transductive": False,
        "cache": str(CACHE_DIR.relative_to(ROOT)),
        "threshold": 0.350,
        "case_count": len(records),
        "summary": summary,
        "paired_statistics": paired,
    }
    with (core.ARTIFACT_DIR / "strict_nose_topology_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, indent=2)
    plot(summary)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
