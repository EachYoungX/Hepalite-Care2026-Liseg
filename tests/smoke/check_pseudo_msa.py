import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PseudoDetailedChecker:
    def __init__(self):
        self.stats = {
            "total_files": 0,
            "real_label_cases": 0,
            "valid_pseudo_cases": 0,
            "shape_mismatch": 0,
            "fg_too_large": 0,
            "fg_too_small": 0,
            "low_confidence": 0,
            "total_black_volumes": 0,
        }
        self.bad_cases = []

    def check_single(self, data_path):
        self.stats["total_files"] += 1

        mask_path = data_path.replace("_data.npy", "_mask.npy")
        pseudo_path = data_path.replace("_data.npy", "_pseudo_mask.npy")
        weight_path = data_path.replace("_data.npy", "_weight.npy")

        if os.path.exists(mask_path) and "_pseudo" not in mask_path:
            self.stats["real_label_cases"] += 1
            return

        if not os.path.exists(pseudo_path):
            return

        try:
            data = np.load(data_path)
            p_mask = np.load(pseudo_path)
            weight = np.load(weight_path)

            if data.ndim == 4:
                data = data[0]
            if p_mask.ndim == 4:
                p_mask = p_mask[0]
            if weight.ndim == 4:
                weight = weight[0]

            if data.shape != p_mask.shape:
                self.stats["shape_mismatch"] += 1
                return

            self.stats["valid_pseudo_cases"] += 1

            fg_ratio = np.sum(p_mask) / p_mask.size
            mean_weight = np.mean(weight)

            black_slices_count = 0
            for z in range(p_mask.shape[0]):
                if np.sum(p_mask[z]) == 0 and weight[z].mean() > 0.5:
                    black_slices_count += 1

            reason = None
            if fg_ratio == 0:
                self.stats["total_black_volumes"] += 1
                reason = "empty volume (no prediction)"
            elif fg_ratio > 0.20:
                self.stats["fg_too_large"] += 1
                reason = "excessive foreground area"
            elif fg_ratio < 0.001:
                self.stats["fg_too_small"] += 1
                reason = "insufficient foreground area"
            elif mean_weight < 0.25:
                self.stats["low_confidence"] += 1
                reason = "very low confidence"

            if reason:
                self.bad_cases.append(
                    {
                        "name": os.path.basename(data_path),
                        "fg_ratio": fg_ratio,
                        "mean_weight": mean_weight,
                        "reason": reason,
                        "data": data,
                        "mask": p_mask,
                        "weight": weight,
                    }
                )

        except Exception as e:
            print(f"Failed to parse {data_path}: {e}")

    def run(self):
        data_files = glob.glob(
            os.path.join(DATA_ROOT, "**", "*_data.npy"), recursive=True
        )
        for fp in tqdm(data_files, desc="Running quality checks"):
            self.check_single(fp)

        print("\n" + "=" * 50)
        print("         Pseudo-label quality report")
        print("=" * 50)
        print(f"Total cases scanned:       {self.stats['total_files']}")
        print(f"Manual-label cases skipped: {self.stats['real_label_cases']}")
        print(f"Pseudo-label cases checked: {self.stats['valid_pseudo_cases']}")
        print("-" * 30)
        print(f"[ERROR] Shape mismatches:       {self.stats['shape_mismatch']}")
        print(f"[ERROR] Empty volumes:           {self.stats['total_black_volumes']}")
        print(f"[WARNING] Excessive area:        {self.stats['fg_too_large']}")
        print(f"[WARNING] Insufficient area:     {self.stats['fg_too_small']}")
        print(f"[WARNING] Low confidence:        {self.stats['low_confidence']}")
        print("-" * 30)
        good = self.stats["valid_pseudo_cases"] - len(self.bad_cases)
        print(f"[DONE] Valid cases:       {good}")
        print(f"Pass rate:                {(good/self.stats['valid_pseudo_cases']*100):.2f}%")
        print("=" * 50)

        if self.bad_cases:
            self.visualize_bad()
            self.save_bad_list()

    def save_bad_list(self):
        report_path = os.path.join(OUTPUT_DIR, "bad_cases_list.txt")
        with open(report_path, "w") as f:
            f.write("Pseudo-label quality check - invalid cases\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Invalid case count: {len(self.bad_cases)}\n\n")
            for i, c in enumerate(self.bad_cases):
                f.write(
                    f"{i+1}. {c['name']} | {c['reason']} | fg_ratio={c['fg_ratio']:.4f} | mean_weight={c['mean_weight']:.3f}\n"
                )
        print(f"\nInvalid-case list saved to: {report_path}")

    def visualize_bad(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        samples = sorted(self.bad_cases, key=lambda x: x["fg_ratio"], reverse=True)[:8]
        for i, c in enumerate(samples):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            z = c["mask"].shape[0] // 2
            axes[0].imshow(c["data"][z], cmap="gray")
            axes[0].set_title("Input")
            axes[1].imshow(c["mask"][z], cmap="jet")
            axes[1].set_title(f"Mask (Ratio:{c['fg_ratio']:.3f})")
            axes[2].imshow(c["weight"][z], cmap="hot")
            axes[2].set_title(f"Weight (Mean:{c['mean_weight']:.2f})")
            plt.suptitle(f"{c['name']} | {c['reason']}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"fail_case_{i}.png"))
            plt.close()


if __name__ == "__main__":
    DATA_ROOT = str(PROJECT_ROOT / "dataset" / "processed")
    OUTPUT_DIR = str(PROJECT_ROOT / "artifacts" / "analysis" / "pseudo_quality_report")
    checker = PseudoDetailedChecker()
    checker.run()
