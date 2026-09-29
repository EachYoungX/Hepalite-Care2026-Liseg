#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remove pseudo-label and confidence-weight files while preserving manual masks."""

import os
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "dataset" / "processed"


def main():
    print("=" * 60)
    print("Pseudo-label and confidence-weight cleanup utility")
    print("=" * 60)

    pseudo_mask_files = glob.glob(str(DATA_ROOT / "**" / "*_pseudo_mask.npy"), recursive=True)

    weight_files = glob.glob(str(DATA_ROOT / "**" / "*_weight.npy"), recursive=True)

    real_mask_files = glob.glob(str(DATA_ROOT / "**" / "*_mask.npy"), recursive=True)
    real_mask_files = [f for f in real_mask_files if "_pseudo_mask" not in f]

    print(f"Pseudo-label files found: {len(pseudo_mask_files)}")
    print(f"Confidence-weight files found: {len(weight_files)}")
    print(f"Manual-mask files retained: {len(real_mask_files)}")
    print("-" * 60)

    if not pseudo_mask_files and not weight_files:
        print("[DONE] No files require removal.")
        return

    if pseudo_mask_files:
        print("\nPseudo-label file preview:")
        for f in pseudo_mask_files[:10]:
            rel = Path(f).relative_to(PROJECT_ROOT)
            print(f"  - {rel}")
        if len(pseudo_mask_files) > 10:
            print(f"  ... {len(pseudo_mask_files) - 10} additional files")

    if weight_files:
        print("\nConfidence-weight file preview:")
        for f in weight_files[:10]:
            rel = Path(f).relative_to(PROJECT_ROOT)
            print(f"  - {rel}")
        if len(weight_files) > 10:
            print(f"  ... {len(weight_files) - 10} additional files")

    print("\n" + "=" * 60)
    print("[WARNING] The pseudo-label and confidence-weight files above will be removed.")
    print("[WARNING] Manual masks will not be removed.")
    print("=" * 60)

    confirm = input("\nConfirm removal? (enter y to continue): ").strip().lower()
    if confirm != "y":
        print("[CANCELLED] Removal cancelled.")
        return

    removed_pseudo = 0
    removed_weight = 0

    for f in pseudo_mask_files:
        try:
            os.remove(f)
            removed_pseudo += 1
        except Exception as e:
            print(f"[ERROR] Failed to remove {f}: {e}")

    for f in weight_files:
        try:
            os.remove(f)
            removed_weight += 1
        except Exception as e:
            print(f"[ERROR] Failed to remove {f}: {e}")

    print("\n" + "=" * 60)
    print("[DONE] Removal completed.")
    print(f"  Pseudo-labels removed: {removed_pseudo}")
    print(f"  Confidence weights removed: {removed_weight}")
    print(f"  Manual masks retained: {len(real_mask_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
