#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_WEIGHTS_DIR = Path(
    os.environ.get("CARE2026_DEPLOY_DIR", PROJECT_ROOT.parent / "CARE2026_Deploy")
) / "weights"

DEFAULT_TRAINING_OUTPUT = "20260612_104443_20260503_HepaLite_Student_SemiSup_V1"


def copy_weights_to_deploy(training_output_name):
    """Copy checkpoints from one training run into the deployment directory."""
    training_output_path = PROJECT_ROOT / "artifacts" / "training_runs" / training_output_name

    if not training_output_path.exists():
        print(f"[ERROR] Training output path not found: {training_output_path}")
        return

    DEPLOY_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Checkpoint deployment utility")
    print("=" * 60)
    print(f"Source: {training_output_path}")
    print(f"Destination: {DEPLOY_WEIGHTS_DIR}")
    print("-" * 60)

    fold_dirs = sorted([d for d in training_output_path.iterdir()
                       if d.is_dir() and d.name.startswith("fold_")])

    if not fold_dirs:
        print("[WARNING] No fold directories found.")
        return

    print(f"Found {len(fold_dirs)} fold directories.")
    print("-" * 60)

    copied_count = 0

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        fold_num = fold_name.split("_")[1]

        checkpoint_path = fold_dir / "checkpoints" / "best_model.pth"

        if not checkpoint_path.exists():
            print(f"[SKIP] {fold_name}: best_model.pth not found.")
            continue

        target_path = DEPLOY_WEIGHTS_DIR / f"fold_{fold_num}.pth"

        try:
            shutil.copy2(checkpoint_path, target_path)
            print(f"[DONE] {fold_name}/checkpoints/best_model.pth -> fold_{fold_num}.pth")
            copied_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to copy {fold_name}: {e}")

    print("-" * 60)
    print(f"[DONE] Copied {copied_count}/{len(fold_dirs)} checkpoints.")
    print(f"Checkpoints saved to: {DEPLOY_WEIGHTS_DIR}")
    print("=" * 60)


def list_available_training_outputs():
    """List available training output directories."""
    outputs_dir = PROJECT_ROOT / "artifacts" / "training_runs"

    if not outputs_dir.exists():
        print("[ERROR] artifacts/training_runs directory not found.")
        return []

    training_outputs = sorted([d.name for d in outputs_dir.iterdir()
                              if d.is_dir() and not d.name.startswith(".")])

    return training_outputs


def main():
    """List available training outputs and copy the configured run."""
    print("\nAvailable training outputs:")
    print("-" * 60)

    available_outputs = list_available_training_outputs()

    if not available_outputs:
        print("[ERROR] No training outputs found.")
        return

    for i, output_name in enumerate(available_outputs, 1):
        print(f"{i}. {output_name}")

    print("-" * 60)
    print(f"\nUsing default training output: {DEFAULT_TRAINING_OUTPUT}")

    if DEFAULT_TRAINING_OUTPUT in available_outputs:
        copy_weights_to_deploy(DEFAULT_TRAINING_OUTPUT)
    else:
        print(f"[ERROR] Default training output not found: {DEFAULT_TRAINING_OUTPUT}")
        print("Update DEFAULT_TRAINING_OUTPUT in the script.")


if __name__ == "__main__":
    main()
