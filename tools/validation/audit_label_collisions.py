"""Fail when a manually labeled case also has stale pseudo-label artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path


def find_collisions(data_root: Path) -> list[tuple[Path, Path, Path]]:
    collisions = []
    for real_mask in sorted(data_root.rglob("*_mask.npy")):
        if real_mask.name.endswith("_pseudo_mask.npy"):
            continue
        case_id = real_mask.name.removesuffix("_mask.npy")
        pseudo_mask = real_mask.with_name(f"{case_id}_pseudo_mask.npy")
        weight_map = real_mask.with_name(f"{case_id}_weight.npy")
        if pseudo_mask.exists() or weight_map.exists():
            collisions.append((real_mask, pseudo_mask, weight_map))
    return collisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/processed/training_set"),
    )
    args = parser.parse_args()

    collisions = find_collisions(args.data_root)
    if not collisions:
        print(f"OK: no real/pseudo label collisions under {args.data_root}")
        return 0

    print(f"ERROR: found {len(collisions)} real/pseudo label collision(s):")
    for real_mask, pseudo_mask, weight_map in collisions:
        print(f"  real={real_mask}")
        if pseudo_mask.exists():
            print(f"  pseudo={pseudo_mask}")
        if weight_map.exists():
            print(f"  weight={weight_map}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
