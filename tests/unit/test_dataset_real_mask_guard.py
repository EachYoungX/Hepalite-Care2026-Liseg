from pathlib import Path

import numpy as np

from src.data.dataset import HybridDataset25D


def _write_case(root: Path, case_id: str, *, real: bool, pseudo: bool) -> None:
    vendor = root / "Vendor_A"
    vendor.mkdir(parents=True, exist_ok=True)
    data = np.ones((7, 4, 8, 8), dtype=np.float32)
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    mask[1, 2:6, 2:6] = 1
    np.save(vendor / f"{case_id}_data.npy", data)
    if real:
        np.save(vendor / f"{case_id}_mask.npy", mask)
    if pseudo:
        np.save(vendor / f"{case_id}_pseudo_mask.npy", mask)
        np.save(vendor / f"{case_id}_weight.npy", np.ones_like(mask, dtype=np.float32))


def test_real_case_never_falls_through_to_pseudo_pool(tmp_path):
    _write_case(tmp_path, "held-out-real", real=True, pseudo=True)
    _write_case(tmp_path, "unlabeled", real=False, pseudo=True)

    dataset = HybridDataset25D(
        train_dir=str(tmp_path),
        vendors=("Vendor_A",),
        case_ids=[],
        is_train=True,
    )

    pseudo_ids = {sample["case_name"] for sample in dataset.all_pseudo_samples}
    assert pseudo_ids == {"unlabeled"}


def test_real_mask_wins_when_case_is_in_training_fold(tmp_path):
    _write_case(tmp_path, "train-real", real=True, pseudo=True)

    dataset = HybridDataset25D(
        train_dir=str(tmp_path),
        vendors=("Vendor_A",),
        case_ids=["train-real"],
        is_train=True,
    )

    assert [sample["case_name"] for sample in dataset.all_real_samples] == [
        "train-real"
    ]
    assert dataset.all_pseudo_samples == []


def test_epoch_pseudo_selection_is_reproducible(tmp_path):
    for index in range(8):
        _write_case(tmp_path, f"pseudo-{index}", real=False, pseudo=True)

    first = HybridDataset25D(
        train_dir=str(tmp_path),
        vendors=("Vendor_A",),
        case_ids=[],
        is_train=True,
        sampling_seed=123,
    )
    second = HybridDataset25D(
        train_dir=str(tmp_path),
        vendors=("Vendor_A",),
        case_ids=[],
        is_train=True,
        sampling_seed=123,
    )

    first.set_epoch(10, pseudo_ratio=20)
    second.set_epoch(10, pseudo_ratio=20)
    assert first.selected_pseudo_case_names == second.selected_pseudo_case_names
