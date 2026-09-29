"""Data-free path checks for the public inference entry point."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infer import find_cases, find_checkpoints, main


class InferencePathTests(unittest.TestCase):
    def test_case_discovery_accepts_nested_vendor_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Vendor_A").mkdir()
            (root / "Vendor_A" / "case1_data.npy").touch()
            self.assertEqual([path.name for path in find_cases(root)], ["case1_data.npy"])

    def test_duplicate_case_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for vendor in ("Vendor_A", "Vendor_B"):
                (root / vendor).mkdir()
                (root / vendor / "case1_data.npy").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate case names"):
                find_cases(root)

    def test_five_checkpoints_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold in range(5):
                checkpoint = root / f"fold_{fold}" / "checkpoints" / "best_model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
            self.assertEqual(len(find_checkpoints(root)), 5)

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "case1_data.npy").touch()
            checkpoints_dir = root / "checkpoints"
            for fold in range(5):
                checkpoint = checkpoints_dir / f"fold_{fold}" / "checkpoints" / "best_model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "case1_mask.npy").touch()
            arguments = ["infer.py", "--input-dir", str(input_dir),
                         "--checkpoints-dir", str(checkpoints_dir),
                         "--output-dir", str(output_dir)]
            with patch("sys.argv", arguments), self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
