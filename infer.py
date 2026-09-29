"""Run the paper's five-student NoSE inference on preprocessed GED4 volumes.

Inputs must use the 32 x 256 x 256 grid produced by src/preprocess/preprocess.py.
The output is a processed-grid NumPy mask, not a restored NIfTI image.
"""

import argparse
from pathlib import Path


def find_cases(input_dir: Path) -> list[Path]:
    cases = sorted(input_dir.rglob("*_data.npy"))
    if not cases:
        raise ValueError(f"No *_data.npy volumes found under {input_dir}")

    names = [path.name.removesuffix("_data.npy") for path in cases]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate case names would overwrite output masks")
    return cases


def find_checkpoints(checkpoints_dir: Path) -> list[Path]:
    paths = sorted(checkpoints_dir.glob("fold_*/checkpoints/best_model.pth"))
    if len(paths) != 5:
        raise ValueError(f"Expected five fold checkpoints; found {len(paths)}")
    return paths


def load_models(checkpoints: list[Path], device: str):
    import torch

    from src.models.models import HepaLiteStudent25D

    models = []
    for checkpoint in checkpoints:
        model = HepaLiteStudent25D(
            in_channels=5,
            out_classes=1,
            init_features=16,
            deep_supervision=True,
            use_se=False,
        )
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        models.append(model.to(device).eval())
    return models


def predict_case(case_path: Path, models, device: str):
    import numpy as np
    import torch

    from src.postprocess.topology import postprocess_prob_volume

    image = np.load(case_path, allow_pickle=False)
    if image.ndim == 4:
        image = image[6] if image.shape[0] > 6 else image[0]
    if image.shape != (32, 256, 256):
        raise ValueError(
            f"Expected a (32, 256, 256) preprocessed volume in {case_path}; got {image.shape}"
        )
    if not np.isfinite(image).all():
        raise ValueError(f"Volume contains NaN or Inf: {case_path}")

    depth = image.shape[0]
    probability = np.zeros(image.shape, dtype=np.float32)
    with torch.no_grad():
        for z in range(depth):
            indices = [max(0, z - 2), max(0, z - 1), z,
                       min(depth - 1, z + 1), min(depth - 1, z + 2)]
            tensor = torch.from_numpy(image[indices].copy()).unsqueeze(0).float().to(device)
            fold_probabilities = [
                torch.sigmoid(model(tensor))[0, 0].cpu().numpy() for model in models
            ]
            probability[z] = np.mean(fold_probabilities, axis=0)

    return postprocess_prob_volume(
        probability,
        threshold=0.375,
        opening=True,
        fill_holes=True,
        min_size_filter=50,
        min_overlap_iou=0.03,
        centroid_jump_limit=40.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing preprocessed *_data.npy volumes")
    parser.add_argument("--checkpoints-dir", type=Path, required=True,
                        help="Five-fold NoSE experiment directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for processed-grid *_mask.npy outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output masks")
    args = parser.parse_args()

    cases = find_cases(args.input_dir)
    checkpoints = find_checkpoints(args.checkpoints_dir)
    outputs = [
        args.output_dir / f"{path.name.removesuffix('_data.npy')}_mask.npy"
        for path in cases
    ]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("An output mask already exists; use --overwrite to replace it")

    import numpy as np
    import torch
    from tqdm import tqdm

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    models = load_models(checkpoints, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case_path in tqdm(cases, desc="NoSE inference"):
        case_name = case_path.name.removesuffix("_data.npy")
        mask = predict_case(case_path, models, device)
        np.save(args.output_dir / f"{case_name}_mask.npy", mask.astype(np.uint8))
    print(f"Saved {len(cases)} processed-grid masks to {args.output_dir}")


if __name__ == "__main__":
    main()
