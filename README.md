# HepaLite: CARE 2026 LiSeg Liver Segmentation

This repository is the authors' implementation of *Topology-Guided Lightweight 2.5D Liver Segmentation for Limited-Annotation Multi-Center Fibrosis MRI*.

This project addresses the CARE-Liver track of CARE 2026, which studies multi-center liver image segmentation and liver fibrosis analysis. Our work focuses on the GED4 liver segmentation task and explores lightweight 2.5D semi-supervised learning with limited manual annotations and differences between imaging vendors.

For citation, use the [official MICCAI Society paper page](https://papers.miccai.org/miccai-2026-sat/CARE_012.html), which provides the open-access PDF and the current BibTeX record.

## Method overview

The method takes single-modality GED4 hepatobiliary-phase MRI as input. It combines five adjacent axial slices into a 2.5D input and adds CoordConv spatial-coordinate channels. Training consists of five-fold teacher training, confidence-weighted pseudo-labeling, and five-fold student training. The final student omits SE, uses four-level deep supervision, and has 504,052 trainable parameters per model. At inference, voxel probabilities from the five students are averaged, a deterministic 3D volumetric post-processing procedure is applied at threshold 0.375, and the result is restored to the original NIfTI grid.

The teacher uses SE. The training data comprise 30 GED4 cases with manual liver masks, 430 unlabeled training cases, and 60 validation images. Using validation images for pseudo-label training is a transductive setting. The real-label out-of-fold (OOF) experiment, internal semi-supervised cross-validation, and official hidden tests have different evidence boundaries.

## Results reported in the paper

| Evaluation | Dice | HD95 (mm) | Interpretation |
| --- | ---: | ---: | --- |
| Real-label OOF, selected architecture with 3D post-processing | 0.9242 | 11.01 | Held-out-fold labels were used for model selection; no pseudo-label training |
| Internal semi-supervised cross-validation | 0.9243 | 9.08 | Shared teacher pseudo-labels; not a fully isolated OOF evaluation |
| Official hidden test, A/B1/B2 | 0.9140 | 35.65 | Organizer evaluation |
| Official hidden test, Vendor C | 0.9247 | 21.76 | Both images and labels hidden from participants |

The [official published paper](https://papers.miccai.org/miccai-2026-sat/CARE_012.html) gives the full experimental definitions, ablations, and limitations. The [reproducibility notes](docs/REPRODUCIBILITY.md) consolidate the paper's tables, result provenance, paths, and experiment workflow.

## Code layout

```text
src/preprocess/                 GED4 preprocessing
src/data/                       2.5D dataset and augmentation
src/models/                     Teacher and student models
src/losses/                     Training losses and metrics
src/engine/                     Training loops
src/postprocess/                3D volumetric post-processing
train_teacher.py                Five-fold teacher training
generate_pseudo_topo.py         Teacher pseudo-label and weight-map generation
train_student.py                Five-fold student training
experiments/ablation_liseg/     Paper-related experiments and analyses
tests/                          Unit tests and data-dependent checks
```

Input, training, prediction, and evaluation artifact locations are collected in the [reproducibility notes](docs/REPRODUCIBILITY.md).

## Data and model weights

This repository does not provide CARE MRI, manual masks, preprocessed arrays, pseudo-labels, predicted volumes, or trained checkpoints. Obtain data access through the [official CARE website](https://zmic.org.cn/care_2026/) and comply with the [Data Access Agreement](https://zmic.org.cn/flask/download/Agreement_CARE_Username.pdf/).

The default paths in the code assume that scripts are run from the repository root. Dataset layout and output locations are listed in the [reproducibility notes](docs/REPRODUCIBILITY.md).

## Environment and entry points

Dependencies are listed in `requirements_pip.txt`; test dependencies are in `requirements-dev.txt`. Match the PyTorch/CUDA build to the target hardware.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_pip.txt
pip install -r requirements-dev.txt
```

Preprocessing and training scripts primarily use path and parameter constants near the top of each file. Final student inference also has a command-line interface that accepts user directories. After obtaining authorized data and checking these settings, run the stages in this order:

```bash
python src/preprocess/preprocess.py
python train_teacher.py
python generate_pseudo_topo.py
python train_student.py --no-se
```

`generate_pseudo_topo.py` defaults to `validation_set` only. To generate the pseudo-labels for all unlabeled cases used in the paper's training workflow, review and set `PARAMS["target_sets"]`. Final prediction also requires the trained weights, which are not provided here.

`infer.py` implements the paper's final NoSE student ensemble with threshold 0.375. It accepts preprocessed volumes, five-fold checkpoints, and an output directory. It writes `.npy` masks on the preprocessed grid; restoring the original NIfTI grid is a separate step described in the [reproducibility notes](docs/REPRODUCIBILITY.md).

Code-level unit tests are in `tests/unit/`; checks requiring authorized data or a submission artifact are in `tests/smoke/`. With the test dependencies installed, run:

```bash
pytest tests/unit
```

## Evaluation boundaries and reproducibility

- The architectural ablation uses five-fold OOF predictions from real-label training. Labels of the held-out fold informed learning-rate adjustment and checkpoint selection.
- Shared five-fold teachers generated pseudo-labels for final student training. Internal semi-supervised cross-validation is therefore not a fully label-isolated independent test.
- Vendor C images and labels were completely hidden from participants and were not used for local training or selection.
- Per-epoch pseudo-case sampling state from the historical training runs was not fully preserved; retraining with the same configuration is not guaranteed to reproduce historical checkpoints bit for bit.
- Final Docker system metrics come from the organizer evaluation. This repository contains neither trained weights nor official hidden-test data.

## Paper and license

For citation, use the authors, title, conference information, and BibTeX on the [official paper page](https://papers.miccai.org/miccai-2026-sat/CARE_012.html). DOI and page numbers should be added only when the final publication record provides them. If a post-publication revision of Figure 1 is displayed separately, its difference from the figure in the official paper should be identified explicitly.

The source code is licensed under the Apache License 2.0 in the repository-root `LICENSE`. This code license does not grant redistribution rights for CARE data, typeset paper PDFs, MRI images, or third-party assets. Any author-layout figure containing CARE MRI slices requires a separate review against the data agreement before public release.

## Acknowledgments

We thank the CARE 2026 organizers for providing the evaluation platform.
