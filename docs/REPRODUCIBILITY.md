# Reproducibility notes and paper tables

This page consolidates the tables used in the official paper, their experimental provenance, the ablation workflow, and local path conventions. It accompanies the [project README](../README.md) and the [official published paper](https://papers.miccai.org/miccai-2026-sat/CARE_012.html).

## Paper tables

### Table 1: Data use and evaluation boundaries

| Item | Setting |
| --- | --- |
| Labeled GED4 cases | 30 |
| Unlabeled training cases | 430 |
| Transductive validation use | 60 images; pseudo-label weight 0.35 |
| Input modality and context | GED4 only; five adjacent slices |
| OOF split | Five folds, random seed 42; held-out-fold labels used for model selection |
| Pseudo-label candidate pool | Foreground-slice proportion in (0.05, 0.8); at most 20 cases per epoch |
| Controlled OOF | Real-label training only, including within-fold model selection |
| Internal semi-supervised evaluation | Shared teachers; transductive use of validation images |
| Hidden out-of-distribution test | Vendor C; not used for training or tuning |
| Pseudo-label source | Probability ensemble of five teacher models |
| Final inference | Five-fold student ensemble and deterministic 3D volumetric refinement |

### Table 2: Ablation of 3D volumetric refinement for the selected architecture

Real-label-only OOF evaluation, `n=30`.

| Stage | Dice | HD95 (mm) | Interpretation |
| --- | ---: | ---: | --- |
| Raw prediction, threshold 0.350 | 0.9083 | 23.67 | Mask without volumetric constraints |
| + In-plane opening | 0.9097 | 22.68 | In-plane cleanup |
| + Largest 3D connected component | 0.9232 | 11.37 | Retains the main connected region |
| + 3D hole filling | 0.9241 | 11.01 | Fills interior holes |
| + Axial-chain filtering | 0.9242 | 11.01 | Axial continuity |
| + End-slice rescue | 0.9242 | 11.01 | No further change in the mean |

### Table 3: Architectural ablation with real labels only

`n=30`. All variants use threshold 0.350 and the complete 3D volumetric refinement.

| Experiment | Parameters | Dice | HD95 (mm) |
| --- | ---: | ---: | ---: |
| 1 slice + CoordConv + SE | 516,916 | 0.7494 | 45.15 |
| 3 slices + CoordConv + SE | 517,268 | 0.9177 | 11.79 |
| 5 slices + CoordConv + SE | 517,620 | 0.9171 | 11.73 |
| 5 slices + SE, without CoordConv | 517,268 | 0.8965 | 17.53 |
| 5 slices + CoordConv, without SE | 504,052 | 0.9242 | 11.01 |

### Table 4: Local and official evidence under different evaluation boundaries

| Evaluation | Evidence boundary | Dice | HD95 (mm) |
| --- | --- | ---: | ---: |
| Real-label-only OOF | Held-out fold used for model selection; no pseudo-labels | 0.9242 | 11.01 |
| Semi-supervised cross-validation | Shared teachers; transductive setting | 0.9243 | 9.08 |
| Docker in-distribution test (A/B1/B2) | Organizer evaluation | 0.9140 | 35.65 |
| Docker out-of-distribution test (C) | Images and labels both hidden | 0.9247 | 21.76 |

## Provenance of the paper values

Table 1 describes data use and evaluation settings rather than scores from a single script. Table 2 starts from five-fold OOF probability maps for the selected `real_5slice_no_se` architecture trained on 30 real-label cases. `experiments/ablation_liseg/evaluate_strict_nose_topology.py` applies the 3D post-processing stages at threshold 0.350 and reports case-level mean Dice and HD95. Table 3 uses `experiments/ablation_liseg/evaluate_structure_ablation.py` to compare five real-label-only architectures under the same threshold and complete 3D post-processing; it reports their 30-case OOF means and parameter counts.

The real-label-only OOF row of Table 4 is the final stage of Table 2. Its internal semi-supervised row comes from the NoSE evaluation in `experiments/ablation_liseg/evaluate_nose_semisup.py` at threshold 0.375. The two Docker rows are official organizer evaluation results, not local OOF calculations. The paper rounds mean Dice to four decimal places and mean HD95 to two, for example `0.924175… → 0.9242` and `9.082017… → 9.08 mm`. Thus each paper value comes from its corresponding experiment or official evaluation.

## Ablation workflow and scope

`experiments/ablation_liseg/run_ablation.py` is a historical OOF comparison of three training configurations:

- `teacher_supervised`: five-fold teachers trained on labeled CARE cases only.
- `student_topology_pseudo`: students trained with topology-refined pseudo-labels from the unlabeled training set.
- `student_transductive`: students additionally trained with validation-set pseudo-labels weighted by 0.35.

Each labeled case is evaluated with its held-out-fold checkpoint only. OOF evaluation does not ensemble all five folds, because the other four models saw that case during training. The student comparison reports these cumulative post-processing stages:

1. `raw_050`
2. `raw_035`
3. `opening`
4. `cca`
5. `fill_holes`
6. `z_chain`
7. `final_topology`

This OOF comparison uses neither validation-case identifiers nor manually selected axial ranges. Six fixed validation-case Z starts in `generate_pseudo_topo.py` belong to the paper's pseudo-label generation workflow; they do not affect these OOF metrics or the final generic student inference entry point. The historical online candidate packer is kept in a private archive, outside this code repository.

Run from the repository root:

```bash
python experiments/ablation_liseg/run_ablation.py
```

To regenerate probability caches:

```bash
python experiments/ablation_liseg/run_ablation.py --rebuild-cache
```

Physical-distance metrics use the preprocessed-grid spacing `(z, y, x) = (7.0, 1.5, 1.5) mm`. Official online results are reported separately from these OOF results.

## Paths and generated artifacts

The paths below are relative to the `hepalite-care2026-liseg/` repository root.

| Purpose | Path or naming convention | Notes |
| --- | --- | --- |
| Authorized raw images | `dataset/raw/<training_set or validation_set>/Vendor_*/<case_id>/GED4.nii.gz` | Request data from the CARE organizers; not Git content |
| Labeled training masks | `dataset/raw/training_set/Vendor_*/<case_id>/mask_GED4.nii.gz` | Present only for authorized labeled cases |
| Preprocessed volumes | `dataset/processed/<set>/Vendor_*/<case_id>_data.npy` | Grid `(32, 256, 256)` with ZYX array order |
| Preprocessed real masks | `<case_id>_mask.npy` in the same directory | Unlabeled cases must not have this file |
| Pseudo-label and weight maps | `<case_id>_pseudo_mask.npy` and `<case_id>_weight.npy` in the same directory | Derived from authorized data and teachers; excluded from Git |
| Five-fold model checkpoints | `artifacts/training_runs/<experiment>/fold_*/checkpoints/` | Teacher and student runs write to separate local experiment directories |
| Final student inference output | `infer.py --output-dir <directory>` writes `<case_id>_mask.npy` | NoSE five-fold ensemble at threshold 0.375; preprocessed grid |
| Original-grid NIfTI restoration and submission packaging | `PACK_CONFIG` near the top of `reconstruct_and_package.py` | Fixed path configuration; separate from `infer.py` |
| Ablation metrics and compact results | `experiments/ablation_liseg/results/` | Historical local results; ignored by Git; paper tables are above |
| Large experiment caches and logs | `artifacts/experiments/ablation_liseg/` | Probability volumes, controlled training files, and diagnostic logs kept locally |
| Paper figures and author-layout resources | Outer `publication/camera_ready/CARE_12_source/figures/` | Outside the code repository; redistribution rights require separate review |
| Internal review and archive records | Outer `internal_docs/` | Outside the public code repository |

Preprocessing, teacher training, pseudo-label generation, and student training primarily use path constants near the top of their scripts. Preprocessing and restoration to the original NIfTI grid follow CARE 2026 LiSeg data organization, modality, and grid conventions; they serve as a reference for reproducing this dataset-specific workflow. `infer.py` accepts directory arguments, but its inputs must still follow this project's preprocessing convention. Original-grid restoration is performed separately. This repository does not implement a general adapter for arbitrary NIfTI data.
