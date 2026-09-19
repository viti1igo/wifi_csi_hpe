---
title: "Cross-Dataset Human Pose Estimation Using WiFi Channel State Information"
status: "Implemented"
last_updated: "2026-09-19"
scope: "Wi-Pose supervised learning, WiMANS label-free encoder adaptation, and frozen cross-dataset evaluation"
---

# System Design

## Objective

The system estimates a canonical two-dimensional BODY-14 pose from a five-packet WiFi CSI window. It tests whether masked reconstruction on unlabelled target-domain CSI improves transfer without using target-domain pose labels.

Wi-Pose is the labelled source domain. WiMANS is the target domain. Condition A trains only on Wi-Pose. Condition B pretrains an encoder on balanced unlabelled Wi-Pose and permitted WiMANS CSI, then fine-tunes on exactly the same labelled Wi-Pose data as Condition A.

## Architecture

```text
                         +-----------------------------+
Wi-Pose labelled CSI --->| shared CSI preprocessing    |---> [9,5,30]
                         +-----------------------------+          |
                                                                    v
                                                         +------------------+
                                                         | PoseCNN encoder  |
                                                         | 32/64/128/192   |
                                                         +------------------+
                                                                    |
                                                                    v
                                                         canonical BODY-14

WiMANS permitted CSI -> masked autoencoder -> pretrained encoder -> Condition B
WiMANS empty-room CSI ---------------------------------------------> evaluation
```

PoseCNN applies four residual two-dimensional convolution blocks across the time and subcarrier plane. The first block keeps the input resolution; later blocks downsample only the subcarrier axis. Adaptive pooling and an MLP output 28 coordinates.

## Core invariants

1. Raw and extracted datasets remain unchanged.
2. Participant groups do not cross Wi-Pose train, monitor and test partitions.
3. Wi-Pose training data alone defines normalization statistics.
4. WiMANS empty-room recordings never enter pretraining, fine-tuning or checkpoint selection.
5. WiMANS pose pseudo-references never enter an optimizer step.
6. Both conditions share the supervised architecture, loss, labelled data and selection rule.
7. Saved checksums bind checkpoints to their split, normalizer and configuration.

## Data contracts

### Supervised Wi-Pose sample

| Field | Contract |
|---|---|
| `sample_id` | Stable source identifier |
| `csi` | Five-packet CSI, reshaped to `float32 [9,5,30]` by the loader |
| `joints` | Canonical `float32 [14,2]` target |
| `joints_pixel` | BODY-14 pixel coordinates |
| `joints_18_pixel` | Complete source AlphaPose pose |
| `confidence` | BODY-14 AlphaPose confidence |
| `participant`, `activity`, `partition` | Grouping and analysis metadata |

### WiMANS sample

The manifest stores a recording identifier, CSI path, environment, activity, location and partition. SSL samples have no pose field. Evaluation pseudo-references live under a separate provenance-controlled namespace.

## Model conditions

| Design choice | Condition A | Condition B |
|---|---|---|
| Encoder start | Random initialization | Masked-CSI pretrained initialization |
| Supervised source | Wi-Pose | Wi-Pose |
| Target CSI before inference | None | Classroom and meeting-room recordings |
| Target pose supervision | None | None |
| Selected by | Wi-Pose monitor NME | Wi-Pose monitor NME |

This design isolates encoder initialization as the controlled experimental difference.

## Loss design

The supervised loss combines coordinate accuracy with skeletal structure:

```text
L_pose = L_coordinate + 0.25 L_bone + 0.05 L_symmetry
```

Smooth L1 coordinate loss uses AlphaPose confidence as a weight. Bone-vector loss compares connected joint offsets. Symmetry loss discourages large left-right bone-length differences. The structural terms regularize implausible poses without replacing coordinate supervision.

Condition B pretraining minimizes masked reconstruction MSE. A shared encoder maps both datasets because preprocessing converts their stored layouts to the same `[9,5,30]` interface.

## Evaluation design

Wi-Pose provides native held-out target labels. Metrics include NME or 2D MPJPE, PCK at three thresholds, normalized PCK-AUC, bone-vector error, symmetry gap, per-joint error and a nearest-joint confusion matrix.

WiMANS evaluation compares predictions with AlphaPose pseudo-references. It reports the same aggregate pose metrics, valid-frame coverage, confidence sensitivity, alignment sensitivity and recording-grouped bootstrap intervals. Qualitative figures and MP4 videos show identical frames for both conditions.

## Final evidence

| Dataset | Condition A NME | Condition B NME | Interpretation |
|---|---:|---:|---|
| Wi-Pose test | 0.206182 | 0.202268 | Small improvement after SSL initialization |
| WiMANS pseudo-reference | 0.353495 | 0.336049 | Better broad cross-domain agreement |

Condition B improved Wi-Pose PCK at every reported threshold. On WiMANS, it improved NME, PCK@0.20, PCK-AUC, bone error and symmetry error, while Condition A retained higher PCK@0.05 and PCK@0.10. The target-domain result supports fewer large errors, not uniformly tighter joint localization.

## Limitations

- WiMANS has no native pose ground truth.
- Uniform CSI-to-video alignment introduces timing uncertainty.
- A five-packet window offers little temporal context.
- Canonical targets remove global position and body scale, so predictions cannot recover either quantity without extra information.
- AlphaPose uncertainty affects Wi-Pose supervision and WiMANS pseudo-reference evaluation.
- Results cover the selected datasets, partitions and room conditions; they do not establish deployment performance in arbitrary environments.

## Repository outputs

- Executed notebook: `notebooks/24681189_notebook.ipynb`
- Shared implementation: `shared/`
- Condition-specific configuration, logs and selected checkpoints: `conditions/`
- Final metric tables and media: `results/evaluation/`
- Data and split contracts: `data/manifests/` and `data/processed/`

Git excludes raw data and most generated results. Selected evaluation videos are committed separately for GitHub access because GitHub may suppress HTML5 players embedded in notebook output.
