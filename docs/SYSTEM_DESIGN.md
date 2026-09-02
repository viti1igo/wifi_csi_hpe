---
title: "WiFi-CSI Human Pose Estimation Data and Experiment System"
status: "Proposed"
owner: "31050 ML project"
last_updated: "2026-09-02"
authors: "Project team"
related_docs: "PREPROCESSING.md; PIPELINE.md; TRAINING_SETTINGS.md; NOTEBOOK_FLOW.md"
scope: "Reproducible ingestion, preprocessing, training, evaluation, and reporting for Wi-Pose-to-WiMANS WiFi-CSI HPE."
---

# 1. Abstract

This design specifies the data and experiment system for a 2D WiFi-CSI human pose
estimation project. It transforms extracted Wi-Pose and WiMANS data into versioned,
auditable tensors, trains two controlled conditions, and produces assignment-facing
evidence in one reproducible notebook.

Wi-Pose provides CSI synchronized with 18-point AlphaPose labels and is the only
supervised pose source. WiMANS provides CSI and activity/environment metadata but
not native pose ground truth. It is therefore used only as unlabelled target-domain
CSI during self-supervised adaptation and as a frozen target-domain evaluation
domain. Any pose recovered from WiMANS video is explicitly a pseudo-label.

The system's central invariant is: **no raw file is changed, no target pose label is
used in training, and no monitor/final-test information influences preprocessing or
checkpoint selection.**

# 2. Goals and non-goals

| Goals | Non-goals |
|---|---|
| Trace every tensor to a raw file, participant, action, frame, and preprocessing version. | Claim physical 3D reconstruction from 2D AlphaPose supervision. |
| Produce leakage-safe 80/10/10 participant-grouped Wi-Pose experiments. | Treat WiMANS actions as human-pose ground truth. |
| Compare strict source-only transfer with label-free WiMANS adaptation. | Train on or tune against WiMANS pseudo-labels. |
| Preserve all 18 Wi-Pose AlphaPose points for inspection and derive a named 14-joint target. | Make the paper's 1–18 figure numbering interchangeable with AlphaPose indices. |
| Save reproducible manifests, parameters, logs, checkpoints, predictions, and figures. | Use the final test split repeatedly for model selection. |

# 3. Background and problem statement

The raw Wi-Pose data is organised as MATLAB files with CSI and `SkeletonPoints`.
The CSI storage layout is `[tx=3, rx=3, subcarrier=30, packet=5]`; the model needs
`[link=9, time=5, subcarrier=30]`. `SkeletonPoints` contains 18 x coordinates, 18
y coordinates, and 18 AlphaPose confidences. WiMANS recordings have a different
schema and domain conditions, so its loader/windowing path must be independent of
Wi-Pose's synchronized frame path.

The previous exploratory conversion saved useful `.npz` samples but did not retain
all 18 source joints or an explicit mapping contract. This design upgrades the data
contract so visualisation cannot confuse AlphaPose index 16 (right ear) with the
paper figure's joint 16 (a knee). It also prevents accidental domain leakage during
normalization and adaptation.

# 4. Proposed architecture

```text
immutable raw/extracted files
          |
          v
schema audit + inventory ----------> quality report / rejected-sample report
          |
          +--> Wi-Pose synchronizer --> CSI cleaner --> named 18->14 mapper
          |                                      |             |
          |                                      +--> train-only normalizer
          |
          +--> WiMANS selector/window builder --> optional SSL-only windows
                                                   |
                         versioned processed index + split manifest
                                                   |
       +------------------------+------------------+--------------------+
       v                        v                                       v
source-only trainer      SSL pretrainer -> fine-tuner             notebook/report
       |                        |                                       |
       +------------------------+------------------+--------------------+
                                                   v
                                      logs, checkpoints, metrics, figures
```

| Component | Responsibility | Durable outputs | Safe failure behaviour |
|---|---|---|---|
| Inventory builder | Enumerate inputs and metadata without mutation. | JSONL inventory, checksum report. | Fail file-level; record reason; never skip silently. |
| Wi-Pose processor | Decode CSI, synchronize frame/window, map joints. | Versioned pose shards/index. | Reject invalid shapes/anchors; preserve raw source. |
| WiMANS processor | Select eligible target records and create CSI windows. | SSL and evaluation indexes. | Keep excluded groups separate; no fabricated labels. |
| Split service | Deterministically assign complete groups to 8/1/1. | Split JSON and audit table. | Fail on overlap or fewer than three groups. |
| Normalizer | Fit source stats and apply frozen transform. | NPZ statistics + metadata. | Refuse non-training fit input. |
| Trainer | DDP training, per-epoch monitor evaluation, checkpointing. | CSV/JSONL, checkpoint, prediction files. | Stop safely on OOM/NaN; no final-test selection. |
| Notebook | Render evidence from stored artefacts. | Executed public notebook. | Clearly label unavailable/unfinished stages. |

# 5. Data lifecycle

1. Verify archive checksums and extract only to `data/extracted/`.
2. Discover files, parse metadata, and write an inventory record for every input.
3. Validate required keys, dimensions, finite values, and duplicate identifiers.
4. For Wi-Pose, select exactly five synchronized packets for each labelled frame,
   transpose CSI to `[9, 5, 30]`, and retain source packet indices.
5. Decode all 18 AlphaPose points. Map the named body points to the 14-joint
   target; retain both coordinate systems and confidence arrays.
6. Create a participant/recording grouped split before learning any data statistic.
7. Fit CSI cleaning thresholds and normalization using Wi-Pose training records
   only, then freeze them as versioned parameters.
8. Build disjoint WiMANS SSL-adaptation and evaluation indexes. No evaluation
   recording becomes part of SSL pretraining.
9. Serialize processed shards and validate their contract before training.
10. Train source-only and SSL-adapted conditions; select solely by Wi-Pose monitor
    performance; evaluate untouched Wi-Pose test and frozen WiMANS subsets.

# 6. Data contracts

## Processed supervised sample

| Field | Type | Required | Contract |
|---|---|---:|---|
| `sample_id` | string | Yes | Stable identifier from source filename. |
| `csi` | float32 `[9,5,30]` | Yes | Cleaned, normalized model input; finite. |
| `joints_14` | float32 `[14,2]` | Yes | Hip-centred, torso-normalized target. |
| `joints_18_pixel` | float32 `[18,2]` | Yes | Unmodified AlphaPose coordinate space for audit/plots. |
| `confidence_18` | float32 `[18]` | Yes | AlphaPose label confidence, never model confidence. |
| `joint_valid_14` | bool `[14]` | Yes | Declared confidence/anchor validity mask. |
| `participant`, `recording`, `action` | string | Yes | Grouping and reporting metadata. |
| `frame_index`, `csi_packet_start/end` | integer | Yes | Synchronization provenance. |
| `split`, `preprocessing_version` | string | Yes | Reproducibility boundary. |

## Unlabelled WiMANS sample

WiMANS output contains `sample_id`, `csi`, eligible-domain metadata, window timing,
split role (`ssl_train` or `zero_shot_test`), and preprocessing version. It must not
contain a training `joints` field. Optional video-pose output is stored separately
as `pseudo_labels`, with estimator version, timestamps, confidence, and audit flag.

# 7. Consistency, idempotency, and replay

The processed output version is immutable once accepted. A rerun with the same raw
checksums, code/config revision, and split manifest must create equivalent indexes;
existing shards may be reused only after contract validation. A changed mapping,
filter, normalization statistic, source checksum, or split creates a new version.

| Scenario | Expected behaviour |
|---|---|
| Duplicate input | Retain one canonical record; log source duplicates. |
| Partial processor failure | Produce a failure record and continue other independent samples. |
| Resume after interruption | Verify existing shard contract/checksum, then process only missing records. |
| Changed configuration | Refuse mixed-version outputs; create a new output directory/version. |
| Invalid skeleton anchor | Do not canonicalize; record a rejection reason. |
| GPU OOM during training | Reduce synchronised per-GPU batch size; retain logs and prior checkpoint. |

# 8. Security and privacy

Datasets remain on the controlled project storage and are excluded from Git history.
No credentials, Kaggle tokens, or private URLs are written into manifests/notebooks.
Logs contain dataset identifiers and aggregate metrics only; visual material should
be used under the datasets' applicable licences and assignment rules. Raw video or
identifiable imagery is not exported into public repositories.

# 9. Operational readiness

| Signal | Launch gate |
|---|---|
| Raw inventory coverage | Every discovered input is processed or has a reason code. |
| Tensor contract | 100% valid processed tensors have expected dtype, shape, and finite values. |
| Split integrity | Zero group overlap across train/monitor/test/adaptation/evaluation partitions. |
| Normalization integrity | Statistics reference Wi-Pose training split checksum only. |
| Skeleton integrity | Named mapping and anatomy-edge unit tests pass. |
| Synchronization | Random sequence audit and timing-offset report accepted. |
| Training readiness | One CPU/GPU smoke batch, mixed-precision check, and local log write pass. |

Preprocessing uses bounded CPU workers and disk-aware chunking. Training uses two
NVIDIA GPUs through DDP only after a safe synchronized batch probe leaves a VRAM
reserve. All long runs must write periodic progress and be resumable from a saved
checkpoint or manifest.

# 10. Alternatives considered

| Alternative | Benefit | Decision |
|---|---|---|
| Treat WiMANS activity as pose labels | Simple cross-dataset score. | Rejected: activity is not a skeleton target. |
| Use paper figure indices directly | Visually familiar. | Rejected: it conflicts with AlphaPose source ordering. |
| Random frame split | Higher nominal sample balance. | Rejected: participant/recording leakage invalidates generalization claims. |
| Fit normalizers per dataset | May improve target scores. | Rejected for strict zero-shot: leaks target distribution. |
| Train one mixed-domain supervised model | Simple implementation. | Rejected: no WiMANS native pose labels and weak comparison. |

# 11. Open questions

1. Confirm the exact Wi-Pose CSI-to-video timestamp rule from the dataset/paper and
   quantify offset for randomly selected recordings.
2. Confirm which WiMANS recording groups are eligible for the project's final
   single-person, 5 GHz target protocol.
3. Decide whether final WiMANS reporting is representation/action-proxy only or
   includes carefully labelled AlphaPose pseudo-label agreement.
4. Decide after schema audit whether CountFi's Demo3 filter parameters transfer
   unchanged to Wi-Pose's sampling rate.

# 12. Decision and next steps

Adopt the versioned preprocessing system before any model training. First deliver a
corrected named skeleton contract and schema report, then lock the grouped split and
source-only normalization. Only then run training and cross-domain comparisons.

| Milestone | Deliverable | Exit criteria |
|---|---|---|
| M1 | Schema audit and corrected skeleton contract. | 18-point mapping and visual edge tests pass. |
| M2 | Versioned Wi-Pose processed data and 8/1/1 split. | No leakage; source-only normalization is frozen. |
| M3 | WiMANS SSL/evaluation window indexes. | Disjoint target partitions and documented eligibility. |
| M4 | Two-condition training and notebook evidence. | Monitor-selected checkpoints and untouched final evaluations. |
