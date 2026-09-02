# Preprocessing Pipeline

## Purpose

This pipeline converts the extracted Wi-Pose and WiMANS files into reproducible,
model-ready samples for WiFi-CSI human pose estimation (HPE). It is designed to
prevent participant leakage, preserve the original data, keep CSI and pose frames
synchronized, and make every transformation auditable from the final notebook.

The supervised task is:

```text
CSI window -> normalized CSI tensor -> pose estimator -> 2D body joints
```

Wi-Pose is the labelled source domain. WiMANS is the unlabelled target domain used
for self-supervised adaptation and frozen cross-dataset testing. WiMANS does not
provide native skeleton ground truth; a skeleton extracted from its video would be
a video-model pseudo-label, not measured ground truth.

## Current state

| Item | State | Meaning |
|---|---|---|
| Raw archives | Complete | Preserved under `data/raw/` |
| Extraction | Complete | Wi-Pose and WiMANS are under `data/extracted/` |
| File inventories | Complete | Initial JSON/JSONL manifests exist |
| Exploratory Wi-Pose NPZ conversion | Provisional | Must be regenerated after the 18-joint mapping and synchronization audits below |
| Final preprocessing | Not run | No processed artefact is approved for training yet |
| Participant-grouped split | Not run | Split membership must be created before fitting normalization statistics |

The files currently in `data/processed/wipose/` are exploratory artefacts. They are
useful for notebook visualisation but must not be treated as final training data.
The current converter truncates the first 14 of 18 AlphaPose joints. The final
converter must instead use an explicit named joint mapping and verify every edge.

## Non-negotiable data rules

1. Never modify files in `data/raw/` or `data/extracted/`.
2. Never infer joint meaning from its numerical position alone.
3. Never fit a normalizer, filter threshold, imputer, or calibration parameter on
   monitor, final-test, or zero-shot target samples.
4. Split by participant/recording group before fitting learned preprocessing.
5. Store the split membership and preprocessing parameters with checksums.
6. Keep raw pixel joints, canonical joints, confidence, and validity masks. Do not
   silently discard low-confidence joints.
7. Do not call AlphaPose output physical ground truth. It is the Wi-Pose paper's
   video-derived supervisory label.
8. Do not call CSI-Former output ground truth. It is a model prediction.
9. Preserve enough metadata to trace every tensor back to its source file, action,
   participant, frame, CSI packet range, and preprocessing version.

## Expected artefacts

```text
data/
├── extracted/                         # immutable extracted datasets
├── manifests/
│   ├── wipose_inventory.jsonl         # one source record per file/frame
│   ├── wimans_inventory.jsonl
│   ├── split_v1.json                   # group membership and seed
│   ├── preprocessing_v1.json           # parameters and code/config checksums
│   └── quality_report_v1.json
└── processed/v1/
    ├── wipose/{train,monitor,test}/
    ├── wimans/{ssl_train,zero_shot_test}/
    ├── normalization_wipose_train.npz
    └── index.jsonl
```

Each processed index record should contain at least:

```text
sample_id, dataset, source_path, participant, recording, action,
frame_index, csi_packet_start, csi_packet_end, csi_shape, pose_shape,
pose_source, split, valid_joint_count, preprocessing_version, output_path
```

## Stage 0 — Reproducible environment and configuration

- Run every preprocessing command and notebook cell with the `ML_31005` conda
  environment.
- Record Python, NumPy, SciPy, h5py, PyTorch, CUDA, and package versions.
- Fix all random seeds to `31050` unless an experiment explicitly overrides it.
- Save the resolved YAML configuration beside the processed data.
- Record the Git commit and SHA-256 checksums of the raw archives and generated
  manifests.

Preprocessing is primarily CPU, memory, and storage I/O work. It should not reserve
both GPUs merely to increase utilization. The two-GPU/full-VRAM policy applies to
training; preprocessing should use bounded parallel readers and avoid exhausting
RAM or flooding shared storage.

## Stage 1 — Raw schema and integrity audit

### Wi-Pose

For every MATLAB file:

- verify that the CSI and `SkeletonPoints` keys exist;
- record MATLAB/HDF5 storage type and the raw axis order;
- verify finite values, dtype, dimensions, packet count, subcarrier count, and
  transmitter/receiver dimensions;
- decode filename metadata into action, participant, and frame number;
- verify that the decoded frame agrees with the sample contents and directory;
- inspect the complete 18-joint vector as 18 x-coordinates, 18 y-coordinates, and
  18 confidence values;
- count missing, zero, non-finite, and out-of-frame joints;
- check for duplicate sample IDs and duplicate source contents.

### WiMANS

For every CSI recording and annotation file:

- record room, receiver/transmitter setting, band, subject count, activity labels,
  recording ID, duration, and raw tensor shape when available;
- confirm whether values are complex CSI, amplitude/phase, or already transformed;
- record timestamps or sample rates required for windowing;
- check missing recordings, inconsistent annotation lengths, non-finite values,
  saturated channels, and duplicates;
- identify the exact eligible subset for single-person 5 GHz experiments;
- separate empty-room, multi-person, and other excluded samples without deleting
  them.

The audit produces tables and plots in the notebook but does not yet transform the
data.

## Stage 2 — Define the learning unit and synchronize modalities

The model input is one short CSI window associated with one pose frame. The base
configuration expects five packets, 30 subcarriers, and nine Tx-Rx links:

```text
raw logical layout:   [time=5, subcarrier=30, tx=3, rx=3]
model layout:         [link=9, time=5, subcarrier=30]
```

For each Wi-Pose pose frame:

1. determine the corresponding CSI timestamp or packet centre using the dataset's
   documented synchronization rule;
2. select the five-packet window around that frame;
3. record the exact packet indices in the manifest;
4. reject or pad boundary windows only according to a declared policy;
5. plot random synchronized examples: CSI traces/heatmap beside the complete
   18-joint AlphaPose frame;
6. audit motion continuity across the full recording. A frame labelled `run` is
   not expected to look recognizably like running in isolation, so action validity
   must also be checked as a sequence.

For WiMANS, fixed-duration windows are created from the recording timestamps. They
are not paired with a skeleton during self-supervised pretraining. If WiMANS video
is later processed with AlphaPose, those predictions must be timestamp-aligned and
stored in a separate `pseudo_labels/` namespace.

No interpolation or nearest-neighbour alignment should be accepted until the
offset distribution is reported. The manifest must store alignment error in
milliseconds where timestamps permit it.

## Stage 3 — CSI calibration and signal cleaning

The first implementation uses amplitude as the common representation:

\[
A_{t,s,l}=|H_{t,s,l}|=\sqrt{\Re(H)^2+\Im(H)^2}.
\]

Processing order:

1. Decode complex CSI without discarding its imaginary component prematurely.
2. Reorder axes explicitly to `[link, time, subcarrier]`.
3. Detect invalid packets and replace/drop them using a recorded validity mask.
4. Remove packet-wise outliers with a robust rule fitted on source-training data,
   such as a median/MAD threshold. Report the affected percentage.
5. Suppress static offsets. Candidate operations include per-link/per-subcarrier
   centring or a calibrated reference subtraction; select one only after the raw
   distributions are inspected.
6. Apply CountFi-style temporal denoising consistently with the earlier Demo3
   implementation: Hampel/outlier filtering followed by low-pass or wavelet-based
   smoothing where the sampling rate supports it. Copy the parameters exactly and
   document any change needed for Wi-Pose's different sampling rate.
7. Optionally compute sanitized phase only as an ablation. Raw phase should not be
   used until linear phase error and per-packet offsets are corrected.
8. Preserve an unnormalized cleaned amplitude tensor for diagnostics.

Important: a five-packet input is extremely short. Filters with a window longer
than five packets must be applied to the continuous recording before extracting
five-packet model windows, not independently to each five-packet sample.

## Stage 4 — Skeleton decoding and correction

Wi-Pose stores 18 AlphaPose/OpenPose-style points. The final pipeline keeps all 18
for visualisation and creates an explicit training target rather than taking
`points[:14]`.

Required steps:

1. Confirm the 18-joint names and handedness from the dataset code/paper.
2. Create a source-index-to-name table and test it on standing, sitting, bending,
   walking, and running sequences.
3. Define the training topology by joint names. The proposed 14-joint body target
   is nose, neck, right shoulder/elbow/wrist, left shoulder/elbow/wrist, right
   hip/knee/ankle, and left hip/knee/ankle.
4. Map the named 18-joint source indices to those 14 target names. Eye/ear points
   may be excluded from the 14-joint loss but remain in the stored 18-joint data.
5. Define edges only after mapping. Validate that every edge connects anatomical
   neighbours; there must be no ankle-to-head or leg-to-head connection.
6. Retain per-joint AlphaPose confidence and create a validity mask using a declared
   threshold. Confidence is used as a loss/evaluation weight, not presented as
   CSI-Former confidence.
7. Preserve the original pixel coordinates before any rotation or normalization.

The paper's rotation diagram describes a coordinate transformation for pose
construction; rotating a Matplotlib camera is only a visualisation choice. A 2D
pixel skeleton cannot become true 3D merely by changing the plot angle.

## Stage 5 — Pose canonicalization

The model should learn body configuration rather than absolute image location.
For each valid skeleton:

1. calculate the hip centre from the left and right hip;
2. translate all joints so that the hip centre is the origin;
3. calculate torso length from hip centre to neck;
4. divide coordinates by torso length;
5. store the translation and scale so predictions can be converted back for plots;
6. keep a visibility mask for joints with insufficient confidence;
7. reject a sample only when its required anchor joints are invalid or scale is
   non-finite/near zero.

For joint coordinate \(p_j\), hip centre \(c\), and torso scale \(s\):

\[
\tilde p_j=\frac{p_j-c}{\max(s,\epsilon)}.
\]

Mirroring is optional augmentation, not preprocessing. If used, swap left/right
joint identities as well as negating/flipping coordinates.

## Stage 6 — Grouped 80/10/10 split before learned normalization

Create a deterministic participant-grouped split:

- 80% training;
- 10% monitor/validation;
- 10% untouched final test.

All frames belonging to the same participant, and preferably the same recording,
must remain in one partition. Report participant IDs, sample counts, action counts,
and achieved ratios for each partition. Check pairwise intersection of groups and
fail if it is non-empty.

Do not rely blindly on the dataset's `Train`/`Test` folder names if the assignment
requires a fresh 8/1/1 protocol. Build one auditable split from the full eligible
Wi-Pose inventory, while optionally retaining the original split field for later
comparison.

The monitor split is used every epoch for model selection and early stopping. The
final 10% split is evaluated only after the checkpoint is selected.

## Stage 7 — Fit normalization on Wi-Pose training samples only

Fit CSI statistics using only the 80% Wi-Pose training partition. For channel/link
\(l\), use:

\[
X'_{l,t,s}=\frac{X_{l,t,s}-\mu_{l,s}}{\max(\sigma_{l,s},\epsilon)}.
\]

The precise reduction axes must be fixed in code and documented. The current
prototype averages over samples and time while retaining link and subcarrier
statistics. Save `mean`, `std`, sample count, epsilon, split checksum, and code
version in `normalization_wipose_train.npz`.

Apply the same frozen statistics to Wi-Pose monitor/test and WiMANS for the strict
source-only condition. Never fit target statistics for zero-shot testing.

For the SSL-adapted condition, any target-domain normalization is part of the
adaptation method and must use only its declared unlabelled adaptation partition.
The zero-shot evaluation subset must remain excluded. Report this separately so the
comparison with source-only remains interpretable.

## Stage 8 — WiMANS target-domain construction

WiMANS has two roles:

### Unlabelled SSL adaptation

- choose the eligible single-person 5 GHz recordings;
- window their CSI using the agreed duration/stride;
- exclude all zero-shot evaluation recording groups;
- mask time/subcarrier/link regions during training;
- reconstruct cleaned CSI or learn a contrastive representation;
- use no action or video pose label in the SSL loss.

### Frozen cross-dataset evaluation

Two evaluation levels are possible:

1. **Representation/action proxy evaluation:** evaluate learned CSI features using
   WiMANS activity metadata. This does not directly measure HPE accuracy.
2. **Pose pseudo-label evaluation:** run a frozen video pose estimator such as
   AlphaPose on synchronized WiMANS video, manually audit a subset, and compare the
   frozen CSI pose model against those pseudo-labels. Report this as agreement with
   AlphaPose, not true pose ground-truth accuracy.

CSI-Former predictions may be displayed as a model baseline if its official model
and compatible input are available. They must appear in a separate panel and must
never be substituted for AlphaPose labels.

## Stage 9 — Quality control and rejection policy

Generate a machine-readable quality report covering:

- number of files discovered, loaded, failed, and duplicated;
- sample counts by dataset, participant, recording, action, room, band, and split;
- raw and processed CSI shape/dtype/range;
- non-finite and zero-value rates;
- outlier replacement/drop rates;
- pose confidence distribution by joint;
- invalid-anchor and rejected-pose counts;
- CSI/pose timing offsets;
- split leakage checks;
- normalization statistics and post-normalization mean/std;
- random before/after CSI plots;
- random full 18-joint sequence videos and canonical 14-joint training-target
  videos with correct labels and edges.

A sample must not disappear silently. Every rejection receives a reason code, for
example `missing_csi`, `invalid_shape`, `invalid_hip_anchor`, `timing_out_of_range`,
or `insufficient_visible_joints`.

## Stage 10 — Serialization and loading

Write compressed, versioned shards rather than millions of tiny ad-hoc files when
I/O profiling shows a benefit. Each supervised Wi-Pose example should expose:

```text
csi                 float32 [9, 5, 30]
joints_14           float32 [14, 2]
joints_18_pixel     float32 [18, 2]
joint_confidence_18 float32 [18]
joint_valid_14      bool    [14]
action              string/integer metadata
participant         group metadata
sample_id           traceability metadata
```

An unlabelled WiMANS example should expose CSI and metadata but no invented joint
target. Loader tests must verify shape, dtype, finiteness, stable ordering, and
round-trip equality for stored diagnostic fields.

## Stage 11 — Notebook evidence

The master notebook should show, in order:

1. environment and path check using `ML_31005`;
2. inventory summaries for Wi-Pose (Section 1.1) and WiMANS (Section 1.2);
3. action/participant/room/band distributions;
4. raw CSI shapes, descriptive statistics, traces, and heatmaps;
5. raw 18-joint AlphaPose examples with correct joint names and topology;
6. synchronized CSI and complete skeleton sequence/video examples;
7. preprocessing functions used by the importable pipeline;
8. raw-versus-cleaned CSI comparisons;
9. pixel-versus-canonical skeleton comparisons;
10. grouped 8/1/1 split counts and leakage assertions;
11. normalization statistics fitted only on the training partition;
12. processed tensor examples and automated quality checks.

Notebook cells should call the shared implementation where possible. Small key
functions and formulas can also be displayed as assignment-facing evidence, but a
second independent notebook implementation would risk inconsistency.

## Stage 12 — Acceptance checks before training

Preprocessing is considered complete only when all checks pass:

- [ ] Every raw sample is represented in an inventory or failure report.
- [ ] CSI axes are confirmed and every model tensor is `[9, 5, 30]`.
- [ ] All 18 Wi-Pose points have verified names and correct visual edges.
- [ ] The 18-to-14 mapping is explicit and covered by tests.
- [ ] No leg-to-head or other non-anatomical edges remain.
- [ ] CSI windows and pose frames pass the synchronization audit.
- [ ] Participant/recording groups do not overlap across 80/10/10 splits.
- [ ] All fitted preprocessing statistics use training data only.
- [ ] WiMANS SSL and evaluation recording groups are disjoint.
- [ ] WiMANS pseudo-labels, if generated, are clearly identified as such.
- [ ] Random sequence videos show plausible temporal movement and correct actions.
- [ ] Processed loaders pass shape, dtype, finite-value, and reproducibility tests.
- [ ] Manifests, parameters, quality report, and processed artefacts are versioned.

## Execution order from here

1. Correct and test the Wi-Pose 18-joint decoder and named 18-to-14 mapping.
2. Rebuild both complete raw inventories and produce the schema/quality report.
3. Validate CSI-to-frame synchronization on several full action sequences.
4. Define participant/recording groups and write the deterministic 80/10/10 split.
5. Implement continuous-recording CSI cleaning, then extract five-packet windows.
6. Fit source normalization on the 80% Wi-Pose training partition only.
7. Canonicalize pose targets and serialize versioned Wi-Pose tensors.
8. Build disjoint WiMANS SSL-adaptation and zero-shot evaluation windows.
9. Add the audit tables, plots, and full-sequence videos to the master notebook.
10. Run acceptance checks. Only then start the source-only training condition.

## Condition-specific differences

| Decision | Source-only zero-shot | SSL domain-adapted |
|---|---|---|
| Supervised pose labels | Wi-Pose train only | Wi-Pose train only |
| WiMANS used during representation learning | No | Yes, CSI without pose labels |
| Source normalization | Fit on Wi-Pose train | Same source parameters retained |
| Target statistics | Never fit | Only adaptation partition if explicitly configured |
| Checkpoint selection | Wi-Pose monitor | Wi-Pose monitor; no zero-shot-test feedback |
| Final comparison | Wi-Pose test + frozen WiMANS evaluation | Same untouched evaluation sets |

This keeps the scientific question clean: does unlabelled WiMANS adaptation improve
cross-domain CSI representations and pose transfer without using target pose labels?
