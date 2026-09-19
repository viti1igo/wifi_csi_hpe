# Implemented Preprocessing

The preprocessing code converts Wi-Pose and WiMANS CSI into the same model input while keeping their data roles separate. Wi-Pose supplies pose supervision. WiMANS supplies unlabelled adaptation CSI and a disjoint final-evaluation domain.

## Output contracts

| Item | Shape | Type | Meaning |
|---|---|---|---|
| CSI sample | `[9,5,30]` | `float32` | Nine Tx-Rx links, five packets and thirty subcarriers |
| Wi-Pose source pose | `[18,2]` | `float32` | Original AlphaPose pixel coordinates |
| BODY-14 target | `[14,2]` | `float32` | Hip-centred, torso-normalized pose |
| Joint confidence | `[14]` | `float32` | AlphaPose confidence used by loss and evaluation |
| Normalizer mean/std | `[9,30]` | `float32` | Training-only link and subcarrier statistics |

## CSI transformation

Both loaders take five packets with logical shape `[5,3,3,30]`. They convert complex CSI to amplitude,

```text
A = |H| = sqrt(Re(H)^2 + Im(H)^2)
```

then transpose and flatten the Tx-Rx pair:

```text
[5,3,3,30] -> [3,3,5,30] -> [9,5,30]
```

The nine channels therefore represent the Cartesian product of three transmitters and three receivers. The reshape changes axis organization, not the measurements.

The implementation accepts the two observed five-packet layouts, `[5,3,3,30]` and `[5,30,3,3]`, and rejects any other shape.

## Normalization

The pipeline fits one mean and standard deviation for each link and subcarrier. It averages over Wi-Pose training samples and the five-packet time axis:

```text
X'[l,t,s] = (X[l,t,s] - mean[l,s]) / max(std[l,s], epsilon)
```

Only the 132,786 Wi-Pose training samples contribute to these statistics. The monitor split, test split and held-out WiMANS recordings never affect them. Both conditions use the saved normalizer during supervised training and final evaluation.

The implemented version does not apply ACF, Hampel, Savitzky-Golay or band-pass filtering to each five-packet sample. Such windows are too short for those operations to be justified without a separate continuous-recording study.

## Wi-Pose processing

Each MATLAB sample contains CSI and `SkeletonPoints`. The processor:

1. Loads CSI and stores it in the five-packet logical layout.
2. Decodes `SkeletonPoints` as 18 x coordinates, 18 y coordinates and 18 confidence values.
3. Preserves all 18 points for audit and visualisation.
4. Selects the 14 body joints through explicit names, excluding eyes and ears.
5. Stores the selected pixel coordinates and confidence values.
6. Canonicalizes the BODY-14 target.
7. Writes a compressed NPZ shard and a traceable manifest record.

The BODY-14 order is nose, neck, right shoulder, right elbow, right wrist, left shoulder, left elbow, left wrist, right hip, right knee, right ankle, left hip, left knee and left ankle.

### Pose canonicalization

For right hip `p_rh`, left hip `p_lh`, neck `p_n` and joint `p_j`, the processor calculates

```text
c = (p_rh + p_lh) / 2
s = ||p_n - c||_2
p_tilde_j = (p_j - c) / max(s, epsilon)
```

This removes image translation and approximate person scale. The model predicts relative body configuration in the same canonical coordinate system.

## Wi-Pose split

The deterministic participant-grouped split contains:

| Partition | Participant groups | Frames | Use |
|---|---:|---:|---|
| Train | 96 | 132,786 | Parameter learning and normalization fit |
| Monitor | 12 | 16,169 | Checkpoint selection and early stopping |
| Test | 12 | 17,645 | One-time final evaluation |

All frames from one participant stay in one partition.

## WiMANS processing

WiMANS stores continuous recordings as `[packets,3,3,30]`. The SSL loader samples five consecutive packets from permitted classroom and meeting-room recordings. It applies the same amplitude conversion, reshape and saved normalizer used for Wi-Pose.

The target-domain partition contains:

| Role | Recordings | Environments | Labels used |
|---|---:|---|---|
| SSL adaptation | 1,188 | Classroom and meeting room | No pose or action labels |
| Frozen evaluation | 594 | Empty room | AlphaPose pseudo-reference only after training |

During Condition B pretraining, the loader masks 30% of the common CSI tensor and trains the autoencoder to reconstruct masked values. Condition A never loads WiMANS during learning.

## WiMANS video alignment and pseudo-reference

Each held-out video contains 90 frames. Because the release does not provide authoritative packet-to-frame timestamps, evaluation maps the video-frame centres uniformly across the CSI recording. It clips centres at recording boundaries and extracts five packets around each centre. Offsets from -2 to +2 frames measure alignment sensitivity.

AlphaPose produces COCO-17 detections from the held-out videos. The evaluation pipeline:

1. Selects the highest-scoring detected person.
2. Maps COCO-17 joints to BODY-14.
3. Converts image coordinates `(x,y)` to the Wi-Pose camera frame `(y,-x)`.
4. Requires confident hip and neck anchors and at least ten confident BODY-14 joints.
5. Applies the same hip-centre and torso-scale canonicalization.
6. Stores raw, mapped and canonical coordinates with confidence and provenance.

The primary confidence threshold is 0.30. Thresholds 0.10 and 0.50 support sensitivity analysis. AlphaPose output remains a pseudo-reference and does not enter training or checkpoint selection.

## Quality checks

- Reject unexpected CSI and skeleton shapes.
- Reject non-finite or zero torso scales.
- Preserve raw pixel coordinates and confidence values.
- Verify that split groups do not overlap.
- Store source paths, identifiers, partitions and preprocessing versions.
- Verify split, normalizer, model and pseudo-reference checksums before evaluation.
- Report missing AlphaPose frames as reduced coverage instead of replacing them with zero poses.

The completed pipeline produced 166,600 Wi-Pose shards and 52,684 valid WiMANS evaluation frames from 594 recordings.
