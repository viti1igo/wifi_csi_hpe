# Implemented Pipeline

The project compares source-only pose learning with label-free target-domain encoder pretraining.

```text
Wi-Pose MAT files
  -> CSI amplitude [5,3,3,30]
  -> transpose and flatten Tx-Rx links [9,5,30]
  -> Wi-Pose-training normalization
  -> PoseCNN
  -> canonical BODY-14 pose [14,2]

WiMANS recordings
  -> select a centred five-packet window [5,3,3,30]
  -> the same [9,5,30] representation and frozen normalizer
  -> Condition A or Condition B PoseCNN
```

## Stage contract

| Stage | Implemented output | Acceptance check |
|---|---|---|
| Inventory | Wi-Pose and WiMANS JSONL manifests | Required files, keys and metadata exist |
| Wi-Pose conversion | 166,600 traceable NPZ shards | CSI, AlphaPose-18, BODY-14 and confidence shapes pass |
| Split | 96/12/12 participant groups | No participant overlap |
| Normalization | Per-link, per-subcarrier mean and standard deviation | Fitted on 132,786 Wi-Pose training samples only |
| WiMANS partition | 1,188 SSL recordings and 594 held-out recordings | Empty-room groups excluded from SSL |
| Condition A | Monitor-selected source-only checkpoint | No WiMANS input before inference |
| Condition B pretraining | Masked-CSI encoder checkpoint | No pose label enters masked MSE |
| Condition B fine-tuning | Monitor-selected pose checkpoint | Same labelled Wi-Pose data and supervised settings as A |
| Wi-Pose evaluation | Native test predictions and metrics | 17,645 untouched samples |
| WiMANS evaluation | Pseudo-reference predictions, metrics and sensitivity | 594 recordings and 52,684 valid frames |

## Condition A

Condition A initializes PoseCNN randomly and trains it on the Wi-Pose training partition. The monitor partition controls checkpoint selection and early stopping. WiMANS appears only at final inference.

## Condition B

Condition B balances unlabelled Wi-Pose training windows with permitted WiMANS classroom and meeting-room recordings. It masks 30% of each CSI sample and minimizes reconstruction error. The pretrained encoder then initializes the same PoseCNN architecture used by Condition A. Fine-tuning uses the same labelled Wi-Pose split, loss weights, optimizer and selection metric.

## Frozen evaluation

```text
Wi-Pose test -> selected A/B checkpoints -> native pose metrics

WiMANS held-out CSI -> five-packet windows -> selected A/B checkpoints
WiMANS held-out video -> AlphaPose COCO-17 -> BODY-14 -> canonical pose
                                            |
                                            -> paired agreement metrics
```

The release does not contain authoritative packet-to-video timestamps. Evaluation maps 90 video-frame centres uniformly across each CSI recording, extracts centred five-packet windows, and repeats the analysis at offsets of -2, -1, 0, +1 and +2 frames. The offset analysis measures sensitivity to this alignment assumption.

AlphaPose image coordinates `(x,y)` are converted to the Wi-Pose camera frame `(y,-x)` before hip centring and torso normalization. Display code applies the inverse upright orientation only when plotting.
