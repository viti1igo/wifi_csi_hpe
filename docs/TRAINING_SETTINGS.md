# Training Settings and Experiment Contract

## Shared supervised configuration

| Setting | Value |
|---|---|
| Seed | `31050` |
| Input | `float32 [B,9,5,30]` CSI amplitude |
| Target | `float32 [B,14,2]` canonical pose |
| Encoder channels | `32, 64, 128, 192` |
| Dropout | `0.20` |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Maximum supervised epochs | `100` |
| Early-stopping patience | `15` |
| Precision | BF16 when supported, otherwise FP16 AMP |
| Selection metric | Lowest Wi-Pose monitor NME |
| PCK thresholds | `0.05`, `0.10`, `0.20` |

PoseCNN uses four residual convolution blocks. Frequency is downsampled after the first block; the five-packet time axis remains intact. Adaptive average pooling and an MLP produce 28 values, reshaped to 14 two-dimensional joints.

## Supervised objective

For prediction `p`, target `y` and AlphaPose confidence `w`, training minimizes

```text
L = 1.00 L_coordinate + 0.25 L_bone + 0.05 L_symmetry
```

`L_coordinate` is confidence-weighted Smooth L1 loss over joint coordinates. `L_bone` applies Smooth L1 loss to vectors along the BODY-14 skeleton edges. `L_symmetry` penalizes differences between corresponding left and right predicted bone lengths.

## Condition controls

| Property | Condition A | Condition B |
|---|---|---|
| Encoder initialization | Random | Masked-CSI pretrained |
| Labelled pose data | Wi-Pose train only | Same Wi-Pose train data |
| WiMANS CSI before inference | None | Unlabelled classroom and meeting-room recordings |
| WiMANS pose labels | None | None |
| Held-out empty-room CSI | Final evaluation only | Final evaluation only |
| Checkpoint selection | Wi-Pose monitor NME | Wi-Pose monitor NME |

Condition B pretraining uses a mask ratio of 0.30, AdamW at `1e-3`, and 50 epochs. The reconstruction loss is mean squared error evaluated only at masked positions. Balanced batches alternate unlabelled Wi-Pose and permitted WiMANS samples.

## Selected runs

| Run | Last epoch | Selected epoch | Selection value |
|---|---:|---:|---:|
| Condition A supervised | 78 | 63 | monitor NME 0.211997 |
| Condition B masked reconstruction | 50 | 50 | monitor masked MSE 5.475179 |
| Condition B supervised fine-tuning | 79 | 64 | monitor NME 0.207625 |

## Final results

| Dataset | Metric | Condition A | Condition B |
|---|---|---:|---:|
| Wi-Pose | NME / 2D MPJPE | 0.2062 | 0.2023 |
| Wi-Pose | PCK@0.20 | 0.6687 | 0.6865 |
| Wi-Pose | PCK-AUC | 0.6445 | 0.6548 |
| WiMANS | NME / 2D MPJPE | 0.3535 | 0.3360 |
| WiMANS | PCK@0.20 | 0.5140 | 0.5143 |
| WiMANS | PCK-AUC | 0.5239 | 0.5331 |

The participant-grouped Wi-Pose bootstrap estimated a Condition B minus A NME change of -0.004029 with a 95% interval of [-0.007606, -0.000060]. The recording-grouped WiMANS estimate was -0.017837 with a 95% interval of [-0.022474, -0.013166].
