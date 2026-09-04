# Model Structure — SSL Domain-adapted

## Stage 1: masked CSI reconstruction

`MaskedCSIAutoencoder` randomly masks 30% of CSI positions in the shared `[9, 5, 30]` representation. It uses the exact four-block PoseCNN encoder, reducing frequency `30 -> 15 -> 8 -> 4` while retaining all five packets. A lightweight decoder reconstructs the original amplitude, and loss is computed only on masked positions.

## Stage 2: pose regression

All pretrained encoder parameters strictly initialize `DomainAdaptedPoseModel`. The decoder is discarded, and the same regression head and 14-joint output used by the source-only condition are trained with Wi-Pose pose labels.

No WiMANS pose labels enter either stage. Because target-domain CSI is seen during pretraining, this condition is label-free target-domain adaptation, not strict zero-shot learning.
