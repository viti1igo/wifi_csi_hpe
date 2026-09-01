# Model Structure — SSL Domain-adapted

## Stage 1: masked CSI reconstruction

`MaskedCSIAutoencoder` randomly masks 30% of CSI positions in the shared `[9, 5, 30]` representation. A residual CNN encoder processes the masked tensor and a lightweight decoder reconstructs the original amplitude. Reconstruction loss is computed on masked positions so the encoder must learn relationships between links, time, and subcarriers.

## Stage 2: pose regression

Compatible pretrained encoder parameters initialize `DomainAdaptedPoseModel`. The decoder is discarded, and the same regression head and 14-joint output used by the source-only condition are trained with Wi-Pose pose labels.

No WiMANS pose labels enter either stage. Because target-domain CSI is seen during pretraining, this condition is label-free target-domain adaptation, not strict zero-shot learning.

