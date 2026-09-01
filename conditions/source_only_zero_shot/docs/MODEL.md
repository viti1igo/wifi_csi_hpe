# Model Structure — Source-only Zero-shot

## Interface

- Input: standardized CSI amplitude tensor `[batch, 9, 5, 30]`.
- Output: canonical body coordinates `[batch, 14, 2]`.
- The 14 joints exclude eyes and ears because they are weakly observable from WiFi and do not map robustly across pose-label formats.

## Architecture

`PoseCNN` contains four residual convolution blocks with 32, 64, 128, and 192 channels. Convolution operates over the five-packet temporal axis and 30-subcarrier frequency axis. Downsampling occurs only along the subcarrier axis so the short temporal context is preserved. Adaptive pooling produces a fixed representation, and an MLP regresses 28 coordinates.

This condition uses random initialization and receives supervision only from Wi-Pose. The trained weights are frozen before WiMANS inference.

## Controlled ablations

The architecture remains fixed while loss weights change:

1. coordinate-only: `bone_weight=0`, `symmetry_weight=0`;
2. coordinate plus bone: `symmetry_weight=0`;
3. full skeleton-aware objective.

