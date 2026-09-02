# Preprocessing — Source-only Zero-shot

The complete shared procedure and acceptance criteria are defined in
[`../../../docs/PREPROCESSING.md`](../../../docs/PREPROCESSING.md). This file records
only the decisions specific to this condition.

No preprocessing has been run.

## Planned Wi-Pose mapping

- Read `csi_serial` only after schema confirmation.
- Convert complex CSI to amplitude.
- Validate raw shape `5 × 30 × 3 × 3`.
- Transpose and flatten antenna pairs to `9 × 5 × 30`.
- Map the available pose labels to the common 14-joint ordering.
- Centre each skeleton on the hip midpoint and divide by torso length.
- Retain pose confidence for confidence-weighted loss and metrics.

## Planned WiMANS mapping

- Use single-person, 5 GHz samples only.
- Align five CSI packets around each selected video timestamp.
- Use Wi-Pose training statistics for strict zero-shot normalization.
- Never fit a target-specific normalizer in this condition.
