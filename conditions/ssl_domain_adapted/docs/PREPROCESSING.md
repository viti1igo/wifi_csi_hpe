# Preprocessing — SSL Domain-adapted

The complete shared procedure and acceptance criteria are defined in
[`../../../docs/PREPROCESSING.md`](../../../docs/PREPROCESSING.md). This file records
only the decisions specific to this condition.

Shared preprocessing and readiness validation completed on 2026-09-03. This
condition may use 1,188 single-person 5 GHz classroom/meeting-room recordings for
unlabelled SSL; 594 empty-room recordings remain isolated for zero-shot evaluation.

Both datasets will be mapped to `[9, 5, 30]` CSI amplitude windows. Wi-Pose pose targets will be mapped to canonical 14-joint coordinates. WiMANS classroom and meeting-room CSI may be used without pose labels during masked pretraining; empty-room CSI and all video-derived test labels remain unavailable until final evaluation.

Normalization policy will be finalized only after inspecting raw value distributions. Any domain-specific normalization must be fitted without reading the held-out empty-room partition and must be applied identically to both compared transfer conditions where required for fairness.
