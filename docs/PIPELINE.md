# Master Pipeline

This is the execution contract for the project. The engineering rationale, data
contracts, failures, and operational gates are in
[`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md); exact transformation details are in
[`PREPROCESSING.md`](PREPROCESSING.md).

| Stage | Input | Output | Exit gate |
|---|---|---|---|
| 0. Reproducibility | `ML_31005`, raw archives | run/config record | package/path/checksum checks pass |
| 1. Inventory | extracted files | immutable JSONL inventories | every input has a record or reason code |
| 2. Schema audit | inventory + raw files | shape/range/metadata report | CSI and labels decode correctly |
| 3. Wi-Pose processing | CSI + 18 AlphaPose points | versioned supervised shards | named 18→14 mapping and synchronization pass |
| 4. Split and normalization | eligible Wi-Pose records | 8/1/1 split + frozen statistics | zero group leakage; training-only fit |
| 5. WiMANS processing | target CSI and metadata | disjoint SSL/evaluation windows | no target evaluation leakage |
| 6. Source-only training | Wi-Pose training partition | selected source checkpoint | monitor metrics logged every epoch |
| 7. SSL adaptation | allowed unlabelled CSI | pretrained encoder + selected pose checkpoint | no target pose labels used |
| 8. Evaluation | frozen selected checkpoint | final metrics/predictions/figures | test used only after selection |
| 9. Assignment notebook | saved artefacts | reproducible narrative | claims map to saved evidence |
| 10. WiMANS video reference | 594 held-out videos | BODY-14 pseudo-reference NPZ files | provenance, confidence, and coverage checks pass |

```text
Wi-Pose raw MAT -> audit -> synchronized CSI + AlphaPose-18
                                     |
                                     +-> named BODY-14 target -> grouped 8/1/1 -> source trainer
                                     |
WiMANS raw MAT -> audit -> eligible CSI windows -> SSL pretraining only -> fine-tuning
```

The source-only condition never sees WiMANS during learning. The SSL condition may
see only its declared unlabelled WiMANS adaptation partition; its zero-shot
evaluation partition and all target pose pseudo-labels remain unavailable until
final evaluation. Raw/extracted data is immutable and ignored by Git. Each condition
writes only under its own `logs/` and `results/` directories.

## Frozen evaluation flow

```text
Wi-Pose test ───────────────> Condition A/B best.pt ──> native test metrics
WiMANS held-out CSI ────────> identical 5-packet windows ─┐
WiMANS held-out video ──────> AlphaPose COCO-17 ─> BODY-14 ─> Wi-Pose camera frame
                                                           └─> paired agreement + sensitivity
```

Video frame centres are mapped uniformly across each CSI recording because the
release has no authoritative packet-to-frame timestamps.  Offset results at
`-2,-1,0,+1,+2` frames bound this assumption.  AlphaPose labels are unavailable to
training and checkpoint selection.

The completed extraction produced 52,684 primary-valid frames (98.55% of 53,460).
AlphaPose image coordinates are rotated from `(x,y)` to the Wi-Pose stored-camera
frame `(y,-x)` before hip-centering and torso normalization; upright rotation is
then used only for display.
Wi-Pose measures native target accuracy; WiMANS measures agreement with AlphaPose
under domain and synchronization uncertainty. Condition B improved both Wi-Pose NME
and held-out WiMANS pseudo-reference agreement after coordinate-frame alignment.
