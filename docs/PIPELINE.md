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
