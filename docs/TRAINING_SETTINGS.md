# Training Settings and Experiment Contract

| Setting | Value / rule |
|---|---|
| Environment | `ML_31005` only |
| Seed | `31050`, stored in resolved config and run metadata |
| Input | finite `float32 [B, 9, 5, 30]` cleaned CSI amplitude tensor |
| Pose target | `float32 [B, 14, 2]`, hip-centred and torso-normalized |
| Label reliability | AlphaPose confidence and validity mask weight coordinate loss/metrics |
| Split | deterministic participant/recording-grouped 80/10/10 |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Schedule | maximum 100 supervised epochs, patience 15 |
| Precision | BF16 when supported, otherwise FP16 AMP |
| Parallelism | two-rank DDP only if both GPUs are visible and verified |
| Batch probe | largest safe per-GPU batch divisible by 8, retaining configured VRAM reserve |
| Checkpoint choice | smallest Wi-Pose monitor NME; final test unavailable to selection |

The supervised objective is confidence-weighted coordinate loss plus optional bone
and left/right symmetry regularizers. Every epoch writes total/coordinate/bone/
symmetry loss, NME, PCK, bone error, learning rate, batch size, wall time, and GPU
memory to condition-local CSV and JSONL files. The monitor partition is evaluated at
the end of every epoch; the final test is evaluated only after selection.

| Property | `source_only_zero_shot` | `ssl_domain_adapted` |
|---|---|---|
| WiMANS CSI seen before inference | No | Only declared unlabelled adaptation windows |
| WiMANS pose labels seen | No | No |
| Checkpoint selection | Wi-Pose monitor NME | Wi-Pose monitor NME |
| Final evaluation sets | Identical, frozen | Identical, frozen |

GPU probing and training must not begin until the preprocessing acceptance checks in
[`PREPROCESSING.md`](PREPROCESSING.md) pass.
