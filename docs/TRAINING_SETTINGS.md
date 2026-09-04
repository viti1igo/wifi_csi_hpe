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
| Parallelism | reliable single-GPU notebook runs; two-rank DDP is an optional external workflow |
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

## Completed selected runs

| Condition | Stopped | Selected `best.pt` | Best monitor NME |
|---|---:|---:|---:|
| Source-only | epoch 78 | epoch 63 | 0.211997 |
| SSL reconstruction | epoch 50 | epoch 50 | masked MSE 5.475179 |
| SSL-adapted fine-tuning | epoch 79 | epoch 64 | 0.207625 |

Section 6 loads only the two supervised `best.pt` checkpoints.  It performs no
optimizer step, early stopping, checkpoint selection, or training resumption.

## Frozen evaluation result

| Dataset/reference | Condition A NME | Condition B NME | B minus A paired result |
|---|---:|---:|---:|
| Wi-Pose native test, 17,645 samples | 0.206182 | 0.202268 | -0.004029; 95% CI [-0.007606, -0.000060] |
| WiMANS AlphaPose pseudo-reference, 52,684 valid frames | 0.353495 | 0.336049 | -0.017837; 95% CI [-0.022474, -0.013166] |

Condition B improves slightly on native Wi-Pose and improves cross-domain
agreement in the held-out WiMANS environment. The second row is agreement with a
video estimator under approximate synchronization, not physical pose accuracy.
