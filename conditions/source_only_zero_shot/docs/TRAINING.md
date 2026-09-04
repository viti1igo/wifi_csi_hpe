# Training Method — Source-only Zero-shot

## Split

Samples are grouped by participant before splitting. The requested ratio is 80/10/10 for training, per-epoch monitoring, and untouched final testing. With 12 indivisible participants, the nearest deterministic allocation is 10/1/1; achieved sample ratios will be logged after exploration.

The monitor split is displayed after every epoch and controls early stopping and best-checkpoint selection. The final test split is evaluated once after selection.

## Optimization

- AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Maximum 100 epochs with monitor patience 15.
- BF16 when supported, otherwise FP16.
- Reliable single-GPU notebook execution; two-GPU DDP remains an external option.
- Automatic batch probing chooses the largest safe batch divisible by eight while retaining the configured reserve.

## Logs

Each epoch writes CSV and JSONL records locally with training and monitor losses, NME, PCK, bone error, learning rate, batch sizes, GPU memory, elapsed time, and checkpoint path. Logs must not be used to inspect the final test set during model development.
## Completed run

Training stopped at epoch 78 after the configured patience.  The selected
`best.pt` is epoch 63 with monitor NME `0.211997`.  Section 6 reloads this frozen
checkpoint; `last.pt` remains continuation state and is not the reported model.
