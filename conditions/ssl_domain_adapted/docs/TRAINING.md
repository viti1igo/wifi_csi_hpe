# Training Method — SSL Domain-adapted

## Pretraining partition

Use unlabelled Wi-Pose training CSI plus WiMANS classroom and meeting-room single-person 5 GHz CSI. A seeded hash reserves 10% of both permitted sources for monitoring. Each epoch pairs every remaining WiMANS recording with an equally sized random Wi-Pose subset. Reserve all empty-room WiMANS samples for final cross-domain evaluation. Pretraining runs for up to 50 epochs and selects by monitor masked MSE.

## Fine-tuning split

Use the same participant-grouped Wi-Pose 80/10/10 split as the source-only condition: 96/12/12 of 120 participants. Show training and monitor metrics each epoch; evaluate the final-test participants only after checkpoint selection.

## Compute and logging

Both completed notebook stages used one GPU, mixed precision, and independent automatic batch probing; two-rank DDP remains an external option. Logs and checkpoints remain local to this condition. The pretraining and fine-tuning stages use separate log files so their objectives cannot be confused.
