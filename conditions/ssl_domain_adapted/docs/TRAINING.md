# Training Method — SSL Domain-adapted

## Pretraining partition

Use unlabeled Wi-Pose training CSI plus WiMANS classroom and meeting-room single-person 5 GHz CSI. Reserve all empty-room WiMANS samples for final cross-domain evaluation. Masked reconstruction pretraining runs for up to 50 epochs and selects its checkpoint using reconstruction performance on a monitor partition.

## Fine-tuning split

Use the same participant-grouped Wi-Pose 80/10/10 target ratio as the source-only condition. With 12 whole participants, the closest allocation is 10/1/1. Show training and monitor metrics each epoch; evaluate the final test participant only after checkpoint selection.

## Compute and logging

Both stages use two-rank DDP, mixed precision, and independent automatic batch probing. Logs and checkpoints remain local to this condition. The pretraining and fine-tuning stages use separate log files so their objectives cannot be confused.

