# Full Pipeline — SSL Domain-adapted

1. Download and verify both datasets.
2. Extract and inspect only after review.
3. Build common CSI manifests and enforce environment exclusions.
4. Fit preprocessing from permitted training/adaptation data only.
5. Pretrain the CSI encoder with masked reconstruction.
6. Transfer compatible encoder weights into the pose model.
7. Fine-tune with Wi-Pose pose labels and skeleton-aware loss.
8. Select checkpoints using the Wi-Pose monitor participant.
9. Evaluate once on the Wi-Pose final-test participant.
10. Freeze weights and evaluate on unseen WiMANS empty-room recordings.
11. Compare against source-only predictions using identical pseudo-label frames.
12. Present the controlled comparison in the master notebook.

