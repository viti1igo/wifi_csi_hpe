# Full Pipeline — Source-only Zero-shot

1. Download and verify Wi-Pose and WiMANS.
2. Extract archives only after review.
3. Explore file schemas and construct explicit JSONL manifests.
4. Map both datasets to the shared CSI and 14-joint interfaces.
5. Create participant-grouped training, monitor, and final-test splits.
6. Fit CSI normalization using Wi-Pose training data only.
7. Probe safe per-GPU batch size, then launch two-rank DDP.
8. Train loss ablations and select checkpoints using monitor NME.
9. Evaluate the selected checkpoint once on the Wi-Pose final test set.
10. Freeze all parameters and infer on the reserved WiMANS test domain.
11. Compare predictions with video-derived, confidence-filtered pseudo-labels.
12. Populate the master notebook and assignment report sections.

