# Reproduction Workflow

## Environment and invariants

- Use the `ML_31005` environment for preprocessing, model training and evaluation.
- Use seed `31050` unless a recorded experiment overrides it.
- Keep raw and extracted data immutable.
- Check split, normalizer and checkpoint hashes before loading trained weights.
- Never use the Wi-Pose test split or WiMANS empty-room subset for checkpoint selection.
- Run AlphaPose in its separate environment; use its output only as a final-evaluation pseudo-reference.

## Recommended execution order

1. Build or verify the dataset inventories.
2. Convert Wi-Pose samples and audit the named BODY-14 mapping.
3. Verify the participant-grouped 80/10/10 split.
4. Load the Wi-Pose-training normalizer.
5. Verify the disjoint WiMANS SSL and empty-room partitions.
6. Load or reproduce Condition A training.
7. Load or reproduce Condition B masked pretraining and fine-tuning.
8. Run Condition A standalone evaluation in notebook Section 6.
9. Run the frozen comparative evaluation in Section 7.
10. Review Section 8 before exporting the notebook.

Every experimental section in the notebook includes a restart cell. A reader can start from Section 4, 5, 6 or 7 once the required saved artefacts exist.

## Failure rules

| Failure | Response |
|---|---|
| Invalid tensor shape or non-finite value | Stop and report the source sample |
| Split overlap | Invalidate the split, normalizer and dependent checkpoints |
| Checksum mismatch | Refuse to load the checkpoint |
| GPU out of memory | Record the attempted batch size and rerun the common batch probe |
| NaN loss | Stop training and retain the preceding checkpoint and log |
| Missing AlphaPose detection | Exclude the frame from valid coverage; do not create a zero pose |
| Notebook and saved result disagreement | Treat the notebook output as stale and rerun the affected evaluation cell |

## WiMANS evaluation replay

```bash
python -m shared.evaluation.alphapose_pipeline prepare
# Run official AlphaPose on the prepared held-out video stream.
python -m shared.evaluation.alphapose_pipeline convert --raw-json PATH/alphapose-results.json
python -m shared.evaluation.run_final all
```

The converter reuses complete per-recording NPZ files. Use `--force` only after changing the coordinate conversion or pseudo-reference contract.

## Publishing

Commit the executed notebook only after checking that it contains no error output. Keep large raw data, checkpoints and intermediate results outside Git. Selected MP4 evaluation outputs can be committed separately. GitHub may not display base64 HTML5 video embedded inside a notebook; provide a file link or preview image when public playback matters.
