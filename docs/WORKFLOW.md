# Engineering Workflow

## Working agreement

Use `ML_31005` exclusively. Before a job, capture package versions, Git revision,
resolved configuration, raw checksums, split version, and output version. Raw and
extracted files are immutable. Each task tracker row has exactly one status.

## Execution sequence

1. Change the shared contract and its validation before changing a condition.
2. Run a deterministic small-sample audit before a whole-dataset operation.
3. Review schema, synchronization, named-joint, and split-leakage reports before
   fitting normalization.
4. Run a single-batch smoke test before DDP or full training.
5. Train source-only first; it is the controlled baseline for SSL adaptation.
6. Persist machine-readable results before notebook plots/tables.
7. Cite saved result paths and split names for every notebook conclusion.

## Failure handling

| Event | Required response |
|---|---|
| Invalid file/schema | Record a reason code; never silently skip it. |
| Interrupted preprocessing | Resume only after checking existing output contracts/version. |
| OOM | Record attempted batch, reduce it on all ranks, resume from checkpoint. |
| NaN/Inf | Stop, save diagnostic metadata, inspect preprocessing before retry. |
| Split overlap | Invalidate affected normalizers and downstream runs. |
| Notebook/result mismatch | Treat the notebook output as stale and regenerate it. |

The detailed system lifecycle and acceptance gates are in
[`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md).
