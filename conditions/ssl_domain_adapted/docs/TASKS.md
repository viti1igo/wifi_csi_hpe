# Task Tracker — SSL Domain-adapted

Exactly one status checkbox is selected per task.

## Scaffold and acquisition

| Task | NOT RUN | RUNNING | DONE |
|---|:---:|:---:|:---:|
| Project directories and condition files | [ ] | [ ] | [x] |
| Pretrainer, pose model, loss, trainer, and metrics code | [ ] | [ ] | [x] |
| Condition and master notebook scaffolds | [ ] | [ ] | [x] |
| Wi-Pose archive downloaded and checksum recorded | [ ] | [ ] | [x] |
| WiMANS files downloaded and checksums recorded | [ ] | [ ] | [x] |

## Deferred stages

| Task | NOT RUN | RUNNING | DONE |
|---|:---:|:---:|:---:|
| Archive extraction and data exploration | [ ] | [ ] | [x] |
| Corrected shared preprocessing after manifest construction | [ ] | [ ] | [x] |
| Participant and environment split audit | [ ] | [ ] | [x] |
| WiMANS SSL and zero-shot partition isolation | [ ] | [ ] | [x] |
| Frozen Wi-Pose training normalizer | [ ] | [ ] | [x] |
| Two-GPU visibility and model-forward smoke test | [ ] | [ ] | [x] |
| Dynamic batch-size probe at training launch | [ ] | [ ] | [x] |
| Masked CSI self-supervised pretraining | [ ] | [ ] | [x] |
| Wi-Pose supervised fine-tuning | [ ] | [ ] | [x] |
| Monitor-based checkpoint selection | [ ] | [ ] | [x] |
| Final Wi-Pose test | [ ] | [ ] | [x] |
| Held-out WiMANS empty-room inference | [ ] | [ ] | [x] |
| WiMANS pseudo-label evaluation | [ ] | [ ] | [x] |
## Current result

SSL pretraining completed 50 epochs and fine-tuning stopped at epoch 79.  The
selected pose checkpoint is epoch 64 with monitor NME 0.207625.  Native test
evaluation is saved under this condition's `results/`; the common WiMANS comparison
is documented in master Section 6. Native test NME is 0.202268; WiMANS
pseudo-reference NME is 0.336049, so adaptation improves that endpoint after the
WiMANS reference is aligned to the Wi-Pose stored-camera coordinate frame.
