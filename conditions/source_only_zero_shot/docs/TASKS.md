# Task Tracker — Source-only Zero-shot

Exactly one status checkbox is selected per task.

## Scaffold and acquisition

| Task | NOT RUN | RUNNING | DONE |
|---|:---:|:---:|:---:|
| Project directories and condition files | [ ] | [ ] | [x] |
| Model, loss, trainer, metrics, and configuration code | [ ] | [ ] | [x] |
| Condition and master notebook scaffolds | [ ] | [ ] | [x] |
| Wi-Pose archive downloaded and checksum recorded | [ ] | [ ] | [x] |
| WiMANS files downloaded and checksums recorded | [ ] | [ ] | [x] |

## Deferred stages

| Task | NOT RUN | RUNNING | DONE |
|---|:---:|:---:|:---:|
| Archive extraction and data exploration | [ ] | [ ] | [x] |
| Corrected preprocessing after manifest construction | [ ] | [ ] | [x] |
| Participant split audit | [ ] | [ ] | [x] |
| Frozen Wi-Pose training normalizer | [ ] | [ ] | [x] |
| Frozen WiMANS zero-shot partition | [ ] | [ ] | [x] |
| Two-GPU visibility and model-forward smoke test | [ ] | [ ] | [x] |
| Dynamic batch-size probe at training launch | [ ] | [ ] | [x] |
| Wi-Pose supervised training | [ ] | [ ] | [x] |
| Monitor-based checkpoint selection | [ ] | [ ] | [x] |
| Final Wi-Pose test | [ ] | [ ] | [x] |
| Strict frozen WiMANS zero-shot inference | [ ] | [ ] | [x] |
| WiMANS pseudo-label evaluation | [ ] | [ ] | [x] |
## Current result

Source-only training is complete: stopped epoch 78, selected epoch 63, monitor
NME 0.211997.  Native test evaluation is saved under this condition's `results/`;
the shared WiMANS pseudo-reference comparison is documented in master Section 6.
Native test NME is 0.206182; coordinate-aligned WiMANS pseudo-reference NME is 0.353495.
