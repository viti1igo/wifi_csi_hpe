# Assignment Notebook Flow

The submitted notebook is [`notebooks/24681189_notebook.ipynb`](../notebooks/24681189_notebook.ipynb). It contains 138 cells: 72 Markdown cells and 66 executed code cells. The saved notebook has no error outputs. Shared modules under `shared/` remain the implementation source of truth.

| Section | Purpose | Saved evidence |
|---|---|---|
| 1. Task definition and data | Audit Wi-Pose and WiMANS schemas, distributions, CSI and pose examples | Inventories, plots, synchronized examples and videos |
| 2. Preprocessing | Convert both datasets to the common input contract and prepare BODY-14 targets | Manifests, split audit, normalizer and processed examples |
| 3. Model and theory | Define PoseCNN, masked reconstruction, loss functions and metrics | Tensor checks, model summaries and formulas |
| 4. Condition A | Train PoseCNN only with labelled Wi-Pose | Epoch log, selected checkpoint and training curves |
| 5. Condition B | Pretrain the encoder with unlabelled CSI, then fine-tune with the same Wi-Pose labels | SSL and fine-tuning logs, checkpoints and curves |
| 6. Condition A evaluation | Test Condition A alone on the untouched Wi-Pose split | Aggregate and per-joint metrics, confusion matrix and inference video |
| 7. Comparative evaluation | Compare both conditions on Wi-Pose and held-out WiMANS | Tables, bootstrap intervals, sensitivity checks, skeleton plots and videos |

Each numbered experimental section starts with a restart cell that reloads saved artefacts. Sections 6 and 7 do not train, resume, select or overwrite a model.

## Evidence boundaries

- Wi-Pose provides the only pose supervision. Its labels are video-derived AlphaPose labels supplied with the dataset.
- Condition A sees no WiMANS data before inference.
- Condition B sees unlabelled WiMANS classroom and meeting-room CSI during masked reconstruction. It receives no WiMANS pose label.
- The 594 WiMANS empty-room recordings stay outside training and checkpoint selection.
- WiMANS video poses are AlphaPose pseudo-references used only for final evaluation.

## Final evaluation coverage

The Wi-Pose test partition contains 17,645 samples from 12 held-out participant groups. WiMANS evaluation covers all 594 held-out recordings. AlphaPose produced 52,684 valid BODY-14 frames from 53,460 video frames, giving 98.55% primary coverage.

Condition B reduced Wi-Pose NME from 0.206182 to 0.202268. On WiMANS it reduced pseudo-reference NME from 0.353495 to 0.336049. The WiMANS number measures agreement with a video estimator under approximate synchronization, not physical ground-truth pose accuracy.

## Media on GitHub

Jupyter displays animations locally through HTML5 video output. GitHub renders notebooks as static HTML and may suppress those embedded players. The repository therefore stores selected MP4 outputs under `results/evaluation/figures/`; readers can open them as separate files.
