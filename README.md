<<<<<<< HEAD
# wifi_csi_hpe
=======
# WiFi-CSI Human Pose Estimation

Notebook-first Assignment 2 project comparing two transfer conditions:

1. `source_only_zero_shot`: supervised Wi-Pose training and frozen WiMANS inference.
2. `ssl_domain_adapted`: masked-CSI pretraining on permitted unlabeled data, followed by supervised Wi-Pose fine-tuning.

The final assignment narrative lives in `notebooks/WiFi_CSI_HPE_A2.ipynb`. Reusable implementation is kept in `shared/`; condition folders contain only condition-specific configuration, wrappers, documentation, logs, results, and supporting notebooks.

## Current milestone

Only scaffolding, static validation, and raw dataset acquisition are complete. Do not extract, explore, preprocess, probe GPUs, or train until the task trackers are reviewed.

## Intended runtime

- Python 3.11+
- PyTorch with NCCL
- Two NVIDIA GPUs, 24 GiB each
- Launch training later with `torchrun --standalone --nproc_per_node=2 ...`

The requested environment is named `ML_31005`. Its reproducible specification is
stored in `environment.yml`.

## Data sources

- Wi-Pose: https://github.com/NjtechCVLab/Wi-PoseDataset
- WiMANS: https://www.kaggle.com/datasets/shuokanghuang/wimans

Downloaded archives and binaries are excluded from Git. Download provenance and SHA-256 hashes are written to `data/manifests/`.
>>>>>>> 412253b (Add WiFi-CSI HPE assignment scaffold and notebooks.)
