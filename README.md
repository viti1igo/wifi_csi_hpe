# Cross-dataset WiFi-CSI Human Pose Estimation

Notebook-first Assignment 2 project comparing two transfer conditions:

1. `source_only_zero_shot`: supervised Wi-Pose training and frozen WiMANS inference.
2. `ssl_domain_adapted`: masked-CSI pretraining on permitted unlabeled data, followed by supervised Wi-Pose fine-tuning.

The final assignment narrative lives in `notebooks/24681189_notebook.ipynb`. Reusable implementation is kept in `shared/`; condition folders contain only condition-specific configuration, wrappers, documentation, logs, results, and supporting notebooks.

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
