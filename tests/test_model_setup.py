import json

import numpy as np
import pytest
import torch

from shared.data.preprocessing import CSINormalizer
from shared.models import PoseCNN
from shared.training import CheckpointManager, EpochLogger, SkeletonAwareLoss, Trainer
from shared.training.setup import build_wipose_loaders, load_trained_model, resume_training


def test_model_setup_contract(tmp_path):
    manifest = tmp_path / "data/manifests/wipose_split_v1.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = []
    for index, partition in enumerate(("train", "monitor", "test")):
        sample = tmp_path / f"{partition}.npz"
        np.savez(
            sample,
            csi=np.ones((5, 3, 3, 30), dtype=np.float32) * (index + 1),
            joints=np.zeros((14, 2), dtype=np.float32),
            confidence=np.ones(14, dtype=np.float32),
        )
        rows.append({"csi_path": str(sample), "participant": str(index), "partition": partition})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    normalizer = CSINormalizer(np.zeros((9, 30), np.float32), np.ones((9, 30), np.float32))
    loaders = build_wipose_loaders(tmp_path, normalizer.transform, preview_batch=1)
    assert set(loaders) == {"train", "monitor", "test"}
    assert next(iter(loaders["train"]))["csi"].shape == (1, 9, 5, 30)

    model = PoseCNN()
    checkpoint = {
        "epoch": 1,
        "condition": "source_only_zero_shot",
        "model": model.state_dict(),
        "optimizer": {},
        "scaler": {},
        "best_monitor_nme": 0.5,
        "epochs_without_improvement": 0,
        "config": {"model": {"channels": [32, 64, 128, 192], "dropout": 0.2}},
        "split_checksum": "split",
        "normalizer_checksum": "normalizer",
    }
    path = tmp_path / "best.pt"
    torch.save(checkpoint, path)

    loaded, metadata = load_trained_model(
        PoseCNN(), path, "source_only_zero_shot",
        {"split_checksum": "split", "normalizer_checksum": "normalizer"},
    )
    assert not loaded.training and metadata["epoch"] == 1
    with pytest.raises(ValueError, match="split_checksum"):
        load_trained_model(PoseCNN(), path, "source_only_zero_shot", {"split_checksum": "wrong"})
    with pytest.raises(ValueError, match="Expected condition"):
        load_trained_model(PoseCNN(), path, "ssl_domain_adapted", {})


def test_training_persistence_and_resume(tmp_path):
    model = PoseCNN(channels=(4, 4, 4, 4), dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, SkeletonAwareLoss(), torch.device("cpu"))
    batch = {
        "csi": torch.randn(2, 9, 5, 30),
        "joints": torch.randn(2, 14, 2),
        "confidence": torch.ones(2, 14),
    }
    config = {"model": {"channels": [4, 4, 4, 4], "dropout": 0.0}}
    manager = CheckpointManager(tmp_path / "results")
    logger = EpochLogger(tmp_path / "logs")
    best = float("inf")
    for epoch in (1, 2):
        train_metrics = trainer.run_epoch([batch], training=True)
        monitor_metrics = trainer.run_epoch([batch], training=False)
        assert all(np.isfinite(value) for value in {**train_metrics, **monitor_metrics}.values())
        best = min(best, monitor_metrics["nme"])
        state = {
            "epoch": epoch, "condition": "source_only_zero_shot", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": trainer.scaler.state_dict(),
            "best_monitor_nme": best, "epochs_without_improvement": 0,
            "config": config, "split_checksum": "split", "normalizer_checksum": "normalizer",
        }
        manager.save(state, monitor_metrics["nme"])
        logger.log({"epoch": epoch, "monitor_nme": monitor_metrics["nme"]})
    assert sum(1 for _ in logger.jsonl_path.open()) == 2

    resumed_model = PoseCNN(channels=(4, 4, 4, 4), dropout=0.0)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_trainer = Trainer(resumed_model, resumed_optimizer, SkeletonAwareLoss(), torch.device("cpu"))
    payload = resume_training(
        resumed_model, resumed_optimizer, resumed_trainer.scaler,
        tmp_path / "results/checkpoints/last.pt", "source_only_zero_shot", config,
        {"split_checksum": "split", "normalizer_checksum": "normalizer"},
    )
    assert payload["epoch"] + 1 == 3
    with pytest.raises(ValueError, match="model configuration"):
        resume_training(
            resumed_model, resumed_optimizer, resumed_trainer.scaler,
            tmp_path / "results/checkpoints/last.pt", "source_only_zero_shot",
            {"model": {"channels": [8, 8, 8, 8]}},
            {"split_checksum": "split", "normalizer_checksum": "normalizer"},
        )
