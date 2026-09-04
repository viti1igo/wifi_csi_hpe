import tempfile
from pathlib import Path

import numpy as np
import torch

from conditions.ssl_domain_adapted.models import DomainAdaptedPoseModel
from shared.data.preprocessing import CSINormalizer
from shared.models import MaskedCSIAutoencoder
from shared.training import CheckpointManager, EpochLogger, MaskedReconstructionTrainer, build_ssl_loaders, resume_pretraining


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_ssl_loader_and_strict_transfer():
    normalizer = CSINormalizer(np.zeros((9, 30), np.float32), np.ones((9, 30), np.float32))
    loaders, summary = build_ssl_loaders(ROOT, normalizer.transform, batch_size=2)
    dataset = loaders["train"].dataset
    assert len(dataset) == 2 * summary["wimans_train_recordings"]
    assert summary["excluded_empty_room_recordings"] == 594
    first_pair = [dataset[0], dataset[1]]
    assert {sample["source"] for sample in first_pair} == {"wipose", "wimans"}
    assert all(set(sample) == {"csi", "source", "sample_id"} for sample in first_pair)
    assert all(sample["csi"].shape == (9, 5, 30) for sample in first_pair)

    autoencoder = MaskedCSIAutoencoder()
    reconstructed, mask = autoencoder(torch.randn(2, 9, 5, 30))
    assert reconstructed.shape == (2, 9, 5, 30) and mask.shape == (2, 1, 5, 30)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best.pt"
        torch.save({"model": autoencoder.state_dict()}, path)
        pose = DomainAdaptedPoseModel()
        transferred = pose.load_pretrained_encoder(path)
    assert transferred == list(pose.encoder.state_dict())


def test_masked_reconstruction_cpu_epoch():
    model = MaskedCSIAutoencoder(channels=(4, 4, 4, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = MaskedReconstructionTrainer(model, optimizer, torch.device("cpu"), 0.3)
    loader = [{"csi": torch.randn(2, 9, 5, 30)}]
    assert np.isfinite(trainer.run_epoch(loader, training=True)["masked_mse"])
    assert np.isfinite(trainer.run_epoch(loader, training=False)["masked_mse"])

    config = {"model": {"channels": [4, 4, 4, 4]}}
    checks = {"source_manifest_checksum": "source", "split_checksum": "split", "normalizer_checksum": "normalizer"}
    with tempfile.TemporaryDirectory() as directory:
        saver = CheckpointManager(directory, directory_name="pretraining_checkpoints")
        logger = EpochLogger(directory, "pretraining_epochs")
        for epoch in (1, 2):
            state = {"epoch": epoch, "condition": "ssl_domain_adapted", "stage": "pretraining",
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": trainer.scaler.state_dict(),
                     "best_monitor_reconstruction": 1 / epoch, "epochs_without_improvement": 0, "config": config, **checks}
            saver.save(state, 1 / epoch); logger.log({"epoch": epoch, "monitor_masked_mse": 1 / epoch})
        restored = resume_pretraining(model, optimizer, trainer.scaler, saver.directory / "last.pt", config, checks)
        assert restored["epoch"] + 1 == 3 and sum(1 for _ in logger.jsonl_path.open()) == 2


def test_masked_loss_ignores_visible_positions():
    class FixedReconstruction(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        def forward(self, x, ratio):
            mask = torch.zeros_like(x[:, :1], dtype=torch.bool); mask[..., 0, 0] = True
            return x + (~mask) * 100 + mask * 2 + self.anchor * 0, mask

    model = FixedReconstruction()
    trainer = MaskedReconstructionTrainer(model, torch.optim.SGD(model.parameters(), 0.1), torch.device("cpu"), 0.3)
    assert trainer.run_epoch([{"csi": torch.zeros(1, 9, 5, 30)}], False)["masked_mse"] == 4.0
