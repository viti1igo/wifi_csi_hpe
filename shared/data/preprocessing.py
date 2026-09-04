from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def csi_amplitude(csi: np.ndarray) -> np.ndarray:
    """Convert complex CSI to float32 amplitude without changing axes."""
    return np.abs(csi).astype(np.float32, copy=False)


def reshape_csi_window(csi: np.ndarray) -> np.ndarray:
    """Convert either observed Wi-Pose layout into [links, time, subcarrier]."""
    if csi.shape == (5, 3, 3, 30):
        return csi.transpose(1, 2, 0, 3).reshape(9, 5, 30).astype(np.float32)
    if csi.shape == (5, 30, 3, 3):
        return csi.transpose(2, 3, 0, 1).reshape(9, 5, 30).astype(np.float32)
    raise ValueError(f"Expected CSI shape (5, 3, 3, 30) or (5, 30, 3, 3), received {csi.shape}")


@dataclass(frozen=True)
class CSINormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, training_samples: np.ndarray, epsilon: float = 1e-6) -> "CSINormalizer":
        if training_samples.ndim != 4 or training_samples.shape[1:] != (9, 5, 30):
            raise ValueError("Expected training samples shaped [N, 9, 5, 30]")
        mean = training_samples.mean(axis=(0, 2))
        std = training_samples.std(axis=(0, 2))
        return cls(mean.astype(np.float32), np.maximum(std, epsilon).astype(np.float32))

    def transform(self, sample: np.ndarray) -> np.ndarray:
        return ((sample - self.mean[:, None, :]) / self.std[:, None, :]).astype(np.float32)


def canonicalize_pose(joints: np.ndarray, confidence: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Hip-centre and torso-normalize a BODY_14 pose.

    Returns canonical coordinates and the original torso scale. Index 1 is neck;
    indices 8 and 11 are right and left hips.
    """
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape != (14, 2):
        raise ValueError(f"Expected joints shape (14, 2), received {joints.shape}")
    if confidence is not None and np.asarray(confidence).shape != (14,):
        raise ValueError("Confidence must have shape (14,)")
    hip_midpoint = (joints[8] + joints[11]) * 0.5
    torso = float(np.linalg.norm(joints[1] - hip_midpoint))
    if not np.isfinite(torso) or torso < 1e-6:
        raise ValueError("Cannot canonicalize a pose with zero or invalid torso length")
    return ((joints - hip_midpoint) / torso).astype(np.float32), torso
