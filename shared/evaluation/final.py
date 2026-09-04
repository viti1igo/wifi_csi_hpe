from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from shared.data.preprocessing import canonicalize_pose, reshape_csi_window
from shared.data.skeleton import BODY_14_EDGES, BODY_14_JOINTS, SYMMETRIC_BONE_PAIRS

COCO_17_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
COCO_INDEX = {name: i for i, name in enumerate(COCO_17_JOINTS)}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coco17_to_body14(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert AlphaPose COCO-17 ``[17,3]`` output to project BODY-14."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (17, 3):
        raise ValueError(f"Expected [17,3] COCO keypoints, received {keypoints.shape}")
    points, confidence = keypoints[:, :2], keypoints[:, 2]
    out = np.empty((14, 2), np.float32)
    score = np.empty(14, np.float32)
    for i, name in enumerate(BODY_14_JOINTS):
        if name == "neck":
            left, right = COCO_INDEX["left_shoulder"], COCO_INDEX["right_shoulder"]
            out[i] = (points[left] + points[right]) / 2
            score[i] = min(confidence[left], confidence[right])
        else:
            j = COCO_INDEX[name]
            out[i], score[i] = points[j], confidence[j]
    return out, score


def canonicalize_body14_sequence(
    joints: np.ndarray, confidence: np.ndarray, threshold: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize frames with confident hips/neck and at least ten joints."""
    canonical = np.full_like(joints, np.nan, dtype=np.float32)
    torso = np.full(len(joints), np.nan, dtype=np.float32)
    hip_midpoint = np.full((len(joints), 2), np.nan, dtype=np.float32)
    valid = np.zeros(len(joints), dtype=bool)
    for i, (pose, score) in enumerate(zip(joints, confidence)):
        visible = score >= threshold
        if visible.sum() < 10 or not visible[[1, 8, 11]].all():
            continue
        try:
            canonical[i], torso[i] = canonicalize_pose(pose, score)
        except ValueError:
            continue
        hip_midpoint[i] = (pose[8] + pose[11]) / 2
        valid[i] = np.isfinite(canonical[i]).all()
    return canonical, torso, hip_midpoint, valid


def to_wipose_camera_frame(joints: np.ndarray) -> np.ndarray:
    """Rotate conventional image coordinates into Wi-Pose's stored camera frame."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape[-1] != 2:
        raise ValueError(f"Expected a final coordinate dimension of 2, received {joints.shape}")
    return np.stack((joints[..., 1], -joints[..., 0]), axis=-1)


def aligned_csi_windows(csi: np.ndarray, frame_count: int, offset_frames: int = 0) -> np.ndarray:
    """Uniformly align frame centres to centred five-packet CSI windows."""
    csi = np.asarray(csi)
    if csi.ndim != 4 or csi.shape[1:] != (3, 3, 30) or len(csi) < 5:
        raise ValueError(f"Expected WiMANS [packets,3,3,30], received {csi.shape}")
    centres = ((np.arange(frame_count) + 0.5) * len(csi) / frame_count - 0.5).round().astype(int)
    centres += np.rint(offset_frames * len(csi) / frame_count).astype(int)
    centres = np.clip(centres, 2, len(csi) - 3)
    return np.stack([reshape_csi_window(csi[c - 2:c + 3]) for c in centres])


def metric_rows(
    prediction: np.ndarray, target: np.ndarray, confidence: np.ndarray,
    metadata: Iterable[dict], threshold: float = 0.0,
) -> pd.DataFrame:
    """One row per pose with aggregate metrics and per-joint distances."""
    prediction, target, confidence = map(np.asarray, (prediction, target, confidence))
    distances = np.linalg.norm(prediction - target, axis=-1)
    valid = confidence >= threshold
    rows = []
    for i, meta in enumerate(metadata):
        keep = valid[i] & np.isfinite(distances[i])
        if not keep.any():
            continue
        pred_lengths = np.array([np.linalg.norm(prediction[i, b] - prediction[i, a]) for a, b in BODY_14_EDGES])
        true_lengths = np.array([np.linalg.norm(target[i, b] - target[i, a]) for a, b in BODY_14_EDGES])
        sym = [abs(np.linalg.norm(prediction[i, b] - prediction[i, a]) - np.linalg.norm(prediction[i, d] - prediction[i, c])) for (a, b), (c, d) in SYMMETRIC_BONE_PAIRS]
        row = dict(meta)
        row.update(nme=float(distances[i, keep].mean()),
                   bone_length_error=float(np.abs(pred_lengths - true_lengths).mean()),
                   symmetry_error=float(np.mean(sym)), valid_joints=int(keep.sum()),
                   mean_reference_confidence=float(confidence[i, keep].mean()))
        for cutoff in (0.05, 0.10, 0.20):
            row[f"pck_{cutoff:.2f}"] = float((distances[i, keep] <= cutoff).mean())
        for j, name in enumerate(BODY_14_JOINTS):
            row[f"joint_{name}"] = float(distances[i, j]) if keep[j] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pck_auc(rows: pd.DataFrame) -> float:
    values = rows.filter(regex=r"^joint_").to_numpy().ravel()
    values = values[np.isfinite(values)]
    thresholds = np.arange(0, 0.501, 0.01)
    return float(np.trapz([(values <= t).mean() for t in thresholds], thresholds) / 0.5)


def paired_bootstrap(
    rows: pd.DataFrame, group: str, metric: str = "nme", resamples: int = 10_000,
    seed: int = 31050,
) -> dict[str, float]:
    pivot = rows.pivot_table(index=group, columns="condition", values=metric, aggfunc="mean").dropna()
    delta = (pivot["Condition B"] - pivot["Condition A"]).to_numpy()
    if not len(delta):
        raise ValueError("No paired groups are available")
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"groups": int(len(delta)), "mean_delta_b_minus_a": float(delta.mean()),
            "ci95_low": float(low), "ci95_high": float(high),
            "condition_b_win_rate": float((delta < 0).mean()), "resamples": resamples, "seed": seed}


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@torch.inference_mode()
def predict_loader(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    predictions, targets, confidences, metadata = [], [], [], []
    model.eval()
    for batch in loader:
        predictions.append(model(batch["csi"].to(device)).cpu().numpy())
        targets.append(batch["joints"].numpy())
        confidences.append(batch["confidence"].numpy())
        for participant, sample_id in zip(batch["participant"], batch["sample_id"]):
            sample_id = str(sample_id)
            metadata.append({"participant": str(participant), "sample_id": sample_id,
                             "recording": sample_id.split("-frame")[0],
                             "activity": sample_id.split("_")[0]})
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(confidences), metadata


@torch.inference_mode()
def predict_windows(
    model: torch.nn.Module, windows: np.ndarray, normalizer, device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    output = []
    model.eval()
    for start in range(0, len(windows), batch_size):
        normalized = np.stack([normalizer(x) for x in windows[start:start + batch_size]])
        output.append(model(torch.from_numpy(normalized).to(device)).cpu().numpy())
    return np.concatenate(output)
