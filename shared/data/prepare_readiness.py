"""Create leakage-safe split, normalization, and WiMANS readiness artefacts."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .preprocessing import csi_amplitude, reshape_csi_window
from .splits import grouped_split

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data/manifests"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_wipose_split() -> tuple[list[dict], dict]:
    rows = read_jsonl(MANIFESTS / "wipose_processed.jsonl")
    split = grouped_split([row["participant"] for row in rows], seed=31050)
    membership = {
        **{group: "train" for group in split.train_groups},
        **{group: "monitor" for group in split.monitor_groups},
        **{group: "test" for group in split.test_groups},
    }
    output = [{**row, "csi_path": row["processed_path"], "partition": membership[row["participant"]]} for row in rows]
    groups = {name: sorted(group for group, part in membership.items() if part == name) for name in ("train", "monitor", "test")}
    assert not (set(groups["train"]) & set(groups["monitor"]) | set(groups["train"]) & set(groups["test"]) | set(groups["monitor"]) & set(groups["test"]))
    summary = {
        "seed": 31050, "group_field": "participant", "requested_ratios": [0.8, 0.1, 0.1],
        "groups": groups,
        "sample_counts": Counter(row["partition"] for row in output),
        "action_counts": {part: Counter(row["activity"] for row in output if row["partition"] == part) for part in groups},
    }
    return output, summary


def build_wimans_partitions() -> tuple[list[dict], dict]:
    source = ROOT / "data/extracted/wimans"
    rows = []
    with (source / "annotation.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["wifi_band"]) != 5.0 or int(row["number_of_users"]) != 1:
                continue
            role = "zero_shot_test" if row["environment"] == "empty_room" else "ssl_train"
            path = source / "wifi_csi/amp" / f'{row["label"]}.npy'
            if not path.exists():
                raise FileNotFoundError(path)
            user = next(i for i in range(1, 7) if row[f"user_{i}_activity"])
            rows.append({
                "sample_id": row["label"], "csi_path": str(path), "partition": role,
                "environment": row["environment"], "wifi_band": 5.0, "number_of_users": 1,
                "user_slot": user, "location": row[f"user_{user}_location"],
                "activity": row[f"user_{user}_activity"],
                "pose_labels_available": False,
            })
    summary = {
        "selection": {"wifi_band": 5.0, "number_of_users": 1},
        "partition_counts": Counter(row["partition"] for row in rows),
        "environment_counts": Counter(row["environment"] for row in rows),
        "activity_counts": Counter(row["activity"] for row in rows),
    }
    assert {row["environment"] for row in rows if row["partition"] == "zero_shot_test"} == {"empty_room"}
    assert all(row["activity"] and row["location"] for row in rows)
    assert not {row["sample_id"] for row in rows if row["partition"] == "ssl_train"} & {row["sample_id"] for row in rows if row["partition"] == "zero_shot_test"}
    return rows, summary


def fit_normalizer(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros((9, 30), dtype=np.float64)
    squared = np.zeros_like(total)
    count = 0
    training = [row for row in rows if row["partition"] == "train"]
    for index, row in enumerate(training, 1):
        with np.load(row["processed_path"], allow_pickle=False) as payload:
            sample = reshape_csi_window(csi_amplitude(payload["csi"]))
        total += sample.sum(axis=1)
        squared += np.square(sample, dtype=np.float64).sum(axis=1)
        count += sample.shape[1]
        if index % 5000 == 0:
            print(f"normalization: {index}/{len(training)} training samples", flush=True)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), len(training)


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    wipose, split_summary = build_wipose_split()
    split_path = MANIFESTS / "wipose_split_v1.jsonl"
    write_jsonl(split_path, wipose)
    write_json(MANIFESTS / "split_v1.json", {**split_summary, "manifest_sha256": sha256(split_path)})

    wimans, wimans_summary = build_wimans_partitions()
    wimans_path = MANIFESTS / "wimans_partitions_v1.jsonl"
    write_jsonl(wimans_path, wimans)

    mean, std, fitted_samples = fit_normalizer(wipose)
    normalizer_path = ROOT / "data/processed/normalization_wipose_train_v1.npz"
    normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(normalizer_path, mean=mean, std=std, fitted_samples=fitted_samples, split_manifest_sha256=sha256(split_path))

    with np.load(wipose[0]["processed_path"], allow_pickle=False) as payload:
        sample = reshape_csi_window(csi_amplitude(payload["csi"]))
    normalized = (sample - mean[:, None, :]) / std[:, None, :]
    assert sample.shape == normalized.shape == (9, 5, 30) and np.isfinite(normalized).all()
    write_json(MANIFESTS / "quality_report_v1.json", {
        "wipose": {**split_summary, "normalizer_fitted_samples": fitted_samples,
                    "normalizer_shape": list(mean.shape), "model_input_shape": list(sample.shape)},
        "wimans": wimans_summary,
        "checks": {"split_overlap": False, "wimans_partition_overlap": False,
                   "normalizer_train_only": True, "loader_shape_and_finite": True},
    })
    print(f"ready: Wi-Pose={len(wipose)}, WiMANS={len(wimans)}", flush=True)


if __name__ == "__main__":
    main()
