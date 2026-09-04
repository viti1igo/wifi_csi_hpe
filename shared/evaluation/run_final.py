from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from shared.data.preprocessing import CSINormalizer
from shared.evaluation.final import (
    aligned_csi_windows, canonicalize_body14_sequence, metric_rows,
    paired_bootstrap, pck_auc, predict_loader, predict_windows,
    to_wipose_camera_frame, write_json,
)
from shared.training.setup import (
    build_source_components, build_ssl_components, build_wipose_loaders,
    load_trained_model,
)
from shared.utils.config import deep_merge, load_yaml


def load_evaluation(root: Path, device: torch.device):
    base = load_yaml(root / "shared/configs/base.yaml")
    configs = {
        "Condition A": deep_merge(base, load_yaml(root / "conditions/source_only_zero_shot/configs/condition.yaml")),
        "Condition B": deep_merge(base, load_yaml(root / "conditions/ssl_domain_adapted/configs/condition.yaml")),
    }
    split = root / "data/manifests/wipose_split_v1.jsonl"
    normalizer_path = root / "data/processed/normalization_wipose_train_v1.npz"
    with np.load(normalizer_path, allow_pickle=False) as saved:
        normalizer = CSINormalizer(saved["mean"], saved["std"])
        split_checksum = str(saved["split_manifest_sha256"])
    assert hashlib.sha256(split.read_bytes()).hexdigest() == split_checksum
    checks = {"split_checksum": split_checksum,
              "normalizer_checksum": hashlib.sha256(normalizer_path.read_bytes()).hexdigest()}
    models, checkpoints = {}, {}
    for name, config in configs.items():
        model = build_source_components(config)[0] if name == "Condition A" else build_ssl_components(config)[1]
        checkpoint_path = root / config["paths"]["result_dir"] / "checkpoints/best.pt"
        model, checkpoint = load_trained_model(
            model, checkpoint_path, config["condition"], checks, config)
        models[name], checkpoints[name] = model.to(device), checkpoint
    return configs, normalizer, checks, models, checkpoints


def evaluate_wipose(root: Path, device: torch.device) -> None:
    configs, normalizer, checks, models, checkpoints = load_evaluation(root, device)
    loader = build_wipose_loaders(root, normalizer.transform, 256, 0, partitions=("test",))["test"]
    all_rows = []
    for name, model in models.items():
        prediction, target, confidence, metadata = predict_loader(model, loader, device)
        rows = metric_rows(prediction, target, confidence, metadata)
        rows.insert(0, "condition", name)
        all_rows.append(rows)
        condition_dir = root / configs[name]["paths"]["result_dir"]
        rows.to_csv(condition_dir / "test_predictions_metrics.csv", index=False)
        np.savez_compressed(condition_dir / "test_predictions.npz", prediction=prediction,
                            target=target, confidence=confidence,
                            sample_id=np.array([item["sample_id"] for item in metadata]))
        summary = {
            "condition": configs[name]["condition"], "checkpoint_epoch": checkpoints[name]["epoch"],
            "samples": len(rows), "nme": float(rows.nme.mean()),
            "pck_0.05": float(rows["pck_0.05"].mean()), "pck_0.10": float(rows["pck_0.10"].mean()),
            "pck_0.20": float(rows["pck_0.20"].mean()), "pck_auc_0_0.50": pck_auc(rows),
            "bone_length_error": float(rows.bone_length_error.mean()),
            "symmetry_error": float(rows.symmetry_error.mean()), **checks,
        }
        write_json(condition_dir / "test_metrics.json", summary)
        print(name, summary)
    output = root / "results/evaluation"; output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output / "wipose_test_per_sample.csv", index=False)
    summary = combined.groupby("condition")[["nme", "pck_0.05", "pck_0.10", "pck_0.20",
                                               "bone_length_error", "symmetry_error"]].mean()
    summary["pck_auc_0_0.50"] = [pck_auc(combined[combined.condition == name]) for name in summary.index]
    summary.to_csv(output / "wipose_test_summary.csv")
    write_json(output / "wipose_paired_bootstrap.json", paired_bootstrap(combined, "participant"))


def evaluate_wimans(root: Path, device: torch.device) -> None:
    configs, normalizer, checks, models, checkpoints = load_evaluation(root, device)
    pseudo_dir = root / "data/pseudo_labels/wimans_alphapose"
    provenance = json.loads((pseudo_dir / "provenance.json").read_text())
    index = json.loads((pseudo_dir / "video_index.json").read_text())
    output = root / "results/evaluation"; output.mkdir(parents=True, exist_ok=True)
    primary_rows, sensitivity, temporal = [], [], []
    example_reference, example_a, example_b, example_ids = [], [], [], []
    started = time.perf_counter()
    for number, row in enumerate(index, 1):
        pseudo = np.load(pseudo_dir / f"{row['sample_id']}.npz", allow_pickle=False)
        csi = np.load(row["csi_path"], mmap_mode="r")
        for offset in (-2, -1, 0, 1, 2):
            windows = aligned_csi_windows(csi, 90, offset)
            predictions = {name: predict_windows(model, windows, normalizer.transform, device)
                           for name, model in models.items()}
            for threshold in (0.10, 0.30, 0.50):
                reference, _, _, valid = canonicalize_body14_sequence(
                    to_wipose_camera_frame(pseudo["body14_pixel"]),
                    pseudo["confidence"], threshold)
                metadata = [{"sample_id": row["sample_id"], "recording": row["sample_id"],
                             "frame": i, "activity": row["activity"], "location": row["location"],
                             "offset_frames": offset, "confidence_threshold": threshold}
                            for i in range(90)]
                threshold_rows = []
                for name, prediction in predictions.items():
                    chosen = np.flatnonzero(valid)
                    if not len(chosen):
                        continue
                    rows = metric_rows(prediction[chosen], reference[chosen], pseudo["confidence"][chosen],
                                       [metadata[i] for i in chosen], threshold)
                    rows.insert(0, "condition", name)
                    threshold_rows.append(rows)
                    if offset == 0 and threshold == 0.30:
                        primary_rows.append(rows)
                if offset == 0 and threshold == 0.30 and valid.any():
                    chosen = np.flatnonzero(valid)
                    example_reference.append(reference[chosen])
                    example_a.append(predictions["Condition A"][chosen])
                    example_b.append(predictions["Condition B"][chosen])
                    example_ids.extend(f"{row['sample_id']}:frame{i}" for i in chosen)
                    for name, prediction in predictions.items():
                        if len(chosen) > 1:
                            ref_velocity = np.linalg.norm(np.diff(reference[chosen], axis=0), axis=-1).mean()
                            pred_velocity = np.linalg.norm(np.diff(prediction[chosen], axis=0), axis=-1).mean()
                            temporal.append({"recording": row["sample_id"], "condition": name,
                                             "reference_velocity": float(ref_velocity),
                                             "prediction_velocity": float(pred_velocity),
                                             "velocity_error": float(abs(pred_velocity - ref_velocity)),
                                             "prediction_jitter": float(np.linalg.norm(np.diff(prediction[chosen], n=2, axis=0), axis=-1).mean()) if len(chosen) > 2 else np.nan})
                if threshold_rows:
                    joined = pd.concat(threshold_rows)
                    for name, group in joined.groupby("condition"):
                        sensitivity.append({"recording": row["sample_id"], "condition": name,
                                            "offset_frames": offset, "confidence_threshold": threshold,
                                            "valid_frames": int(len(group)), "nme": float(group.nme.mean()),
                                            "pck_0.10": float(group["pck_0.10"].mean())})
        if number % 10 == 0 or number == len(index):
            elapsed = time.perf_counter() - started
            print(f"{number}/{len(index)} recordings ({number / elapsed:.2f}/s, ETA {(len(index)-number)*elapsed/number/60:.1f} min)")
    primary = pd.concat(primary_rows, ignore_index=True)
    primary.to_csv(output / "wimans_pseudo_reference_per_frame.csv", index=False)
    sensitivity_frame = pd.DataFrame(sensitivity)
    sensitivity_frame.to_csv(output / "wimans_alignment_confidence_sensitivity.csv", index=False)
    pd.DataFrame(temporal).to_csv(output / "wimans_temporal_metrics.csv", index=False)
    np.savez_compressed(output / "wimans_qualitative_predictions.npz",
                       reference=np.concatenate(example_reference), condition_a=np.concatenate(example_a),
                       condition_b=np.concatenate(example_b), sample_frame=np.array(example_ids))
    summary = primary.groupby("condition")[["nme", "pck_0.05", "pck_0.10", "pck_0.20",
                                              "bone_length_error", "symmetry_error"]].mean()
    summary["pck_auc_0_0.50"] = [pck_auc(primary[primary.condition == name]) for name in summary.index]
    summary["valid_frames"] = primary.groupby("condition").size()
    summary["valid_recordings"] = primary.groupby("condition").recording.nunique()
    summary.to_csv(output / "wimans_pseudo_reference_summary.csv")
    write_json(output / "wimans_paired_bootstrap.json", paired_bootstrap(primary, "recording"))
    write_json(output / "evaluation_provenance.json", {
        "reference": provenance, "checkpoint_epochs": {k: v["epoch"] for k, v in checkpoints.items()},
        "checksums": checks, "alignment": "uniform duration, centred five-packet window",
        "offset_sensitivity_frames": [-2, -1, 0, 1, 2],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("wipose", "wimans", "all"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    if args.dataset in ("wipose", "all"):
        evaluate_wipose(args.root, device)
    if args.dataset in ("wimans", "all"):
        evaluate_wimans(args.root, device)


if __name__ == "__main__":
    main()
