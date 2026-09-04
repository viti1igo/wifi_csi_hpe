"""Prepare and convert AlphaPose outputs for the held-out WiMANS videos.

AlphaPose inference itself stays in its isolated environment.  This module creates
one concatenated video (so the model loads once) and converts its COCO-17 JSON to
one resumable, provenance-rich NPZ per original recording.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from shared.evaluation.final import (
    canonicalize_body14_sequence, coco17_to_body14, sha256,
    to_wipose_camera_frame, write_json,
)


def heldout_rows(root: Path) -> list[dict]:
    path = root / "data/manifests/wimans_partitions_v1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row["partition"] == "zero_shot_test"]


def prepare(root: Path, output: Path) -> None:
    rows = heldout_rows(root)
    assert len(rows) == 594
    output.mkdir(parents=True, exist_ok=True)
    video_dir = root / "data/extracted/wimans/video"
    videos = [video_dir / f"{row['sample_id']}.mp4" for row in rows]
    missing = [str(path) for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} videos; first: {missing[0]}")
    concat_list = output / "heldout_concat.txt"
    concat_list.write_text("".join(f"file '{path.resolve()}'\n" for path in videos))
    index = [{**row, "video_path": str(video.resolve()), "start_frame": i * 90,
              "stop_frame": (i + 1) * 90, "frame_count": 90}
             for i, (row, video) in enumerate(zip(rows, videos))]
    (output / "video_index.json").write_text(json.dumps(index, indent=2) + "\n")
    combined = output / "heldout_594.mp4"
    if not combined.exists():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-f", "concat",
                        "-safe", "0", "-i", str(concat_list), "-c", "copy", str(combined)], check=True)
    print(f"Prepared {len(rows)} videos / {len(rows) * 90:,} frames: {combined}")


def convert(root: Path, output: Path, raw_json: Path, alphapose_root: Path, force: bool = False) -> None:
    index = json.loads((output / "video_index.json").read_text())
    detections = json.loads(raw_json.read_text())
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for item in detections:
        stem = Path(str(item["image_id"])).stem
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            by_frame[int(digits)].append(item)
    frame_base = min(by_frame) if by_frame else 0
    completed, valid_frames = 0, 0
    started = time.perf_counter()
    for row in index:
        target = output / f"{row['sample_id']}.npz"
        if target.exists() and not force:
            with np.load(target, allow_pickle=False) as saved:
                valid_frames += int(saved["valid_primary"].sum())
            completed += 1
            continue
        raw = np.full((90, 17, 3), np.nan, np.float32)
        person_score = np.zeros(90, np.float32)
        for local in range(90):
            candidates = by_frame.get(frame_base + row["start_frame"] + local, [])
            if candidates:
                best = max(candidates, key=lambda item: float(item.get("score", 0)))
                raw[local] = np.asarray(best["keypoints"], np.float32).reshape(17, 3)
                person_score[local] = float(best.get("score", 0))
        body = np.full((90, 14, 2), np.nan, np.float32)
        confidence = np.zeros((90, 14), np.float32)
        for i in range(90):
            if np.isfinite(raw[i]).all():
                body[i], confidence[i] = coco17_to_body14(raw[i])
        model_frame = to_wipose_camera_frame(body)
        canonical, torso, hips, valid = canonicalize_body14_sequence(model_frame, confidence, 0.30)
        valid_frames += int(valid.sum())
        np.savez_compressed(
            target, raw_coco17=raw, body14_pixel=body,
            body14_wipose_camera=model_frame, confidence=confidence,
            body14_canonical=canonical, frame_index=np.arange(90),
            frame_time_seconds=(np.arange(90) + 0.5) / 29.97,
            hip_midpoint_pixel=hips, torso_scale_pixel=torso, person_score=person_score,
            valid_primary=valid, sample_id=row["sample_id"], activity=row["activity"],
            location=row["location"], environment=row["environment"],
        )
        completed += 1
        if completed % 25 == 0 or completed == len(index):
            elapsed = time.perf_counter() - started
            print(f"{completed}/{len(index)} NPZ files, {completed / max(elapsed, 1e-6):.1f}/s")
    provenance = {
        "label": "AlphaPose video-derived pseudo-reference (not physical ground truth)",
        "recordings": len(index), "frames": len(index) * 90, "valid_primary_frames": valid_frames,
        "confidence_primary": 0.30, "confidence_sensitivity": [0.10, 0.50],
        "coordinate_contract": "AlphaPose image (x,y) rotated to Wi-Pose stored camera frame (y,-x) before canonicalization",
        "alphapose_git_commit": subprocess.check_output(
            ["git", "-C", str(alphapose_root), "rev-parse", "HEAD"], text=True).strip(),
        "alphapose_local_patch": "torchvision NMS/RoIAlign fallback; host has no nvcc",
        "config": "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
        "detector": "yolov3-spp", "raw_json_sha256": sha256(raw_json),
        "pose_checkpoint_sha256": sha256(alphapose_root / "pretrained_models/fast_res50_256x192.pth"),
        "detector_checkpoint_sha256": sha256(alphapose_root / "detector/yolo/data/yolov3-spp.weights"),
        "heldout_manifest_sha256": sha256(root / "data/manifests/wimans_partitions_v1.jsonl"),
    }
    write_json(output / "provenance.json", provenance)
    print(f"Converted {completed} recordings; provenance: {output / 'provenance.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "convert"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-json", type=Path)
    parser.add_argument("--alphapose-root", type=Path, default=Path("/data/axnguyen/tools/AlphaPose"))
    parser.add_argument("--force", action="store_true", help="rebuild existing per-recording NPZ files")
    args = parser.parse_args()
    output = args.output or args.root / "data/pseudo_labels/wimans_alphapose"
    if args.command == "prepare":
        prepare(args.root, output)
    elif args.raw_json is None:
        parser.error("convert requires --raw-json")
    else:
        convert(args.root, output, args.raw_json, args.alphapose_root, args.force)


if __name__ == "__main__":
    main()
