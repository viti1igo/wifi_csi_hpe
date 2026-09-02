"""Convert Wi-Pose MATLAB files to traceable NumPy shards and a processed manifest.

The raw AlphaPose layout is retained as 18 named joints.  The 14-joint training
target is selected through a named mapping, never through a paper-figure index.
"""
from pathlib import Path
import argparse
import json, re
import h5py
import numpy as np
from shared.data.skeleton import WIPOSE_TO_BODY_14

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/extracted/wipose/Wi-Pose'
OUT = ROOT / 'data/processed/wipose'

def _is_current(path: Path) -> bool:
    """Return whether an existing shard satisfies the current output contract."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {
                'csi', 'joints', 'joints_pixel', 'confidence', 'joints_18_pixel',
                'confidence_18', 'joint_source_indices',
            }.issubset(payload.files)
    except (OSError, ValueError):
        return False


def main(*, resume: bool = True, limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    written = reused = 0
    for split in ('Train', 'Test'):
        for src in sorted((SRC / split).glob('*.mat')):
            if limit is not None and written + reused >= limit:
                break
            out = OUT / split / (src.stem + '.npz')
            activity = re.match(r'(.+?)_\d+-frame\d+', src.stem)
            row = {'sample_id': src.stem, 'processed_path': str(out),
                   'source_path': str(src), 'participant': src.stem.split('_')[-1].split('-')[0],
                   'activity': activity.group(1) if activity else 'unknown', 'split': split,
                   'pose_source': 'AlphaPose_18',
                   'target_joint_source_indices': list(WIPOSE_TO_BODY_14),
                   'csi_shape': [5, 3, 3, 30], 'target_shape': [14, 2]}
            if resume and _is_current(out):
                reused += 1
                rows.append(row)
                continue
            with h5py.File(src, 'r') as f:
                csi = np.asarray(f['CSI'][()], dtype=np.float32).transpose(3, 0, 1, 2)
                points = np.asarray(f['SkeletonPoints'][()], dtype=np.float32).reshape(3, 18)
            # Wi-Pose AlphaPose/OpenPose layout: 18 x coordinates, 18 y coordinates,
            # then 18 confidences.  Camera rotation is a visualization concern only.
            joints_18_pixel = points[:2, :].T.copy()
            confidence_18 = points[2, :].copy()
            joints_pixel = joints_18_pixel[list(WIPOSE_TO_BODY_14)].copy()
            confidence = confidence_18[list(WIPOSE_TO_BODY_14)].copy()
            hip_midpoint = (joints_pixel[8] + joints_pixel[11]) * 0.5
            torso = max(float(np.linalg.norm(joints_pixel[1] - hip_midpoint)), 1.0)
            joints = ((joints_pixel - hip_midpoint) / torso).astype(np.float32)
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_suffix('.tmp.npz')
            np.savez_compressed(
                temporary, csi=csi, joints=joints, joints_pixel=joints_pixel,
                confidence=confidence, joints_18_pixel=joints_18_pixel,
                confidence_18=confidence_18,
                joint_source_indices=np.asarray(WIPOSE_TO_BODY_14, dtype=np.int16),
            )
            temporary.replace(out)
            written += 1
            row['csi_shape'] = list(csi.shape)
            rows.append(row)
            if (written + reused) % 1000 == 0:
                print(f'progress: {written + reused} checked; {written} rebuilt; {reused} reused', flush=True)
        if limit is not None and written + reused >= limit:
            break
    manifest = ROOT / 'data/manifests/wipose_processed.jsonl'
    temporary_manifest = manifest.with_suffix('.tmp.jsonl')
    with temporary_manifest.open('w') as f:
        for row in rows: f.write(json.dumps(row) + '\n')
    temporary_manifest.replace(manifest)
    print(f'processed {len(rows)} Wi-Pose samples; rebuilt={written}; reused={reused}', flush=True)

if __name__ == '__main__': main()
