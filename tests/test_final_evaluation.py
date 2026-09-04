import numpy as np
import pandas as pd

from shared.evaluation.final import (
    aligned_csi_windows, canonicalize_body14_sequence, coco17_to_body14,
    paired_bootstrap, to_wipose_camera_frame,
)


def test_coco_mapping_canonicalization_and_alignment():
    coco = np.column_stack([np.arange(17), np.arange(17) * 2, np.ones(17)]).astype(np.float32)
    body, confidence = coco17_to_body14(coco)
    assert body.shape == (14, 2) and confidence.shape == (14,)
    assert np.allclose(body[1], (coco[5, :2] + coco[6, :2]) / 2)
    canonical, torso, hips, valid = canonicalize_body14_sequence(body[None], confidence[None])
    assert valid[0] and np.isfinite(canonical).all() and torso[0] > 0 and hips.shape == (1, 2)
    csi = np.arange(101 * 3 * 3 * 30, dtype=np.float32).reshape(101, 3, 3, 30)
    windows = aligned_csi_windows(csi, 9)
    assert windows.shape == (9, 9, 5, 30) and np.isfinite(windows).all()


def test_wimans_coordinates_rotate_into_wipose_camera_frame():
    joints = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    assert np.array_equal(to_wipose_camera_frame(joints), [[20.0, -10.0], [40.0, -30.0]])


def test_paired_bootstrap_is_deterministic():
    rows = pd.DataFrame({"recording": ["a", "a", "b", "b"],
                         "condition": ["Condition A", "Condition B"] * 2,
                         "nme": [0.3, 0.2, 0.4, 0.5]})
    assert paired_bootstrap(rows, "recording", resamples=100) == paired_bootstrap(rows, "recording", resamples=100)
