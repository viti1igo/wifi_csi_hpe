from __future__ import annotations

BODY_14_JOINTS = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
)

BODY_14_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (8, 11),
)

SYMMETRIC_BONE_PAIRS = (
    ((1, 2), (1, 5)), ((2, 3), (5, 6)), ((3, 4), (6, 7)),
    ((1, 8), (1, 11)), ((8, 9), (11, 12)), ((9, 10), (12, 13)),
)

JOINT_INDEX = {name: index for index, name in enumerate(BODY_14_JOINTS)}

