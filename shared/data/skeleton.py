from __future__ import annotations

BODY_14_JOINTS = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
)

# Wi-Pose stores its labels in the standard 18-point AlphaPose/OpenPose order.
# This numbering is *not* the paper figure's independent 1--18 anatomical
# notation. Keep the source convention available for provenance and plotting.
WIPOSE_ALPHAPOSE_18_JOINTS = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)

# A name-based mapping makes the compatible 18 -> 14 selection explicit.  The
# target intentionally excludes eye and ear points, rather than silently treating
# an index in the paper illustration as the same source index.
WIPOSE_TO_BODY_14 = tuple(WIPOSE_ALPHAPOSE_18_JOINTS.index(name) for name in BODY_14_JOINTS)

WIPOSE_ALPHAPOSE_18_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17),
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
