from __future__ import annotations

from shared.models import PoseCNN


class SourceOnlyPoseModel(PoseCNN):
    """PoseCNN initialized and supervised only by Wi-Pose."""

