from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class GroupSplit:
    train: np.ndarray
    monitor: np.ndarray
    test: np.ndarray
    train_groups: tuple[Hashable, ...]
    monitor_groups: tuple[Hashable, ...]
    test_groups: tuple[Hashable, ...]

    @property
    def achieved_ratios(self) -> tuple[float, float, float]:
        total = len(self.train) + len(self.monitor) + len(self.test)
        return (len(self.train) / total, len(self.monitor) / total, len(self.test) / total)


def grouped_split(
    groups: Sequence[Hashable],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 31050,
) -> GroupSplit:
    """Deterministically split whole groups, approximating requested ratios."""
    if len(groups) < 3:
        raise ValueError("At least three samples and three groups are required")
    ratios_array = np.asarray(ratios, dtype=float)
    if np.any(ratios_array <= 0) or not np.isclose(ratios_array.sum(), 1.0):
        raise ValueError("Split ratios must be positive and sum to one")
    unique = np.asarray(sorted(set(groups), key=str), dtype=object)
    if len(unique) < 3:
        raise ValueError("At least three distinct groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    raw_counts = ratios_array * len(unique)
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > len(unique):
        candidates = np.where(counts > 1)[0]
        counts[candidates[np.argmax(counts[candidates] - raw_counts[candidates])]] -= 1
    while counts.sum() < len(unique):
        counts[np.argmax(raw_counts - counts)] += 1
    n_train, n_monitor, _ = counts.tolist()
    train_groups = tuple(unique[:n_train])
    monitor_groups = tuple(unique[n_train:n_train + n_monitor])
    test_groups = tuple(unique[n_train + n_monitor:])
    group_array = np.asarray(groups, dtype=object)
    indices = np.arange(len(groups))
    return GroupSplit(
        train=indices[np.isin(group_array, train_groups)],
        monitor=indices[np.isin(group_array, monitor_groups)],
        test=indices[np.isin(group_array, test_groups)],
        train_groups=train_groups,
        monitor_groups=monitor_groups,
        test_groups=test_groups,
    )

