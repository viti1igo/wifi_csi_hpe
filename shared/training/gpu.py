from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Callable, Iterator

import torch
import torch.distributed as dist


@contextmanager
def distributed_context() -> Iterator[tuple[int, int, torch.device]]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the planned training configuration")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    try:
        yield rank, world_size, torch.device("cuda", local_rank)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


@dataclass(frozen=True)
class BatchProbeResult:
    per_gpu_batch: int
    global_batch: int
    free_before: int
    total_memory: int


class AutoBatchSizer:
    """Find the largest safe per-GPU batch before committed training starts."""

    def __init__(self, multiple: int = 8, reserve_gib: float = 2.0, reserve_fraction: float = 0.10, max_batch: int = 8192) -> None:
        self.multiple = multiple
        self.reserve_gib = reserve_gib
        self.reserve_fraction = reserve_fraction
        self.max_batch = max_batch

    def probe(self, trial_step: Callable[[int], None], world_size: int = 1) -> BatchProbeResult:
        free, total = torch.cuda.mem_get_info()
        reserve = max(int(self.reserve_gib * 1024**3), int(total * self.reserve_fraction))
        if free <= reserve:
            raise RuntimeError("Insufficient free VRAM after applying the safety reserve")
        usable_increment = free - reserve

        def fits(batch_size: int) -> bool:
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            try:
                trial_step(batch_size)
                torch.cuda.synchronize()
                incremental_peak = torch.cuda.max_memory_allocated() - baseline
                return incremental_peak <= usable_increment
            except torch.cuda.OutOfMemoryError:
                return False
            finally:
                torch.cuda.empty_cache()

        successful = self.multiple
        candidate = self.multiple
        while candidate <= self.max_batch:
            if fits(candidate):
                successful = candidate
                candidate *= 2
            else:
                break
        low, high = successful, min(candidate - self.multiple, self.max_batch)
        while high - low >= self.multiple:
            midpoint = ((low + high) // (2 * self.multiple)) * self.multiple
            if fits(midpoint):
                low = midpoint
            else:
                high = midpoint - self.multiple
        safe = torch.tensor([low], device="cuda", dtype=torch.int64)
        if world_size > 1:
            dist.all_reduce(safe, op=dist.ReduceOp.MIN)
        batch = int(safe.item())
        return BatchProbeResult(batch, batch * world_size, free, total)
