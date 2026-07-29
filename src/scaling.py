from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllocationResult:
    admitted_mask: np.ndarray
    state_capacity: int
    admitted_flows: int
    unresolved_flows: int


def sample_indices(source_count: int, requested_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if requested_count <= source_count:
        return rng.choice(source_count, requested_count, replace=False)
    repeats, remainder = divmod(requested_count, source_count)
    indices = np.tile(np.arange(source_count, dtype=np.int64), repeats)
    if remainder:
        indices = np.concatenate([indices, rng.choice(source_count, remainder, replace=False)])
    rng.shuffle(indices)
    return indices


def two_choice_allocation(flow_count: int, capacity: int, seed: int) -> AllocationResult:
    admitted = np.zeros(flow_count, dtype=bool)
    if capacity <= 0:
        return AllocationResult(admitted, 0, 0, flow_count)

    rng = np.random.default_rng(seed)
    primary = rng.integers(0, capacity, size=flow_count, dtype=np.int64)
    secondary = rng.integers(0, capacity, size=flow_count, dtype=np.int64)
    occupied = np.zeros(capacity, dtype=bool)
    for index in range(flow_count):
        first = int(primary[index])
        if not occupied[first]:
            occupied[first] = True
            admitted[index] = True
            continue
        second = int(secondary[index])
        if not occupied[second]:
            occupied[second] = True
            admitted[index] = True

    admitted_flows = int(admitted.sum())
    return AllocationResult(admitted, capacity, admitted_flows, flow_count - admitted_flows)
