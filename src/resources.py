from __future__ import annotations

import math
from dataclasses import dataclass

from .config import TargetConfig

BITS_PER_MB = 8_000_000


@dataclass(frozen=True)
class TargetProfile:
    name: str
    feature_width: int
    node_id_width: int
    num_ma_units: int
    bookkeeping_stages: int
    iat_bookkeeping_stages: int
    tcam_capacity_mb: float
    state_memory_mb: float
    resubmission_bits_per_pkt: int
    resubmission_bw_gbps: float
    tcam_entry_width_bits: int
    tcam_entries_per_block: int
    tcam_blocks_per_stage: int
    sram_bits_per_stage: int
    register_word_bits: int

    @classmethod
    def from_config(cls, config: TargetConfig) -> "TargetProfile":
        return cls(**config.__dict__)


@dataclass
class ResourceReport:
    feature_table_entries: int = 0
    tree_table_entries: int = 0
    total_table_entries: int = 0
    tcam_blocks: int = 0
    tcam_stages: int = 0
    tcam_capacity_mb: float = 0.0
    tcam_memory_mb: float = 0.0
    feature_state_bits: int | None = None
    metadata_bits: int | None = None
    logical_entry_bits: int | None = None
    aligned_entry_bits: int | None = None
    register_words_per_flow: int | None = None
    estimated_flow_capacity: int | None = None


def allocated_tcam_entries(entries: int, key_width_bits: int, target: TargetProfile) -> int:
    rounded_entries = max(target.tcam_entries_per_block, math.ceil(entries / target.tcam_entries_per_block) * target.tcam_entries_per_block)
    width_blocks = max(1, math.ceil(key_width_bits / target.tcam_entry_width_bits))
    return rounded_entries * width_blocks


def tcam_stage_count(entries: int, key_width_bits: int, target: TargetProfile) -> int:
    allocated = allocated_tcam_entries(entries, key_width_bits, target)
    per_stage = target.tcam_entries_per_block * target.tcam_blocks_per_stage
    return max(1, math.ceil(allocated / per_stage))


def tcam_block_count(entries: int, key_width_bits: int, target: TargetProfile) -> int:
    return tcam_stage_count(entries, key_width_bits, target) * target.tcam_blocks_per_stage


def tcam_memory_bits(entries: int, key_width_bits: int, target: TargetProfile) -> int:
    return allocated_tcam_entries(entries, key_width_bits, target) * target.tcam_entry_width_bits


def aligned_register_entry_bits(logical_bits: int, word_bits: int = 32) -> int:
    return max(word_bits, math.ceil(logical_bits / word_bits) * word_bits)


def alignment_overhead(logical_bits: int, aligned_bits: int) -> int:
    return aligned_bits - logical_bits


def register_words_per_flow(aligned_bits: int, target: TargetProfile) -> int:
    return max(1, math.ceil(aligned_bits / target.register_word_bits))


def estimated_flow_capacity(memory_mb: float, aligned_bits: int) -> int:
    return int((memory_mb * BITS_PER_MB) // aligned_bits)


def state_update_stages(num_stateful_features: int, target: TargetProfile) -> int:
    return max(0, math.ceil(num_stateful_features / 4))


def _tree_report(
    target: TargetProfile,
    num_features: int,
    tree_entries: int,
    feature_entries: int = 0,
    extra_stages: int = 0,
    state_bits: int | None = None,
    metadata_bits: int | None = None,
) -> ResourceReport:
    key_width = target.node_id_width + max(1, num_features) * target.feature_width
    feature_stages = tcam_stage_count(feature_entries, key_width, target) if feature_entries else 0
    tree_stages = tcam_stage_count(max(1, tree_entries), key_width, target)
    stages = feature_stages + tree_stages + extra_stages
    blocks = (feature_stages + tree_stages) * target.tcam_blocks_per_stage
    tcam_bits = 0
    if feature_entries:
        tcam_bits += tcam_memory_bits(feature_entries, key_width, target)
    tcam_bits += tcam_memory_bits(max(1, tree_entries), key_width, target)
    tcam_memory_mb = tcam_bits / BITS_PER_MB
    logical_bits = None
    aligned_bits = None
    words = None
    capacity = None
    if state_bits is not None:
        logical_bits = state_bits + (metadata_bits or 0)
        aligned_bits = aligned_register_entry_bits(logical_bits, target.register_word_bits)
        words = register_words_per_flow(aligned_bits, target)
        capacity = estimated_flow_capacity(target.state_memory_mb, aligned_bits)
    return ResourceReport(
        feature_table_entries=feature_entries,
        tree_table_entries=tree_entries,
        total_table_entries=feature_entries + tree_entries,
        tcam_blocks=blocks,
        tcam_stages=stages,
        tcam_capacity_mb=target.tcam_capacity_mb,
        tcam_memory_mb=tcam_memory_mb,
        feature_state_bits=state_bits,
        metadata_bits=metadata_bits,
        logical_entry_bits=logical_bits,
        aligned_entry_bits=aligned_bits,
        register_words_per_flow=words,
        estimated_flow_capacity=capacity,
    )


def estimate_splidt_resources(target: TargetProfile, num_features: int, feature_entries: int, tree_entries: int) -> ResourceReport:
    # SpliDT time-shares k feature registers across partitions; metadata carries
    # the active subtree/window bookkeeping needed to reuse those registers.
    feature_state_bits = num_features * target.feature_width
    metadata_bits = target.node_id_width + target.feature_width + (target.iat_bookkeeping_stages * target.feature_width)
    extra_stages = target.bookkeeping_stages + target.iat_bookkeeping_stages + state_update_stages(num_features, target)
    return _tree_report(
        target,
        num_features,
        tree_entries,
        feature_entries,
        extra_stages=extra_stages,
        state_bits=feature_state_bits,
        metadata_bits=metadata_bits,
    )


def estimate_llsy_resources(target: TargetProfile, num_features: int, tree_entries: int) -> ResourceReport:
    # The paper's flow-level IIsy/LLSY variant keeps selected feature values in
    # per-flow registers before applying the lookup-based tree representation.
    feature_state_bits = num_features * target.feature_width
    extra_stages = target.bookkeeping_stages + state_update_stages(num_features, target)
    return _tree_report(
        target,
        num_features,
        tree_entries,
        extra_stages=extra_stages,
        state_bits=feature_state_bits,
        metadata_bits=1,
    )


def estimate_netbeacon_resources(target: TargetProfile, num_features: int, tree_entries: int, num_phases: int) -> ResourceReport:
    state_bits = num_features * target.feature_width + math.ceil(math.log2(max(2, num_phases)))
    return _tree_report(target, num_features, tree_entries, extra_stages=state_update_stages(num_features, target), state_bits=state_bits, metadata_bits=1)


def estimate_adaflow_resources(target: TargetProfile, num_features: int, tree_entries: int, max_phase: int) -> ResourceReport:
    # One aggregated tree, flow features, packet counter, and PrioritySketch bit.
    counter_bits = max(1, math.ceil(math.log2(max_phase + 1)))
    state_bits = num_features * target.feature_width + counter_bits
    metadata_bits = 1  # PrioritySketch's per-flow priority flag.
    extra_stages = target.bookkeeping_stages + state_update_stages(num_features, target)
    return _tree_report(target, num_features, tree_entries, extra_stages=extra_stages, state_bits=state_bits, metadata_bits=metadata_bits)


def estimate_leo_resources(target: TargetProfile, num_features: int, tree_entries: int) -> ResourceReport:
    state_bits = num_features * target.feature_width
    return _tree_report(target, num_features, tree_entries, extra_stages=state_update_stages(num_features, target), state_bits=state_bits, metadata_bits=1)


def estimate_statedt_resources(target: TargetProfile, tree_entries: int, feature_state_bits: int, metadata_bits: int, num_features: int) -> ResourceReport:
    return _tree_report(target, num_features, tree_entries, extra_stages=state_update_stages(num_features, target), state_bits=feature_state_bits, metadata_bits=metadata_bits)
