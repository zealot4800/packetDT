from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    label_column: str
    flow_id_column: str
    window_column: str
    phase_column: str
    packet_column: str
    full_flow_file: str
    partition_file_pattern: str
    phase_file: str
    packet_file: str

    @property
    def directory(self) -> Path:
        return Path(self.root) / self.name


@dataclass(frozen=True)
class TargetConfig:
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
    tcam_entry_width_bits: int = 44
    tcam_entries_per_block: int = 512
    tcam_blocks_per_stage: int = 24
    sram_bits_per_stage: int = 32 * 1024 * 1024
    register_word_bits: int = 32


@dataclass(frozen=True)
class SpliDTConfig:
    max_depth: int
    features_per_partition: int
    partition_sizes: tuple[int, ...]


@dataclass(frozen=True)
class BasicTreeConfig:
    max_depth: int
    max_features: int


@dataclass(frozen=True)
class LLSYConfig(BasicTreeConfig):
    packet_index: int


@dataclass(frozen=True)
class NetBeaconConfig(BasicTreeConfig):
    phases: tuple[int, ...]


@dataclass(frozen=True)
class AdaFlowConfig(BasicTreeConfig):
    trigger_packet: int
    phase_delta: int
    num_dense_phases: int
    determination_threshold: float


@dataclass(frozen=True)
class StateDTConfig(BasicTreeConfig):
    stateful_only: bool
    explicit_features: tuple[str, ...]
    fingerprint_bits: int
    generation_bits: int
    valid_bits: int
    allocator: str
    requested_flows: tuple[int, ...]
    fallback: str


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    dataset: DatasetConfig
    target: TargetConfig
    seed: int
    splidt: SpliDTConfig
    llsy: LLSYConfig
    netbeacon: NetBeaconConfig
    adaflow: AdaFlowConfig
    leo: BasicTreeConfig
    statedt: StateDTConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _positive_int_tuple(values: Any, name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_positive_int(item, name) for item in values)


def _dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    required = [
        "name",
        "root",
        "label_column",
        "flow_id_column",
        "window_column",
        "phase_column",
        "packet_column",
        "full_flow_file",
        "partition_file_pattern",
        "phase_file",
        "packet_file",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"dataset section missing: {', '.join(missing)}")
    filenames = [
        raw["full_flow_file"],
        raw["phase_file"],
        raw["packet_file"],
        raw["partition_file_pattern"].format(num_partitions=1),
    ]
    for filename in filenames:
        if Path(filename).name != filename or not filename.endswith(".pkl"):
            raise ValueError(f"dataset file name must be a local .pkl file: {filename}")
    return DatasetConfig(**{key: str(raw[key]) for key in required})


def load_target_config(path: str | Path, target_name: str) -> TargetConfig:
    raw = _read_yaml(path)
    if target_name not in raw:
        raise ValueError(f"target '{target_name}' not found in {path}")
    target = raw[target_name] or {}
    return TargetConfig(
        name=target_name,
        feature_width=_positive_int(target.get("feature_width"), "feature_width"),
        node_id_width=_positive_int(target.get("node_id_width"), "node_id_width"),
        num_ma_units=_positive_int(target.get("num_ma_units"), "num_ma_units"),
        bookkeeping_stages=_positive_int(target.get("bookkeeping_stages"), "bookkeeping_stages"),
        iat_bookkeeping_stages=_positive_int(target.get("iat_bookkeeping_stages"), "iat_bookkeeping_stages"),
        tcam_capacity_mb=_positive_float(target.get("tcam_capacity_mb"), "tcam_capacity_mb"),
        state_memory_mb=_positive_float(target.get("state_memory_mb"), "state_memory_mb"),
        resubmission_bits_per_pkt=_positive_int(target.get("resubmission_bits_per_pkt"), "resubmission_bits_per_pkt"),
        resubmission_bw_gbps=float(target.get("resubmission_bw_Gbps", target.get("resubmission_bw_gbps", 0))),
    )


def _basic(raw: dict[str, Any], section: str) -> BasicTreeConfig:
    return BasicTreeConfig(
        max_depth=_positive_int(raw.get("max_depth"), f"{section}.max_depth"),
        max_features=_positive_int(raw.get("max_features"), f"{section}.max_features"),
    )


def load_experiment_config(path: str) -> ExperimentConfig:
    config_path = Path(path)
    raw = _read_yaml(config_path)
    for section in ["dataset", "experiment", "splidt", "llsy", "netbeacon", "adaflow", "leo", "statedt"]:
        if section not in raw:
            raise ValueError(f"config missing required section: {section}")

    dataset = _dataset_config(raw["dataset"])
    experiment = raw["experiment"] or {}
    seed = _positive_int(experiment.get("seed"), "experiment.seed")
    target_name = str(experiment.get("target", ""))
    target = load_target_config(config_path.parent.parent / "targets" / "tofino.yaml", target_name)

    splidt_raw = raw["splidt"] or {}
    partition_sizes = _positive_int_tuple(splidt_raw.get("partition_sizes"), "splidt.partition_sizes")
    splidt = SpliDTConfig(
        max_depth=_positive_int(splidt_raw.get("max_depth"), "splidt.max_depth"),
        features_per_partition=_positive_int(splidt_raw.get("features_per_partition"), "splidt.features_per_partition"),
        partition_sizes=partition_sizes,
    )
    if sum(splidt.partition_sizes) != splidt.max_depth:
        raise ValueError("sum(splidt.partition_sizes) must equal splidt.max_depth")

    llsy_raw = raw["llsy"] or {}
    llsy = LLSYConfig(
        max_depth=_positive_int(llsy_raw.get("max_depth"), "llsy.max_depth"),
        max_features=_positive_int(llsy_raw.get("max_features"), "llsy.max_features"),
        packet_index=int(llsy_raw.get("packet_index", 0)),
    )

    netbeacon_raw = raw["netbeacon"] or {}
    phases = _positive_int_tuple(netbeacon_raw.get("phases"), "netbeacon.phases")
    if any(phase & (phase - 1) for phase in phases):
        raise ValueError("netbeacon.phases must contain powers of two")
    netbeacon = NetBeaconConfig(
        max_depth=_positive_int(netbeacon_raw.get("max_depth"), "netbeacon.max_depth"),
        max_features=_positive_int(netbeacon_raw.get("max_features"), "netbeacon.max_features"),
        phases=phases,
    )

    adaflow_raw = raw["adaflow"] or {}
    determination_threshold = float(adaflow_raw.get("determination_threshold", 0.8))
    if not 0 < determination_threshold <= 1:
        raise ValueError("adaflow.determination_threshold must be in (0, 1]")
    adaflow = AdaFlowConfig(
        max_depth=_positive_int(adaflow_raw.get("max_depth"), "adaflow.max_depth"),
        max_features=_positive_int(adaflow_raw.get("max_features"), "adaflow.max_features"),
        trigger_packet=_positive_int(adaflow_raw.get("trigger_packet"), "adaflow.trigger_packet"),
        phase_delta=_positive_int(adaflow_raw.get("phase_delta"), "adaflow.phase_delta"),
        num_dense_phases=_positive_int(adaflow_raw.get("num_dense_phases"), "adaflow.num_dense_phases"),
        determination_threshold=determination_threshold,
    )

    statedt_raw = raw["statedt"] or {}
    selection = statedt_raw.get("feature_selection") or {}
    state = statedt_raw.get("state") or {}
    scaling = statedt_raw.get("scaling") or {}
    if state.get("allocator") != "two_choice":
        raise ValueError("statedt.state.allocator must be 'two_choice'")
    statedt = StateDTConfig(
        max_depth=_positive_int(statedt_raw.get("max_depth"), "statedt.max_depth"),
        max_features=_positive_int(statedt_raw.get("max_features"), "statedt.max_features"),
        stateful_only=bool(selection.get("stateful_only", True)),
        explicit_features=tuple(str(item) for item in selection.get("explicit_features", []) or []),
        fingerprint_bits=_positive_int(state.get("fingerprint_bits"), "statedt.state.fingerprint_bits"),
        generation_bits=_positive_int(state.get("generation_bits"), "statedt.state.generation_bits"),
        valid_bits=_positive_int(state.get("valid_bits"), "statedt.state.valid_bits"),
        allocator=str(state.get("allocator")),
        requested_flows=_positive_int_tuple(scaling.get("requested_flows"), "statedt.scaling.requested_flows"),
        fallback=str(statedt_raw.get("fallback", "majority_class")),
    )
    if statedt.fallback not in {"majority_class", "no_prediction"}:
        raise ValueError("statedt.fallback must be 'majority_class' or 'no_prediction'")

    return ExperimentConfig(
        path=config_path,
        dataset=dataset,
        target=target,
        seed=seed,
        splidt=splidt,
        llsy=llsy,
        netbeacon=netbeacon,
        adaflow=adaflow,
        leo=_basic(raw["leo"] or {}, "leo"),
        statedt=statedt,
    )
