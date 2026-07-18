from __future__ import annotations

import contextlib
import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_full_flow_dataset, load_packet_dataset, load_partition_dataset, load_phase_dataset, split_window_dataset
from .resources import (
    ResourceReport,
    TargetProfile,
    estimate_leo_resources,
    estimate_llsy_resources,
    estimate_netbeacon_resources,
    estimate_splidt_resources,
    estimate_statedt_resources,
)
from .splidt import SpliDT
from .statedt import StateDT, detect_stateful_features
from .tree import fit_tree, predict_with_path, select_top_k_features


FLOW_METRICS_COLUMNS = [
    "dataset",
    "target",
    "model",
    "seed",
    "flow_count",
    "flows_used",
    "num_partitions",
    "f1",
    "feature_state_bits",
    "metadata_bits",
    "logical_entry_bits",
    "aligned_entry_bits",
    "estimated_flow_capacity",
    "feature_table_entries",
    "tree_table_entries",
    "total_table_entries",
    "tcam_blocks",
    "tcam_stages",
    "tcam_capacity_mb",
    "tcam_memory_mb",
    "register_words_per_flow",
]


@dataclass(frozen=True)
class CsvFingerprint:
    mtime_ns: int
    size: int


@dataclass
class ModelEvaluation:
    key: str
    label: str
    y_true: np.ndarray
    predictions: np.ndarray
    train_majority_label: str
    resources: ResourceReport
    num_partitions: int = 1
    stateful: bool = False


@dataclass(frozen=True)
class AllocationResult:
    admitted_mask: np.ndarray
    state_capacity: int
    admitted_flows: int
    unresolved_flows: int


@dataclass(frozen=True)
class Population:
    y_true: np.ndarray
    predictions: np.ndarray


def run_flow_count_metrics(config: ExperimentConfig) -> list[Path]:
    output_paths = _model_metrics_output_paths(config)
    stale_resource_paths = _model_resource_output_paths(config)
    stale_output_path = _stale_dataset_metrics_output_path(config)
    legacy_output_path = _legacy_degradation_output_path(config)
    csv_snapshot = _snapshot_existing_csvs(*output_paths.values(), *stale_resource_paths, stale_output_path, legacy_output_path)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluations = _build_model_evaluations(config)
        rows_by_model = _build_flow_metrics_rows(config, evaluations)
    written_paths = _write_model_metrics_csvs(output_paths, rows_by_model)
    _remove_stale_csvs(stale_resource_paths)
    _remove_stale_csv(stale_output_path)
    _remove_legacy_degradation_csv(legacy_output_path)
    _validate_csv_snapshot(csv_snapshot)
    return written_paths


def _model_metrics_output_paths(config: ExperimentConfig) -> dict[str, Path]:
    base = Path("results") / config.dataset.name
    return {
        "splidt": base / "splidt" / "metrics.csv",
        "llsy": base / "llsy" / "metrics.csv",
        "netbeacon": base / "netbeacon" / "metrics.csv",
        "leo": base / "leo" / "metrics.csv",
        "statedt": base / "statedt" / "metrics.csv",
    }


def _model_resource_output_paths(config: ExperimentConfig) -> list[Path]:
    base = Path("results") / config.dataset.name
    return [
        base / "splidt" / "resource.csv",
        base / "llsy" / "resource.csv",
        base / "netbeacon" / "resource.csv",
        base / "leo" / "resource.csv",
        base / "statedt" / "resource.csv",
    ]


def _stale_dataset_metrics_output_path(config: ExperimentConfig) -> Path:
    return Path("results") / config.dataset.name / "metrics.csv"


def _legacy_degradation_output_path(config: ExperimentConfig) -> Path:
    return Path("results") / config.dataset.name / "degradation" / "flow_accuracy_degradation.csv"


def _snapshot_existing_csvs(*allowed_output_paths: Path) -> dict[Path, CsvFingerprint]:
    allowed_paths = set()
    for path in allowed_output_paths:
        allowed_paths.add(path.resolve())
        allowed_paths.add(path.with_suffix(".tmp.csv").resolve())
    snapshot = {}
    for path in sorted(Path("results").glob("**/*.csv")):
        resolved_path = path.resolve()
        if resolved_path in allowed_paths:
            continue
        stat = path.stat()
        snapshot[path] = CsvFingerprint(mtime_ns=stat.st_mtime_ns, size=stat.st_size)
    return snapshot


def _validate_csv_snapshot(snapshot: dict[Path, CsvFingerprint]) -> None:
    for path, before in snapshot.items():
        if not path.exists():
            raise RuntimeError(f"flow metrics command modified an existing CSV: {path}")
        stat = path.stat()
        after = CsvFingerprint(mtime_ns=stat.st_mtime_ns, size=stat.st_size)
        if after != before:
            raise RuntimeError(f"flow metrics command modified an existing CSV: {path}")


def _write_model_metrics_csvs(output_paths: dict[str, Path], rows_by_model: dict[str, list[dict[str, Any]]]) -> list[Path]:
    written_paths = []
    for model_key, rows in rows_by_model.items():
        output_path = output_paths[model_key]
        _write_flow_metrics_csv(output_path, rows)
        written_paths.append(output_path)
    return written_paths


def _write_flow_metrics_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.csv")
    results_df = pd.DataFrame(rows, columns=FLOW_METRICS_COLUMNS)
    results_df.to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(output_path)


def _remove_stale_csv(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()


def _remove_stale_csvs(output_paths: list[Path]) -> None:
    for output_path in output_paths:
        _remove_stale_csv(output_path)


def _remove_legacy_degradation_csv(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    try:
        output_path.parent.rmdir()
    except OSError:
        pass


def _build_model_evaluations(config: ExperimentConfig) -> list[ModelEvaluation]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    evaluations: list[ModelEvaluation] = []
    for builder in [_evaluate_splidt, _evaluate_llsy, _evaluate_netbeacon, _evaluate_leo, _evaluate_statedt]:
        try:
            evaluations.append(builder(config))
        except FileNotFoundError:
            continue
    return evaluations


def _evaluate_splidt(config: ExperimentConfig) -> ModelEvaluation:
    partition_count = len(config.splidt.partition_sizes)
    root, y_test, predictions = SpliDT(config).evaluate()
    selected = [feature for subtree in root.all_features() for feature in subtree]
    tree_entries = root.deployed_node_count()
    window_data = load_partition_dataset(config.dataset, partition_count)
    y_train = split_window_dataset(config.dataset, window_data, [1]).y_train
    resources = estimate_splidt_resources(
        TargetProfile.from_config(config.target),
        config.splidt.features_per_partition,
        len(selected),
        tree_entries,
    )
    return ModelEvaluation(
        key="splidt",
        label="SpliDT",
        y_true=_string_array(y_test),
        predictions=_string_array(predictions),
        train_majority_label=_majority_label(y_train),
        resources=resources,
        num_partitions=partition_count,
        stateful=True,
    )


def _evaluate_llsy(config: ExperimentConfig) -> ModelEvaluation:
    split = load_packet_dataset(config.dataset, config.llsy.packet_index)
    selected = select_top_k_features(split.X_train, split.y_train, config.llsy.max_features, config.seed)
    model = fit_tree(split.X_train[selected], split.y_train, config.llsy.max_depth, config.seed)
    predictions = model.predict(split.X_test[selected])
    resources = estimate_llsy_resources(
        TargetProfile.from_config(config.target),
        len(selected),
        model.tree_.node_count,
    )
    return ModelEvaluation(
        key="llsy",
        label="LLSY",
        y_true=_string_array(split.y_test),
        predictions=_string_array(predictions),
        train_majority_label=_majority_label(split.y_train),
        resources=resources,
        num_partitions=1,
        stateful=True,
    )


def _evaluate_netbeacon(config: ExperimentConfig) -> ModelEvaluation:
    split = load_phase_dataset(config.dataset, config.netbeacon.phases)
    selected = select_top_k_features(split.X_train, split.y_train, config.netbeacon.max_features, config.seed)
    model = fit_tree(split.X_train[selected], split.y_train, config.netbeacon.max_depth, config.seed)
    predictions = model.predict(split.X_test[selected])
    target = TargetProfile.from_config(config.target)
    resources = estimate_netbeacon_resources(target, len(selected), model.tree_.node_count, len(config.netbeacon.phases))

    return ModelEvaluation(
        key="netbeacon",
        label="NetBeacon",
        y_true=_string_array(split.y_test),
        predictions=_string_array(predictions),
        train_majority_label=_majority_label(split.y_train),
        resources=resources,
        num_partitions=1,
        stateful=True,
    )


def _evaluate_leo(config: ExperimentConfig) -> ModelEvaluation:
    split = load_full_flow_dataset(config.dataset)
    selected = select_top_k_features(split.X_train, split.y_train, config.leo.max_features, config.seed)
    model = fit_tree(split.X_train[selected], split.y_train, config.leo.max_depth, config.seed)
    predictions = model.predict(split.X_test[selected])
    target = TargetProfile.from_config(config.target)
    resources = estimate_leo_resources(target, len(selected), model.tree_.node_count)

    return ModelEvaluation(
        key="leo",
        label="LEO",
        y_true=_string_array(split.y_test),
        predictions=_string_array(predictions),
        train_majority_label=_majority_label(split.y_train),
        resources=resources,
        num_partitions=1,
        stateful=True,
    )


def _evaluate_statedt(config: ExperimentConfig) -> ModelEvaluation:
    split = load_full_flow_dataset(config.dataset)
    target = TargetProfile.from_config(config.target)
    candidates = detect_stateful_features(list(split.X_train.columns)) if config.statedt.stateful_only else list(split.X_train.columns)
    if config.statedt.explicit_features:
        candidates = [feature for feature in config.statedt.explicit_features if feature in split.X_train.columns]
    selected = select_top_k_features(split.X_train, split.y_train, config.statedt.max_features, config.seed, candidates)
    model = fit_tree(split.X_train[selected], split.y_train, config.statedt.max_depth, config.seed)
    statedt = StateDT(config)
    compiled = statedt.compile(model, selected)
    software_predictions, software_paths = predict_with_path(model, split.X_test[selected])
    compiled_pairs = [statedt.predict_compiled(compiled, row) for _, row in split.X_test[selected].iterrows()]
    compiled_predictions = np.array([prediction for prediction, _ in compiled_pairs])
    compiled_paths = [path for _, path in compiled_pairs]
    if not np.array_equal(compiled_predictions, software_predictions) or compiled_paths != software_paths:
        raise RuntimeError("StateDT exact compilation does not reproduce the software tree")

    feature_state_bits = sum(spec.logical_bits for spec in compiled.feature_specs.values())
    metadata_bits = config.statedt.fingerprint_bits + config.statedt.generation_bits + config.statedt.valid_bits
    resources = estimate_statedt_resources(target, model.tree_.node_count, feature_state_bits, metadata_bits, len(selected))

    return ModelEvaluation(
        key="statedt",
        label="StateDT",
        y_true=_string_array(split.y_test),
        predictions=_string_array(software_predictions),
        train_majority_label=_majority_label(split.y_train),
        resources=resources,
        num_partitions=1,
        stateful=True,
    )


def _build_flow_metrics_rows(config: ExperimentConfig, evaluations: list[ModelEvaluation]) -> dict[str, list[dict[str, Any]]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        rows: list[dict[str, Any]] = []
        fallback_label = _fallback_label(config, evaluation)
        label_ids = _label_ids(evaluation.y_true, evaluation.predictions, fallback_label)
        encoded_true = _encode_labels(evaluation.y_true, label_ids)
        encoded_predictions = _encode_labels(evaluation.predictions, label_ids)
        fallback_id = label_ids[fallback_label]
        for flow_count in config.statedt.requested_flows:
            population = _population_for_flow_count(evaluation, encoded_true, encoded_predictions, flow_count, config.seed)
            if evaluation.stateful:
                allocation = _simulate_two_choice_allocation(
                    flow_count,
                    int(evaluation.resources.estimated_flow_capacity or 0),
                    _stable_seed(config.seed, evaluation.key, flow_count, "allocation"),
                )
                rows.append(
                    _row_for_population(
                        config,
                        evaluation,
                        population,
                        fallback_id,
                        allocation,
                    )
                )
            else:
                allocation = AllocationResult(
                    admitted_mask=np.ones(flow_count, dtype=bool),
                    state_capacity=0,
                    admitted_flows=flow_count,
                    unresolved_flows=0,
                )
                rows.append(
                    _row_for_population(
                        config,
                        evaluation,
                        population,
                        fallback_id,
                        allocation,
                    )
                )
        rows_by_model[evaluation.key] = rows
    return rows_by_model


def _population_for_flow_count(
    evaluation: ModelEvaluation,
    encoded_true: np.ndarray,
    encoded_predictions: np.ndarray,
    flow_count: int,
    seed: int,
) -> Population:
    indices = _sample_indices(
        len(encoded_true),
        flow_count,
        _stable_seed(seed, evaluation.key, flow_count),
    )
    return Population(
        y_true=encoded_true[indices],
        predictions=encoded_predictions[indices],
    )


def _sample_indices(source_count: int, requested_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if requested_count <= source_count:
        return rng.choice(source_count, requested_count, replace=False)
    full_repeats, remainder = divmod(requested_count, source_count)
    indices = np.tile(np.arange(source_count, dtype=np.int64), full_repeats)
    if remainder:
        indices = np.concatenate([indices, rng.choice(source_count, remainder, replace=False)])
    rng.shuffle(indices)
    return indices


def _simulate_two_choice_allocation(flow_count: int, capacity: int, seed: int) -> AllocationResult:
    if capacity <= 0:
        return AllocationResult(
            admitted_mask=np.zeros(flow_count, dtype=bool),
            state_capacity=0,
            admitted_flows=0,
            unresolved_flows=flow_count,
        )

    rng = np.random.default_rng(seed)
    primary_choices = rng.integers(0, capacity, size=flow_count, dtype=np.int64)
    secondary_choices = rng.integers(0, capacity, size=flow_count, dtype=np.int64)
    occupied = np.zeros(capacity, dtype=bool)
    admitted_mask = np.zeros(flow_count, dtype=bool)

    for index in range(flow_count):
        primary = int(primary_choices[index])
        if not occupied[primary]:
            occupied[primary] = True
            admitted_mask[index] = True
            continue
        secondary = int(secondary_choices[index])
        if not occupied[secondary]:
            occupied[secondary] = True
            admitted_mask[index] = True

    admitted_flows = int(admitted_mask.sum())
    unresolved_flows = flow_count - admitted_flows
    return AllocationResult(
        admitted_mask=admitted_mask,
        state_capacity=capacity,
        admitted_flows=admitted_flows,
        unresolved_flows=unresolved_flows,
    )


def _row_for_population(
    config: ExperimentConfig,
    evaluation: ModelEvaluation,
    population: Population,
    fallback_id: int,
    allocation: AllocationResult,
) -> dict[str, Any]:
    y_true = population.y_true
    predictions = population.predictions

    overall_predictions = predictions.copy()
    fallback_count = allocation.unresolved_flows if evaluation.stateful else 0
    if fallback_count:
        overall_predictions[~allocation.admitted_mask] = fallback_id

    return {
        "dataset": config.dataset.name,
        "target": config.target.name,
        "model": evaluation.label,
        "seed": config.seed,
        "flow_count": len(y_true),
        "flows_used": allocation.admitted_flows,
        "num_partitions": evaluation.num_partitions,
        "f1": _macro_f1(y_true, overall_predictions),
        **_resource_columns(evaluation.resources),
    }


def _resource_columns(resources: ResourceReport) -> dict[str, Any]:
    return {
        "feature_state_bits": _blank_if_none(resources.feature_state_bits),
        "metadata_bits": _blank_if_none(resources.metadata_bits),
        "logical_entry_bits": _blank_if_none(resources.logical_entry_bits),
        "aligned_entry_bits": _blank_if_none(resources.aligned_entry_bits),
        "estimated_flow_capacity": _blank_if_none(resources.estimated_flow_capacity),
        "feature_table_entries": resources.feature_table_entries,
        "tree_table_entries": resources.tree_table_entries,
        "total_table_entries": resources.total_table_entries,
        "tcam_blocks": resources.tcam_blocks,
        "tcam_stages": resources.tcam_stages,
        "tcam_capacity_mb": resources.tcam_capacity_mb,
        "tcam_memory_mb": resources.tcam_memory_mb,
        "register_words_per_flow": _blank_if_none(resources.register_words_per_flow),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    max_label = int(max(y_true.max(), y_pred.max())) + 1
    confusion = np.bincount(y_true * max_label + y_pred, minlength=max_label * max_label).reshape(max_label, max_label)
    true_positive = np.diag(confusion).astype(float)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    active = (confusion.sum(axis=0) + confusion.sum(axis=1)) > 0
    f1_values = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator != 0)
    return float(f1_values[active].mean()) if active.any() else 0.0


def _label_ids(y_true: np.ndarray, predictions: np.ndarray, fallback_label: str) -> dict[str, int]:
    labels = sorted(set(y_true.tolist()) | set(predictions.tolist()) | {fallback_label})
    return {label: index for index, label in enumerate(labels)}


def _fallback_label(config: ExperimentConfig, evaluation: ModelEvaluation) -> str:
    if not evaluation.stateful or config.statedt.fallback == "majority_class":
        return evaluation.train_majority_label
    return "__NO_PREDICTION__"


def _encode_labels(values: np.ndarray, label_ids: dict[str, int]) -> np.ndarray:
    return np.fromiter((label_ids[str(value)] for value in values), dtype=np.int64, count=len(values))


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big") % (2**32)


def _string_array(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str)


def _majority_label(labels: pd.Series) -> str:
    return str(labels.mode().iloc[0])


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value
