from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import ExperimentConfig
from .data import load_adaflow_dataset, load_full_flow_dataset, load_packet_dataset, load_partition_dataset, load_phase_dataset, split_window_dataset
from .resources import (
    ResourceReport,
    TargetProfile,
    estimate_adaflow_resources,
    estimate_leo_resources,
    estimate_llsy_resources,
    estimate_netbeacon_resources,
    estimate_splidt_resources,
)
from .scaling import AllocationResult, sample_indices, two_choice_allocation
from .splidt import SpliDT
from .statedt import StateDT
from .tree import fit_tree, remove_auxiliary_csvs, select_top_k_features, write_metrics_rows


@dataclass
class ModelEvaluation:
    key: str
    label: str
    y_true: np.ndarray
    predictions: np.ndarray
    train_majority_label: str
    resources: ResourceReport
    max_depth: int
    num_features: int
    num_partitions: int = 1
    stateful: bool = False


@dataclass(frozen=True)
class Population:
    y_true: np.ndarray
    predictions: np.ndarray


def run_flow_count_metrics(config: ExperimentConfig) -> list[Path]:
    output_paths = _model_metrics_output_paths(config)
    stale_output_path = _stale_dataset_metrics_output_path(config)
    legacy_output_path = _legacy_degradation_output_path(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    written_paths = []
    builders = [
        _evaluate_splidt,
        _evaluate_llsy,
        _evaluate_netbeacon,
        _evaluate_adaflow,
        _evaluate_leo,
        _evaluate_statedt,
    ]
    for builder in builders:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation = builder(config)
                rows = _build_flow_metrics_rows(config, [evaluation])[evaluation.key]
            output_path = output_paths[evaluation.key]
            write_metrics_rows(
                output_path,
                [_evaluation_metrics_row(config, evaluation)],
                replace_scope="evaluation",
            )
            _write_flow_metrics_csv(output_path, rows)
            written_paths.append(output_path)
            del rows
            del evaluation
        except FileNotFoundError as exc:
            model_name = builder.__name__.removeprefix("_evaluate_")
            print(f"{model_name} skipped: {exc}", file=sys.stderr)
        finally:
            gc.collect()
    for output_path in output_paths.values():
        remove_auxiliary_csvs(output_path.parent)
    _remove_stale_csv(stale_output_path)
    _remove_legacy_degradation_csv(legacy_output_path)
    return written_paths


def _model_metrics_output_paths(config: ExperimentConfig) -> dict[str, Path]:
    base = Path("results") / config.dataset.name
    return {
        "splidt": base / "splidt" / "metrics.csv",
        "llsy": base / "llsy" / "metrics.csv",
        "netbeacon": base / "netbeacon" / "metrics.csv",
        "adaflow": base / "adaflow" / "metrics.csv",
        "leo": base / "leo" / "metrics.csv",
        "statedt": base / "statedt" / "metrics.csv",
    }


def _stale_dataset_metrics_output_path(config: ExperimentConfig) -> Path:
    return Path("results") / config.dataset.name / "metrics.csv"


def _legacy_degradation_output_path(config: ExperimentConfig) -> Path:
    return Path("results") / config.dataset.name / "degradation" / "flow_accuracy_degradation.csv"


def _write_flow_metrics_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    write_metrics_rows(output_path, rows, replace_scope="flow_scaling")


def _remove_stale_csv(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()


def _remove_legacy_degradation_csv(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    try:
        output_path.parent.rmdir()
    except OSError:
        pass


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
        max_depth=config.splidt.max_depth,
        num_features=len(set(selected)),
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
        max_depth=model.get_depth(),
        num_features=len(selected),
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
        max_depth=model.get_depth(),
        num_features=len(selected),
        num_partitions=1,
        stateful=True,
    )


def _evaluate_adaflow(config: ExperimentConfig) -> ModelEvaluation:
    settings = config.adaflow
    split, phases = load_adaflow_dataset(
        config.dataset,
        settings.trigger_packet,
        settings.phase_delta,
        settings.num_dense_phases,
    )
    selected = select_top_k_features(split.X_train, split.y_train, settings.max_features, config.seed)
    model = fit_tree(
        split.X_train[selected],
        split.y_train,
        settings.max_depth,
        config.seed,
        sample_weight=split.train_sample_weights,
    )
    predictions = model.predict(split.X_test[selected])
    resources = estimate_adaflow_resources(
        TargetProfile.from_config(config.target),
        len(selected),
        model.tree_.node_count,
        max(phases),
    )
    return ModelEvaluation(
        key="adaflow",
        label="AdaFlow",
        y_true=_string_array(split.y_test),
        predictions=_string_array(predictions),
        train_majority_label=_majority_label(split.y_train),
        resources=resources,
        max_depth=model.get_depth(),
        num_features=len(selected),
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
        max_depth=model.get_depth(),
        num_features=len(selected),
        num_partitions=1,
        stateful=True,
    )


def _evaluate_statedt(config: ExperimentConfig) -> ModelEvaluation:
    split = load_full_flow_dataset(config.dataset)
    statedt = StateDT(config)
    evaluated = statedt.evaluate(split)

    return ModelEvaluation(
        key="statedt",
        label="StateDT",
        y_true=_string_array(split.y_test),
        predictions=_string_array(evaluated.predictions),
        train_majority_label=_majority_label(split.y_train),
        resources=evaluated.resources,
        max_depth=evaluated.trained.model.get_depth(),
        num_features=len(evaluated.trained.features),
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
        for flow_count in config.scaling.requested_flows:
            population = _population_for_flow_count(evaluation, encoded_true, encoded_predictions, flow_count, config.seed)
            if evaluation.stateful:
                allocation = two_choice_allocation(
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


def _evaluation_metrics_row(
    config: ExperimentConfig,
    evaluation: ModelEvaluation,
) -> dict[str, Any]:
    return {
        "dataset": config.dataset.name,
        "target": config.target.name,
        "model": evaluation.label,
        "seed": config.seed,
        "scope": "evaluation",
        "flow_count": "",
        "flows_used": "",
        "test_samples": len(evaluation.y_true),
        "macro_f1": float(
            f1_score(
                evaluation.y_true,
                evaluation.predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "max_depth": evaluation.max_depth,
        "num_features": evaluation.num_features,
        "num_partitions": evaluation.num_partitions,
        **_resource_columns(evaluation.resources),
    }


def _population_for_flow_count(
    evaluation: ModelEvaluation,
    encoded_true: np.ndarray,
    encoded_predictions: np.ndarray,
    flow_count: int,
    seed: int,
) -> Population:
    indices = sample_indices(
        len(encoded_true),
        flow_count,
        _stable_seed(seed, evaluation.key, flow_count),
    )
    return Population(
        y_true=encoded_true[indices],
        predictions=encoded_predictions[indices],
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
        "scope": "flow_scaling",
        "flow_count": len(y_true),
        "flows_used": allocation.admitted_flows,
        "test_samples": len(evaluation.y_true),
        "macro_f1": _macro_f1(y_true, overall_predictions),
        "max_depth": evaluation.max_depth,
        "num_features": evaluation.num_features,
        "num_partitions": evaluation.num_partitions,
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
        "within_tcam_capacity": resources.within_tcam_capacity,
        "within_stage_budget": resources.within_stage_budget,
        "target_feasible": resources.target_feasible,
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


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
