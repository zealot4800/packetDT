from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

from .config import ExperimentConfig
from .data import load_full_flow_dataset
from .resources import ResourceReport, TargetProfile, estimate_statedt_resources
from .scaling import sample_indices, two_choice_allocation
from .tree import (
    ModelResult,
    calculate_macro_f1,
    extract_thresholds_by_feature,
    fit_tree,
    predict_with_path,
    save_model_artifacts,
    select_top_k_features,
    serialize_tree,
)


@dataclass(frozen=True)
class StateDTCandidate:
    feature_count: int
    features: tuple[str, ...]
    validation_f1: float
    scaled_validation_f1: float
    feature_state_bits: int
    aligned_entry_bits: int
    estimated_flow_capacity: int
    eligible: bool = False
    selected: bool = False


@dataclass
class TrainedStateDT:
    model: Any
    features: list[str]
    candidates: list[StateDTCandidate]


@dataclass
class EvaluatedStateDT:
    trained: TrainedStateDT
    predictions: np.ndarray
    macro_f1: float
    resources: ResourceReport


@dataclass(frozen=True)
class FeatureStateSpec:
    thresholds: tuple[float, ...]
    logical_bits: int


@dataclass
class CompiledStateDT:
    feature_specs: dict[str, FeatureStateSpec]
    tree: dict[str, Any]


def _logical_bits_for_bins(thresholds: tuple[float, ...]) -> int:
    return 0 if not thresholds else max(1, math.ceil(math.log2(len(thresholds) + 1)))


def _encode_value(value: float, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return index
    return len(thresholds)


class StateDT:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def compile(self, model, features: list[str]) -> CompiledStateDT:
        thresholds = extract_thresholds_by_feature(model, features)
        specs = {
            feature: FeatureStateSpec(values, _logical_bits_for_bins(values))
            for feature, values in thresholds.items()
        }
        return CompiledStateDT(feature_specs=specs, tree=serialize_tree(model, features))

    def predict_compiled(self, compiled: CompiledStateDT, sample: pd.Series) -> tuple[Any, list[int]]:
        states = {
            feature: _encode_value(float(np.float32(sample[feature])), spec.thresholds)
            for feature, spec in compiled.feature_specs.items()
        }
        nodes = {node["node_id"]: node for node in compiled.tree["nodes"]}
        node_id = 0
        path = []
        while True:
            node = nodes[node_id]
            path.append(node_id)
            if node["is_leaf"]:
                return node["prediction"], path
            spec = compiled.feature_specs[node["feature"]]
            threshold_index = spec.thresholds.index(node["threshold"])
            node_id = node["left_child"] if states[node["feature"]] <= threshold_index else node["right_child"]

    def train(self, split) -> TrainedStateDT:
        available_features = list(split.X_train.columns)
        candidates = list(available_features)
        if self.config.statedt.stateful_only:
            candidates = list(available_features)
        if self.config.statedt.explicit_features:
            missing = [
                feature
                for feature in self.config.statedt.explicit_features
                if feature not in split.X_train.columns
            ]
            if missing:
                raise ValueError(f"StateDT explicit feature(s) not found: {', '.join(missing)}")
            candidates = list(self.config.statedt.explicit_features)
        if not candidates:
            raise ValueError("StateDT has no candidate features")

        if not self.config.statedt.scaling_aware:
            selected = select_top_k_features(
                split.X_train,
                split.y_train,
                self.config.statedt.max_features,
                self.config.seed,
                candidates,
            )
            model = fit_tree(split.X_train[selected], split.y_train, self.config.statedt.max_depth, self.config.seed)
            return TrainedStateDT(model, selected, [])

        max_features = min(self.config.statedt.max_features, len(candidates))
        ranked = select_top_k_features(
            split.X_train,
            split.y_train,
            max_features,
            self.config.seed,
            candidates,
        )
        out_of_fold_predictions = _state_out_of_fold_predictions(
            split.X_train,
            split.y_train,
            candidates,
            max_features,
            self.config.statedt.max_depth,
            self.config.statedt.validation_folds,
            self.config.seed,
        )
        metadata_bits = self.config.statedt.fingerprint_bits + self.config.statedt.generation_bits + self.config.statedt.valid_bits
        target = TargetProfile.from_config(self.config.target)
        objective_flows = max(self.config.scaling.requested_flows)
        population_indices = sample_indices(len(split.y_train), objective_flows, self.config.seed)
        training_labels = split.y_train.astype(str).to_numpy()
        population_true = training_labels[population_indices]
        fallback_label = str(split.y_train.mode().iloc[0])
        allocation_masks: dict[int, np.ndarray] = {}
        raw_candidates = []

        for feature_count in range(1, len(ranked) + 1):
            features = ranked[:feature_count]
            validation_predictions = out_of_fold_predictions[feature_count]
            validation_f1 = float(f1_score(training_labels, validation_predictions, average="macro", zero_division=0))
            full_model = fit_tree(split.X_train[features], split.y_train, self.config.statedt.max_depth, self.config.seed)
            feature_state_bits = 0
            resources = estimate_statedt_resources(
                target,
                full_model.tree_.node_count,
                feature_state_bits,
                metadata_bits,
                len(features),
            )
            capacity = int(resources.estimated_flow_capacity or 0)
            admitted = allocation_masks.setdefault(
                capacity,
                two_choice_allocation(objective_flows, capacity, self.config.seed).admitted_mask,
            )
            population_predictions = validation_predictions[population_indices]
            scaled_predictions = population_predictions.copy()
            scaled_predictions[~admitted] = fallback_label
            scaled_f1 = float(f1_score(population_true, scaled_predictions, average="macro", zero_division=0))
            raw_candidates.append(
                {
                    "feature_count": feature_count,
                    "features": tuple(features),
                    "validation_f1": validation_f1,
                    "scaled_validation_f1": scaled_f1,
                    "feature_state_bits": feature_state_bits,
                    "aligned_entry_bits": int(resources.aligned_entry_bits or 0),
                    "estimated_flow_capacity": capacity,
                    "model": full_model,
                }
            )

        best_validation_f1 = max(candidate["validation_f1"] for candidate in raw_candidates)
        minimum_f1 = best_validation_f1 - self.config.statedt.max_f1_drop
        eligible = [candidate for candidate in raw_candidates if candidate["validation_f1"] >= minimum_f1]
        selected_candidate = max(
            eligible,
            key=lambda candidate: (
                candidate["scaled_validation_f1"],
                candidate["validation_f1"],
                candidate["estimated_flow_capacity"],
                -candidate["feature_count"],
            ),
        )
        eligible_feature_counts = {candidate["feature_count"] for candidate in eligible}
        reports = [
            StateDTCandidate(
                feature_count=candidate["feature_count"],
                features=candidate["features"],
                validation_f1=candidate["validation_f1"],
                scaled_validation_f1=candidate["scaled_validation_f1"],
                feature_state_bits=candidate["feature_state_bits"],
                aligned_entry_bits=candidate["aligned_entry_bits"],
                estimated_flow_capacity=candidate["estimated_flow_capacity"],
                eligible=candidate["feature_count"] in eligible_feature_counts,
                selected=candidate is selected_candidate,
            )
            for candidate in raw_candidates
        ]
        return TrainedStateDT(
            selected_candidate["model"],
            list(selected_candidate["features"]),
            reports,
        )

    def evaluate(self, split) -> EvaluatedStateDT:
        trained = self.train(split)
        model = trained.model
        selected = trained.features
        software_predictions, _ = predict_with_path(model, split.X_test[selected])

        metadata_bits = self.config.statedt.fingerprint_bits + self.config.statedt.generation_bits + self.config.statedt.valid_bits
        resources = estimate_statedt_resources(
            TargetProfile.from_config(self.config.target),
            model.tree_.node_count,
            0,
            metadata_bits,
            len(selected),
        )
        return EvaluatedStateDT(
            trained=trained,
            predictions=software_predictions,
            macro_f1=calculate_macro_f1(split.y_test, software_predictions),
            resources=resources,
        )

    def run(self, output_dir: Path) -> ModelResult:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        split = load_full_flow_dataset(self.config.dataset)
        evaluated = self.evaluate(split)
        trained = evaluated.trained
        selected = trained.features
        model = trained.model
        result = ModelResult.from_resources(
            model="StateDT",
            dataset=self.config.dataset.name,
            target=self.config.target.name,
            seed=self.config.seed,
            macro_f1=evaluated.macro_f1,
            max_depth=model.get_depth(),
            num_features=len(selected),
            test_samples=len(split.y_test),
            num_partitions=1,
            resources=evaluated.resources,
        )
        save_model_outputs(output_dir, result, {"model": model, "features": selected})
        return result


def _state_out_of_fold_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    candidates: list[str],
    max_features: int,
    max_depth: int,
    requested_folds: int,
    seed: int,
) -> dict[int, np.ndarray]:
    counts = labels.value_counts()
    can_stratify = len(counts) > 1 and int(counts.min()) >= 2
    folds = min(requested_folds, int(counts.min())) if can_stratify else min(requested_folds, len(labels))
    if folds < 2:
        raise ValueError("scaling-aware StateDT training requires at least two validation folds")
    predictions = {feature_count: np.empty(len(labels), dtype=object) for feature_count in range(1, max_features + 1)}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed) if can_stratify else KFold(n_splits=folds, shuffle=True, random_state=seed)
    split_iterator = splitter.split(features, labels) if can_stratify else splitter.split(features)
    for fold, (fit_positions, validation_positions) in enumerate(split_iterator):
        X_fit = features.iloc[fit_positions]
        y_fit = labels.iloc[fit_positions]
        ranked = select_top_k_features(X_fit, y_fit, max_features, seed + fold, candidates)
        for feature_count in range(1, max_features + 1):
            selected = ranked[:feature_count]
            model = fit_tree(X_fit[selected], y_fit, max_depth, seed + fold)
            predictions[feature_count][validation_positions] = model.predict(features.iloc[validation_positions][selected]).astype(str)
    return predictions


def save_model_outputs(
    output_dir: Path,
    result: ModelResult,
    model_payload,
) -> None:
    save_model_artifacts(output_dir, result, model_payload)


def run_statedt(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return StateDT(config).run(output_dir)


def synthetic_example() -> None:
    rows = pd.DataFrame(
        [
            {"Total Length of Fwd Packet": 1000, "Packet Length Max": 1100, "SYN Flag Count": 0},
            {"Total Length of Fwd Packet": 2500, "Packet Length Max": 1300, "SYN Flag Count": 3},
            {"Total Length of Fwd Packet": 5200, "Packet Length Max": 1300, "SYN Flag Count": 3},
            {"Total Length of Fwd Packet": 5200, "Packet Length Max": 1100, "SYN Flag Count": 6},
        ]
    )
    labels = pd.Series(["Benign", "Benign", "Volumetric Attack", "SYN Attack"])
    model = fit_tree(rows, labels, 3, 42)
    config = object.__new__(StateDT)
    compiled = StateDT.compile(config, model, list(rows.columns))
    sample = rows.iloc[2]
    prediction, path = StateDT.predict_compiled(config, compiled, sample)
    software_prediction, software_path = predict_with_path(model, pd.DataFrame([sample]))
    print({"software_prediction": software_prediction[0], "statedt_prediction": prediction})
    print({"software_path": software_path[0], "statedt_path": path})
    print({"agreement": software_prediction[0] == prediction and software_path[0] == path})
