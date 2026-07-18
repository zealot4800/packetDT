from __future__ import annotations

import math
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_full_flow_dataset
from .resources import TargetProfile, estimate_statedt_resources
from .tree import (
    ModelResult,
    calculate_macro_f1,
    extract_thresholds_by_feature,
    fit_tree,
    predict_with_path,
    select_top_k_features,
    serialize_tree,
    write_json,
)


STATEFUL_PATTERNS = (
    "packet",
    "packets",
    "bytes",
    "length",
    "duration",
    "iat",
    "flag",
    "fwd",
    "bwd",
    "mean",
    "rate",
    "variance",
    "std",
    "max",
    "min",
    "total",
    "count",
)


@dataclass(frozen=True)
class FeatureStateSpec:
    feature: str
    thresholds: tuple[float, ...]
    encoding: str
    logical_bits: int
    incrementally_supported: bool


@dataclass
class CompiledStateDT:
    features: list[str]
    feature_specs: dict[str, FeatureStateSpec]
    tree: dict[str, Any]


def detect_stateful_features(features: list[str]) -> list[str]:
    return [feature for feature in features if any(token in feature.lower() for token in STATEFUL_PATTERNS)]


def logical_bits_for_bins(thresholds: tuple[float, ...]) -> int:
    return 0 if not thresholds else max(1, math.ceil(math.log2(len(thresholds) + 1)))


def encoding_for_feature(feature: str) -> tuple[str, bool]:
    lower = feature.lower()
    if "max" in lower or "maximum" in lower:
        return "maximum_threshold", True
    if "min" in lower or "minimum" in lower:
        return "minimum_threshold", True
    if any(token in lower for token in ["count", "packet", "bytes", "total", "flag"]):
        return "saturating", True
    return "final_value_bins", False


def encode_value(value: float, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return index
    return len(thresholds)


class StateDT:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def compile(self, model, features: list[str]) -> CompiledStateDT:
        thresholds = extract_thresholds_by_feature(model, features)
        specs = {}
        for feature, values in thresholds.items():
            encoding, supported = encoding_for_feature(feature)
            specs[feature] = FeatureStateSpec(feature, values, encoding, logical_bits_for_bins(values), supported)
        return CompiledStateDT(features=features, feature_specs=specs, tree=serialize_tree(model, features))

    def predict_compiled(self, compiled: CompiledStateDT, sample: pd.Series) -> tuple[Any, list[int]]:
        states = {
            # sklearn's tree predictor converts input features to float32 before
            # comparing them with the tree's float64 thresholds. Mirror that
            # conversion so values near a threshold take the identical branch.
            feature: encode_value(float(np.float32(sample[feature])), spec.thresholds)
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
            threshold_index = compiled.feature_specs[node["feature"]].thresholds.index(node["threshold"])
            node_id = node["left_child"] if states[node["feature"]] <= threshold_index else node["right_child"]

    def run(self, output_dir: Path) -> ModelResult:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        split = load_full_flow_dataset(self.config.dataset)
        candidates = detect_stateful_features(list(split.X_train.columns)) if self.config.statedt.stateful_only else list(split.X_train.columns)
        if self.config.statedt.explicit_features:
            candidates = [feature for feature in self.config.statedt.explicit_features if feature in split.X_train.columns]
        selected = select_top_k_features(split.X_train, split.y_train, self.config.statedt.max_features, self.config.seed, candidates)
        model = fit_tree(split.X_train[selected], split.y_train, self.config.statedt.max_depth, self.config.seed)
        compiled = self.compile(model, selected)
        software_predictions, software_paths = predict_with_path(model, split.X_test[selected])
        compiled_pairs = [self.predict_compiled(compiled, row) for _, row in split.X_test[selected].iterrows()]
        compiled_predictions = np.array([prediction for prediction, _ in compiled_pairs])
        compiled_paths = [path for _, path in compiled_pairs]
        if not np.array_equal(compiled_predictions, software_predictions) or compiled_paths != software_paths:
            raise RuntimeError("StateDT exact compilation does not reproduce the software tree")

        feature_state_bits = sum(spec.logical_bits for spec in compiled.feature_specs.values())
        metadata_bits = self.config.statedt.fingerprint_bits + self.config.statedt.generation_bits + self.config.statedt.valid_bits
        target = TargetProfile.from_config(self.config.target)
        best_resources = estimate_statedt_resources(target, model.tree_.node_count, feature_state_bits, metadata_bits, len(selected))
        macro_f1 = calculate_macro_f1(split.y_test, software_predictions)
        result = ModelResult(
            model="StateDT",
            dataset=self.config.dataset.name,
            target=self.config.target.name,
            macro_f1=macro_f1,
            max_depth=model.get_depth(),
            num_features=len(selected),
            num_partitions=1,
            feature_state_bits=feature_state_bits,
            metadata_bits=metadata_bits,
            logical_entry_bits=best_resources.logical_entry_bits,
            aligned_entry_bits=best_resources.aligned_entry_bits,
            estimated_flow_capacity=best_resources.estimated_flow_capacity,
            feature_table_entries=best_resources.feature_table_entries,
            tree_table_entries=best_resources.tree_table_entries,
            total_table_entries=best_resources.total_table_entries,
            tcam_blocks=best_resources.tcam_blocks,
            tcam_stages=best_resources.tcam_stages,
            tcam_capacity_mb=best_resources.tcam_capacity_mb,
            tcam_memory_mb=best_resources.tcam_memory_mb,
            register_words_per_flow=best_resources.register_words_per_flow,
        )
        save_model_outputs(output_dir, result, {"model": model, "features": selected}, compiled)
        return result


def save_model_outputs(output_dir: Path, result: ModelResult, model_payload, compiled: CompiledStateDT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.metrics_row()]).to_csv(output_dir / "metrics.csv", index=False)
    _remove_resource_csv(output_dir)
    pd.DataFrame([asdict(spec) for spec in compiled.feature_specs.values()]).to_csv(output_dir / "feature_state.csv", index=False)
    (output_dir / "summary.json").unlink(missing_ok=True)
    write_json(str(output_dir / "compiler.json"), {"features": compiled.features, "tree": compiled.tree, "feature_specs": {key: asdict(value) for key, value in compiled.feature_specs.items()}})
    with open(output_dir / "model.pkl", "wb") as handle:
        pickle.dump(model_payload, handle)


def _remove_resource_csv(output_dir: Path) -> None:
    resource_path = output_dir / "resource.csv"
    if resource_path.exists():
        resource_path.unlink()


def run_statedt(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return StateDT(config).run(output_dir)


def synthetic_example() -> None:
    rows = pd.DataFrame(
        [
            {"Total Bytes": 1000, "Maximum Packet Length": 1100, "SYN Count": 0},
            {"Total Bytes": 2500, "Maximum Packet Length": 1300, "SYN Count": 3},
            {"Total Bytes": 5200, "Maximum Packet Length": 1300, "SYN Count": 3},
            {"Total Bytes": 5200, "Maximum Packet Length": 1100, "SYN Count": 6},
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
