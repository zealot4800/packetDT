from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_full_flow_dataset
from .resources import (
    ResourceReport,
    TargetProfile,
    aligned_register_entry_bits,
    estimate_statedt_resources,
    estimated_flow_capacity,
)
from .tree import (
    ModelResult,
    calculate_macro_f1,
    extract_thresholds_by_feature,
    fit_tree,
    predict_with_path,
    save_model_artifacts,
    select_top_k_features,
    serialize_tree,
    write_json,
)


STATEFUL_FEATURES = (
    "Packet Length Max",
    "PSH Flag Count",
    "Total Length of Fwd Packet",
    "FIN Flag Count",
    "Fwd Packet Length Max",
    "Total Bwd packets",
)

FEATURE_SEMANTICS = {
    "Packet Length Max": "max",
    "PSH Flag Count": "counter",
    "Total Length of Fwd Packet": "sum",
    "FIN Flag Count": "counter",
    "Fwd Packet Length Max": "max",
    "Total Bwd packets": "counter",
}

SEMANTICS_VERSION = 1
ORIGINAL_FEATURE_STATE_BITS = 16
STATE_LAYOUT_VERSION = 2
DEFAULT_FINGERPRINT_BITS = 16
VALID_BITS = 1
DIRECTION_BITS = 1
STATE_STATUS_CODES = {
    "MATCH": 0,
    "ALLOCATED": 1,
    "FALLBACK_COLLISION": 2,
    "NOT_PROCESSED": 3,
}
STATE_COUNTER_NAMES = (
    "allocations",
    "fingerprint_mismatches",
    "collisions",
    "fallbacks",
)


@dataclass(frozen=True)
class FeatureStateSpec:
    feature: str
    semantics: str
    representation: str
    thresholds: tuple[float, ...]
    logical_bits: int
    state_count: int
    cap: int | None


@dataclass
class CompiledStateDT:
    feature_specs: dict[str, FeatureStateSpec]
    tree: dict[str, Any]
    fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS
    direction_bits: int = DIRECTION_BITS
    valid_bits: int = VALID_BITS

    @property
    def feature_state_bits(self) -> int:
        return sum(spec.logical_bits for spec in self.feature_specs.values())

    @property
    def metadata_bits(self) -> int:
        return self.fingerprint_bits + self.direction_bits + self.valid_bits

    @property
    def entry_bits(self) -> int:
        return self.feature_state_bits + self.metadata_bits

    def state_layout(self) -> dict[str, Any]:
        offset = 0
        fields: list[dict[str, Any]] = []
        for feature, spec in self.feature_specs.items():
            width = spec.logical_bits
            fields.append({
                "name": feature,
                "role": "feature_state",
                "feature": feature,
                "representation": spec.representation,
                "offset": offset,
                "lsb": offset,
                "msb": offset + width - 1 if width else None,
                "width": width,
                "state_count": spec.state_count,
                "initial_state": 0,
            })
            offset += width

        for name, role, width in (
            ("flow_fingerprint", "fingerprint", self.fingerprint_bits),
            ("direction", "direction", self.direction_bits),
            ("valid", "valid", self.valid_bits),
        ):
            field = {
                "name": name,
                "role": role,
                "offset": offset,
                "lsb": offset,
                "msb": offset + width - 1,
                "width": width,
                "initial_state": 0,
            }
            if role == "direction":
                field["semantics"] = "first_packet_low_to_high"
            fields.append(field)
            offset += width
        assert offset == self.entry_bits
        return {
            "version": STATE_LAYOUT_VERSION,
            "bit_order": "lsb0",
            "feature_state_bits": self.feature_state_bits,
            "metadata_bits": self.metadata_bits,
            "entry_bits": self.entry_bits,
            "fingerprint_bits": self.fingerprint_bits,
            "direction_bits": self.direction_bits,
            "valid_bits": self.valid_bits,
            "status_codes": dict(STATE_STATUS_CODES),
            "counters": list(STATE_COUNTER_NAMES),
            "collision_policy": "explicit_fallback_no_state",
            "fields": fields,
        }

    def to_dict(self) -> dict[str, Any]:
        classes = self.tree.get("classes", [])
        return {
            "model": "StateDT",
            "compiler": "decision-sufficient-state",
            "semantics_version": SEMANTICS_VERSION,
            "feature_state_bits": self.feature_state_bits,
            "state_layout": self.state_layout(),
            "features": list(self.feature_specs),
            "class_ids": {label: index for index, label in enumerate(classes)},
            "feature_specs": {
                feature: asdict(spec) for feature, spec in self.feature_specs.items()
            },
            "tree": self.tree,
        }


@dataclass
class TrainedStateDT:
    model: Any
    features: list[str]
    compiled: CompiledStateDT


@dataclass(frozen=True)
class EquivalenceReport:
    samples: int
    predicates: int
    predicate_checks: int
    predicate_mismatches: int
    prediction_agreement: float
    path_agreement: float
    exact: bool


@dataclass
class EvaluatedStateDT:
    trained: TrainedStateDT
    predictions: np.ndarray
    macro_f1: float
    resources: ResourceReport
    equivalence: EquivalenceReport


def logical_bits_for_states(state_count: int) -> int:
    if state_count <= 0:
        raise ValueError("state_count must be positive")
    return 0 if state_count == 1 else math.ceil(math.log2(state_count))


def threshold_region(value: float, thresholds: tuple[float, ...]) -> int:
    value32 = float(np.float32(value))
    return int(np.searchsorted(np.asarray(thresholds), value32, side="left"))


def synthesize_feature_spec(feature: str, thresholds: Iterable[float]) -> FeatureStateSpec:
    if feature not in FEATURE_SEMANTICS:
        raise ValueError(
            f"unsupported StateDT stateful feature {feature!r}; define its online update semantics first"
        )
    ordered = tuple(sorted(set(float(value) for value in thresholds)))
    if not ordered:
        raise ValueError(f"cannot synthesize StateDT state for {feature!r} without tree thresholds")
    semantics = FEATURE_SEMANTICS[feature]
    if semantics == "max":
        state_count = len(ordered) + 1
        return FeatureStateSpec(
            feature, semantics, "threshold_region", ordered,
            logical_bits_for_states(state_count), state_count, None,
        )
    cap = math.floor(max(ordered)) + 1
    if cap < 0:
        raise ValueError(f"{semantics} feature {feature!r} has an invalid negative cap {cap}")
    state_count = cap + 1
    return FeatureStateSpec(
        feature, semantics, "capped_integer", ordered,
        logical_bits_for_states(state_count), state_count, cap,
    )


def encode_concrete_value(value: float, spec: FeatureStateSpec) -> int:
    value32 = float(np.float32(value))
    if spec.representation == "threshold_region":
        return threshold_region(value32, spec.thresholds)
    _validate_nonnegative_integer(value32, spec.feature)
    assert spec.cap is not None
    return min(spec.cap, int(value32))


def update_abstract_state(spec: FeatureStateSpec, state: int, packet_value: float) -> int:
    if state < 0 or state >= spec.state_count:
        raise ValueError(f"abstract state for {spec.feature!r} is outside its representation")
    if spec.representation == "threshold_region":
        return max(state, threshold_region(packet_value, spec.thresholds))
    _validate_nonnegative_integer(packet_value, f"{spec.feature} increment")
    assert spec.cap is not None
    return min(spec.cap, state + int(packet_value))


def predicate_from_abstract(spec: FeatureStateSpec, state: int, threshold: float) -> bool:
    if spec.representation == "threshold_region":
        try:
            threshold_index = spec.thresholds.index(float(threshold))
        except ValueError as exc:
            raise ValueError(f"threshold {threshold} is not compiled for {spec.feature!r}") from exc
        return state <= threshold_index
    return state <= threshold


def _validate_nonnegative_integer(value: float, feature: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"StateDT {feature!r} must contain finite non-negative values")
    if not float(value).is_integer():
        raise ValueError(f"StateDT {feature!r} must contain integer values")


def _validate_feature_domain(frame: pd.DataFrame, features: Iterable[str]) -> None:
    for feature in features:
        if feature not in FEATURE_SEMANTICS:
            raise ValueError(
                f"unsupported StateDT stateful feature {feature!r}; define its online update semantics first"
            )
        values = frame[feature].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"StateDT {feature!r} must contain finite non-negative values")
        if FEATURE_SEMANTICS[feature] in {"counter", "sum"} and not np.equal(values, np.floor(values)).all():
            raise ValueError(f"StateDT {feature!r} must contain integer values")


class StateDT:
    """Compile a fixed tree into exact decision-observable online state.

    For supported updates U, the compiler constructs alpha and U_hat so that
    alpha(U(s, p)) == U_hat(alpha(s), p) with respect to every predicate in the
    fitted tree. Consequently the original and compiled trees follow the same
    path and make the same decision; the fitted tree is never modified.
    """

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config

    def compile(self, model: Any, features: list[str]) -> CompiledStateDT:
        thresholds = extract_thresholds_by_feature(model, features)
        specs = {
            feature: synthesize_feature_spec(feature, thresholds[feature])
            for feature in features
            if feature in thresholds
        }
        fingerprint_bits = (
            self.config.statedt.fingerprint_bits if self.config else DEFAULT_FINGERPRINT_BITS
        )
        direction_bits = self.config.statedt.direction_bits if self.config else DIRECTION_BITS
        valid_bits = self.config.statedt.valid_bits if self.config else VALID_BITS
        return CompiledStateDT(
            specs,
            serialize_tree(model, features),
            fingerprint_bits=fingerprint_bits,
            direction_bits=direction_bits,
            valid_bits=valid_bits,
        )

    def predict_compiled(self, compiled: CompiledStateDT, sample: pd.Series) -> tuple[str, list[int]]:
        states = {
            feature: encode_concrete_value(float(sample[feature]), spec)
            for feature, spec in compiled.feature_specs.items()
        }
        nodes = {node["node_id"]: node for node in compiled.tree["nodes"]}
        node_id = 0
        path: list[int] = []
        while True:
            node = nodes[node_id]
            path.append(node_id)
            if node["is_leaf"]:
                return str(node["prediction"]), path
            spec = compiled.feature_specs[node["feature"]]
            go_left = predicate_from_abstract(spec, states[node["feature"]], node["threshold"])
            node_id = node["left_child"] if go_left else node["right_child"]

    def train(self, split: Any) -> TrainedStateDT:
        if self.config is None:
            raise ValueError("StateDT training requires an experiment configuration")
        available = set(split.X_train.columns)
        explicit = list(self.config.statedt.explicit_features)
        if explicit:
            missing = [feature for feature in explicit if feature not in available]
            if missing:
                raise ValueError(f"StateDT explicit feature(s) not found: {', '.join(missing)}")
            unsupported = [feature for feature in explicit if feature not in FEATURE_SEMANTICS]
            if unsupported:
                raise ValueError(
                    "unsupported StateDT stateful feature(s): " + ", ".join(unsupported)
                    + "; define online update semantics first"
                )
            candidates = [feature for feature in STATEFUL_FEATURES if feature in explicit]
        else:
            candidates = [feature for feature in STATEFUL_FEATURES if feature in available]
        if not candidates:
            raise ValueError("StateDT has no supported stateful candidate features")

        limit = min(self.config.statedt.max_features, len(candidates))
        selected = (
            candidates
            if limit == len(candidates)
            else select_top_k_features(split.X_train, split.y_train, limit, self.config.seed, candidates)
        )
        _validate_feature_domain(
            pd.concat([split.X_train[selected], split.X_test[selected]], ignore_index=True), selected
        )
        model = fit_tree(
            split.X_train[selected], split.y_train,
            self.config.statedt.max_depth, self.config.seed,
        )
        compiled = self.compile(model, selected)
        return TrainedStateDT(model, selected, compiled)

    def validate_exact_equivalence(
        self, trained: TrainedStateDT, samples: pd.DataFrame
    ) -> tuple[EquivalenceReport, np.ndarray]:
        selected = trained.features
        compiled = trained.compiled
        original_predictions, original_paths = predict_with_path(
            trained.model, samples[selected]
        )
        predicate_mismatches = 0
        predicate_count = sum(len(spec.thresholds) for spec in compiled.feature_specs.values())
        for _, sample in samples.iterrows():
            for feature, spec in compiled.feature_specs.items():
                concrete = float(np.float32(sample[feature]))
                abstract = encode_concrete_value(concrete, spec)
                for threshold in spec.thresholds:
                    if (concrete <= threshold) != predicate_from_abstract(spec, abstract, threshold):
                        predicate_mismatches += 1

        compiled_predictions: list[str] = []
        compiled_paths: list[list[int]] = []
        for _, sample in samples.iterrows():
            prediction, path = self.predict_compiled(compiled, sample)
            compiled_predictions.append(prediction)
            compiled_paths.append(path)
        original_strings = np.asarray(original_predictions).astype(str)
        compiled_array = np.asarray(compiled_predictions)
        prediction_agreement = float(np.mean(original_strings == compiled_array)) if len(samples) else 1.0
        path_agreement = float(np.mean([a == b for a, b in zip(original_paths, compiled_paths)])) if len(samples) else 1.0
        exact = predicate_mismatches == 0 and prediction_agreement == 1.0 and path_agreement == 1.0
        report = EquivalenceReport(
            samples=len(samples), predicates=predicate_count,
            predicate_checks=len(samples) * predicate_count,
            predicate_mismatches=predicate_mismatches,
            prediction_agreement=prediction_agreement,
            path_agreement=path_agreement, exact=exact,
        )
        if not exact:
            raise ValueError(
                "StateDT exact-equivalence validation failed: "
                f"predicate_mismatches={predicate_mismatches}, "
                f"prediction_agreement={prediction_agreement:.6f}, "
                f"path_agreement={path_agreement:.6f}"
            )
        return report, compiled_array

    def evaluate(self, split: Any) -> EvaluatedStateDT:
        if self.config is None:
            raise ValueError("StateDT evaluation requires an experiment configuration")
        trained = self.train(split)
        equivalence, predictions = self.validate_exact_equivalence(trained, split.X_test)
        resources = estimate_statedt_resources(
            TargetProfile.from_config(self.config.target), trained.model.tree_.node_count,
            trained.compiled.feature_state_bits,
            len(trained.compiled.feature_specs),
            fingerprint_bits=trained.compiled.fingerprint_bits,
            direction_bits=trained.compiled.direction_bits,
            valid_bits=trained.compiled.valid_bits,
        )
        return EvaluatedStateDT(
            trained, predictions, calculate_macro_f1(split.y_test, predictions),
            resources, equivalence,
        )

    def run(self, output_dir: Path) -> ModelResult:
        if self.config is None:
            raise ValueError("StateDT run requires an experiment configuration")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        split = load_full_flow_dataset(self.config.dataset)
        evaluated = self.evaluate(split)
        trained = evaluated.trained
        result = ModelResult.from_resources(
            model="StateDT", dataset=self.config.dataset.name,
            target=self.config.target.name, seed=self.config.seed,
            macro_f1=evaluated.macro_f1, max_depth=trained.model.get_depth(),
            num_features=len(trained.compiled.feature_specs),
            test_samples=len(split.y_test), resources=evaluated.resources,
        )
        save_model_artifacts(
            output_dir, result, {"model": trained.model, "features": trained.features}
        )
        compiler_payload = trained.compiled.to_dict()
        write_json(output_dir / "compiler.json", compiler_payload)
        write_json(
            output_dir / "state_report.json",
            self._state_report(evaluated),
        )
        (output_dir / "state_report.txt").write_text(
            self._text_report(evaluated), encoding="utf-8"
        )
        return result

    def _state_report(self, evaluated: EvaluatedStateDT) -> dict[str, Any]:
        assert self.config is not None
        compiled = evaluated.trained.compiled
        # The deployed StateDT data plane stores each of these flow features in
        # a 16-bit register. TargetConfig.feature_width is the generic tree-key
        # width and must not be substituted for this original state width.
        original_bits = len(compiled.feature_specs) * ORIGINAL_FEATURE_STATE_BITS
        synthesized_bits = compiled.feature_state_bits
        resources = evaluated.resources
        metadata_bits = int(resources.metadata_bits or 0)
        original_aligned_bits = aligned_register_entry_bits(
            original_bits + metadata_bits, self.config.target.register_word_bits
        )
        original_capacity = estimated_flow_capacity(
            self.config.target.state_memory_mb, original_aligned_bits
        )
        return {
            "model": "StateDT",
            "compiler": "decision-sufficient-state",
            "dataset": self.config.dataset.name,
            "target": self.config.target.name,
            "seed": self.config.seed,
            "tree_depth": evaluated.trained.model.get_depth(),
            "tree_nodes": evaluated.trained.model.tree_.node_count,
            "stateful_features_used": list(compiled.feature_specs),
            "state_layout": compiled.state_layout(),
            "original_feature_state_bits": original_bits,
            "decision_sufficient_feature_state_bits": synthesized_bits,
            "logical_state_reduction": 0.0 if original_bits == 0 else 1.0 - synthesized_bits / original_bits,
            "original_aligned_entry_bits": original_aligned_bits,
            "original_estimated_flow_capacity": original_capacity,
            "per_feature": {
                feature: {
                    "semantics": spec.semantics,
                    "original_bits": ORIGINAL_FEATURE_STATE_BITS,
                    "synthesized_bits": spec.logical_bits,
                    "representation": spec.representation,
                    "thresholds": list(spec.thresholds),
                    "state_count": spec.state_count,
                    "cap": spec.cap,
                }
                for feature, spec in compiled.feature_specs.items()
            },
            "exact_equivalence": asdict(evaluated.equivalence),
            "resource_usage": {
                "metadata_bits": resources.metadata_bits,
                "logical_entry_bits": resources.logical_entry_bits,
                "aligned_entry_bits": resources.aligned_entry_bits,
                "register_words_per_flow": resources.register_words_per_flow,
                "estimated_flow_capacity": resources.estimated_flow_capacity,
                "capacity_improvement": (
                    0.0 if original_capacity == 0
                    else int(resources.estimated_flow_capacity or 0) / original_capacity
                ),
            },
        }

    def _text_report(self, evaluated: EvaluatedStateDT) -> str:
        report = self._state_report(evaluated)
        exact = report["exact_equivalence"]
        resources = report["resource_usage"]
        rows = [
            "StateDT Decision-Sufficient State Report", "========================================", "",
            f"Dataset: {report['dataset']}", f"Target: {report['target']}", "",
            f"Tree depth: {report['tree_depth']}", f"Tree nodes: {report['tree_nodes']}",
            f"Stateful features used: {', '.join(report['stateful_features_used'])}", "",
        ]
        for feature, item in report["per_feature"].items():
            rows.append(
                f"{feature}: {item['semantics']}, {item['original_bits']} -> "
                f"{item['synthesized_bits']} bits ({item['representation']})"
            )
        rows.extend([
            "", f"Predicate checks: {exact['predicate_checks']}",
            f"Predicate mismatches: {exact['predicate_mismatches']}",
            f"Prediction agreement: {exact['prediction_agreement']:.2%}",
            f"Path agreement: {exact['path_agreement']:.2%}", "",
            f"Original feature state bits: {report['original_feature_state_bits']}",
            f"Decision-sufficient state bits: {report['decision_sufficient_feature_state_bits']}",
            f"Metadata bits: {resources['metadata_bits']}",
            f"Logical entry bits: {resources['logical_entry_bits']}",
            f"Aligned entry bits: {resources['aligned_entry_bits']}",
            f"Register words / flow: {resources['register_words_per_flow']}",
            f"Estimated flow capacity: {resources['estimated_flow_capacity']}", "",
            f"Original aligned entry bits: {report['original_aligned_entry_bits']}",
            f"Original estimated flow capacity: {report['original_estimated_flow_capacity']}",
            f"Capacity improvement: {resources['capacity_improvement']:.3f}x", "",
        ])
        return "\n".join(rows)


def run_statedt(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return StateDT(config).run(output_dir)


def synthetic_example() -> None:
    rows = pd.DataFrame([
        {"Total Length of Fwd Packet": 1000, "Packet Length Max": 1100},
        {"Total Length of Fwd Packet": 2500, "Packet Length Max": 1300},
        {"Total Length of Fwd Packet": 5200, "Packet Length Max": 1300},
        {"Total Length of Fwd Packet": 5200, "Packet Length Max": 1100},
    ])
    labels = pd.Series(["Benign", "Benign", "Volumetric Attack", "SYN Attack"])
    features = list(rows.columns)
    model = fit_tree(rows, labels, 3, 42)
    statedt = StateDT()
    compiled = statedt.compile(model, features)
    sample = rows.iloc[2]
    prediction, path = statedt.predict_compiled(compiled, sample)
    software_prediction, software_path = predict_with_path(model, pd.DataFrame([sample]))
    print({"software_prediction": software_prediction[0], "statedt_prediction": prediction})
    print({"software_path": software_path[0], "statedt_path": path})
    print({"agreement": str(software_prediction[0]) == prediction and software_path[0] == path})
