from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


METRICS_COLUMNS = [
    "dataset",
    "target",
    "model",
    "seed",
    "scope",
    "flow_count",
    "flows_used",
    "test_samples",
    "macro_f1",
    "max_depth",
    "num_features",
    "num_partitions",
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
    "within_tcam_capacity",
    "within_stage_budget",
    "target_feasible",
]


@dataclass
class ModelResult:
    model: str
    dataset: str
    target: str
    seed: int
    macro_f1: float | None
    max_depth: int | None
    num_features: int | None
    test_samples: int | None = None
    num_partitions: int | None = None
    feature_state_bits: int | None = None
    metadata_bits: int | None = None
    logical_entry_bits: int | None = None
    aligned_entry_bits: int | None = None
    estimated_flow_capacity: int | None = None
    feature_table_entries: int | None = None
    tree_table_entries: int | None = None
    total_table_entries: int | None = None
    tcam_blocks: int | None = None
    tcam_stages: int | None = None
    tcam_capacity_mb: float | None = None
    tcam_memory_mb: float | None = None
    register_words_per_flow: int | None = None
    within_tcam_capacity: bool | None = None
    within_stage_budget: bool | None = None
    target_feasible: bool | None = None

    def metrics_row(self) -> dict[str, Any]:
        values = {key: _clean(value) for key, value in asdict(self).items()}
        return {
            "dataset": values["dataset"],
            "target": values["target"],
            "model": values["model"],
            "seed": values["seed"],
            "scope": "evaluation",
            "flow_count": "",
            "flows_used": "",
            "test_samples": values["test_samples"],
            "macro_f1": values["macro_f1"],
            "max_depth": values["max_depth"],
            "num_features": values["num_features"],
            "num_partitions": values["num_partitions"],
            **{
                column: values[column]
                for column in METRICS_COLUMNS
                if column in values
                and column
                not in {
                    "dataset",
                    "target",
                    "model",
                    "seed",
                    "test_samples",
                    "macro_f1",
                    "max_depth",
                    "num_features",
                    "num_partitions",
                }
            },
        }

    @classmethod
    def from_resources(
        cls,
        *,
        model: str,
        dataset: str,
        target: str,
        seed: int,
        macro_f1: float,
        max_depth: int,
        num_features: int,
        test_samples: int,
        resources: Any,
        num_partitions: int = 1,
    ) -> "ModelResult":
        resource_fields = {
            name: getattr(resources, name)
            for name in cls.__dataclass_fields__
            if hasattr(resources, name)
        }
        return cls(
            model=model,
            dataset=dataset,
            target=target,
            seed=seed,
            macro_f1=macro_f1,
            max_depth=max_depth,
            num_features=num_features,
            test_samples=test_samples,
            num_partitions=num_partitions,
            **resource_fields,
        )


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def create_decision_tree(max_depth: int, seed: int) -> DecisionTreeClassifier:
    max_leaf_nodes = int(2**max_depth) if max_depth <= 13 else 1024
    return DecisionTreeClassifier(
        random_state=seed,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        criterion="entropy",
        class_weight="balanced",
    )


def fit_tree(X_train: pd.DataFrame, y_train: pd.Series, max_depth: int, seed: int, sample_weight=None) -> DecisionTreeClassifier:
    if X_train.empty:
        raise ValueError("cannot train decision tree with an empty feature matrix")
    model = create_decision_tree(max_depth, seed)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def select_top_k_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int,
    seed: int,
    candidate_features: list[str] | None = None,
) -> list[str]:
    candidates = candidate_features or list(X_train.columns)
    if not candidates:
        raise ValueError("no candidate features available")
    selector = fit_tree(X_train[candidates], y_train, min(5, max(1, len(candidates))), seed)
    ranked = pd.DataFrame(
        {"feature": candidates, "importance": selector.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    return ranked.head(max_features)["feature"].tolist()


def extract_tree_nodes(model: DecisionTreeClassifier, feature_names: list[str]) -> list[dict[str, Any]]:
    tree = model.tree_
    nodes = []
    for node_id in range(tree.node_count):
        feature_index = int(tree.feature[node_id])
        is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
        nodes.append(
            {
                "node_id": node_id,
                "feature": None if is_leaf else feature_names[feature_index],
                "threshold": None if is_leaf else float(tree.threshold[node_id]),
                "left_child": int(tree.children_left[node_id]),
                "right_child": int(tree.children_right[node_id]),
                "prediction": str(model.classes_[tree.value[node_id][0].argmax()]),
                "is_leaf": bool(is_leaf),
            }
        )
    return nodes


def extract_thresholds_by_feature(model: DecisionTreeClassifier, feature_names: list[str]) -> dict[str, tuple[float, ...]]:
    thresholds: dict[str, set[float]] = {}
    for feature_index, threshold in zip(model.tree_.feature, model.tree_.threshold):
        if feature_index >= 0:
            thresholds.setdefault(feature_names[int(feature_index)], set()).add(float(threshold))
    return {feature: tuple(sorted(values)) for feature, values in sorted(thresholds.items())}


def predict_with_path(model: DecisionTreeClassifier, X: pd.DataFrame) -> tuple[np.ndarray, list[list[int]]]:
    predictions = model.predict(X)
    indicator = model.decision_path(X)
    paths = []
    for row_id in range(X.shape[0]):
        paths.append(indicator.indices[indicator.indptr[row_id] : indicator.indptr[row_id + 1]].tolist())
    return predictions, paths


def calculate_macro_f1(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def serialize_tree(model: DecisionTreeClassifier, feature_names: list[str]) -> dict[str, Any]:
    return {
        "comparison_semantics": "left: value <= threshold, right: value > threshold",
        "features": feature_names,
        "classes": [str(label) for label in model.classes_],
        "nodes": extract_tree_nodes(model, feature_names),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def write_metrics_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    replace_scope: str,
) -> None:
    if not rows:
        return
    _validate_metrics_rows(rows, expected_scope=replace_scope)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame(columns=METRICS_COLUMNS)
    if output_path.exists():
        candidate = pd.read_csv(output_path)
        if list(candidate.columns) == METRICS_COLUMNS:
            existing = candidate[candidate["scope"] != replace_scope].copy()

    new_rows = pd.DataFrame(rows, columns=METRICS_COLUMNS)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    scope_order = combined["scope"].map({"evaluation": 0, "flow_scaling": 1}).fillna(2)
    combined = (
        combined.assign(__scope_order=scope_order)
        .sort_values(["__scope_order", "flow_count"], na_position="first")
        .drop(columns="__scope_order")
        .reset_index(drop=True)
    )
    _validate_metrics_rows(combined.to_dict("records"))

    temporary_path = output_path.with_suffix(".tmp.csv")
    combined.to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(output_path)
    remove_auxiliary_csvs(output_path.parent)


def _validate_metrics_rows(
    rows: list[dict[str, Any]],
    expected_scope: str | None = None,
) -> None:
    for position, row in enumerate(rows, start=1):
        scope = str(row.get("scope", ""))
        if scope not in {"evaluation", "flow_scaling"}:
            raise ValueError(f"metrics row {position} has invalid scope: {scope!r}")
        if expected_scope is not None and scope != expected_scope:
            raise ValueError(f"metrics row {position} must use scope {expected_scope!r}")
        for field in ["dataset", "target", "model"]:
            if not str(row.get(field, "")).strip():
                raise ValueError(f"metrics row {position} has no {field}")
        seed = _required_int(row, "seed", position)
        if seed <= 0:
            raise ValueError(f"metrics row {position} seed must be positive")
        macro_f1 = _required_float(row, "macro_f1", position)
        if not 0.0 <= macro_f1 <= 1.0:
            raise ValueError(f"metrics row {position} macro_f1 must be in [0, 1]")
        for field in ["test_samples", "num_features", "num_partitions"]:
            if _required_int(row, field, position) <= 0:
                raise ValueError(f"metrics row {position} {field} must be positive")
        if _required_int(row, "max_depth", position) < 0:
            raise ValueError(f"metrics row {position} max_depth cannot be negative")
        if scope == "flow_scaling":
            flow_count = _required_int(row, "flow_count", position)
            flows_used = _required_int(row, "flows_used", position)
            if flow_count <= 0 or not 0 <= flows_used <= flow_count:
                raise ValueError(f"metrics row {position} has invalid flow counts")
        for field in [
            "feature_table_entries",
            "tree_table_entries",
            "total_table_entries",
            "tcam_blocks",
            "tcam_stages",
            "tcam_capacity_mb",
            "tcam_memory_mb",
        ]:
            if _required_float(row, field, position) < 0:
                raise ValueError(f"metrics row {position} {field} cannot be negative")
        feature_entries = _required_int(row, "feature_table_entries", position)
        tree_entries = _required_int(row, "tree_table_entries", position)
        total_entries = _required_int(row, "total_table_entries", position)
        if total_entries != feature_entries + tree_entries:
            raise ValueError(f"metrics row {position} has inconsistent table entries")
        within_tcam = _required_bool(row, "within_tcam_capacity", position)
        within_stages = _required_bool(row, "within_stage_budget", position)
        feasible = _required_bool(row, "target_feasible", position)
        if feasible != (within_tcam and within_stages):
            raise ValueError(f"metrics row {position} has inconsistent feasibility flags")


def _required_float(row: dict[str, Any], field: str, position: int) -> float:
    value = row.get(field)
    if value == "" or value is None or pd.isna(value):
        raise ValueError(f"metrics row {position} has no {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metrics row {position} {field} must be finite")
    return number


def _required_int(row: dict[str, Any], field: str, position: int) -> int:
    value = _required_float(row, field, position)
    if not value.is_integer():
        raise ValueError(f"metrics row {position} {field} must be an integer")
    return int(value)


def _required_bool(row: dict[str, Any], field: str, position: int) -> bool:
    value = row.get(field)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"metrics row {position} {field} must be boolean")


def remove_auxiliary_csvs(output_dir: Path) -> None:
    for path in output_dir.glob("*.csv"):
        if path.name != "metrics.csv":
            path.unlink()


def save_model_artifacts(output_dir: Path, result: ModelResult, model_payload: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_rows(
        output_dir / "metrics.csv",
        [result.metrics_row()],
        replace_scope="evaluation",
    )
    (output_dir / "summary.json").unlink(missing_ok=True)
    with open(output_dir / "model.pkl", "wb") as handle:
        pickle.dump(model_payload, handle)
