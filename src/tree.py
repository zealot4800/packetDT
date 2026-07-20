from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelResult:
    model: str
    dataset: str
    target: str
    macro_f1: float | None
    max_depth: int | None
    num_features: int | None
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

    def metrics_row(self) -> dict[str, Any]:
        return {key: _clean(value) for key, value in asdict(self).items()}


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


def count_leaf_nodes(model: DecisionTreeClassifier) -> int:
    tree = model.tree_
    return int(np.sum(tree.children_left == tree.children_right))


def serialize_tree(model: DecisionTreeClassifier, feature_names: list[str]) -> dict[str, Any]:
    return {
        "comparison_semantics": "left: value <= threshold, right: value > threshold",
        "features": feature_names,
        "classes": [str(label) for label in model.classes_],
        "nodes": extract_tree_nodes(model, feature_names),
    }


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
