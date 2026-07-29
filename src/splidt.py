from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .config import ExperimentConfig
from .data import DatasetSplit, load_partition_dataset, split_window_dataset
from .resources import TargetProfile, estimate_splidt_resources
from .tree import ModelResult, calculate_macro_f1, save_model_artifacts


@dataclass
class PartitionSubtree:
    """One routed subtree in a SpliDT partition."""

    model: DecisionTreeClassifier
    features: list[str]
    partition_depth: int
    children: dict[int, "PartitionSubtree"] = field(default_factory=dict)

    def _route_values(self, values: np.ndarray) -> tuple[int, object, bool]:
        tree = self.model.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature_indices = tree.feature
        thresholds = tree.threshold
        node_values = tree.value
        node = 0
        decisions = 0
        while children_left[node] != children_right[node]:
            if decisions == self.partition_depth:
                break
            feature_index = int(feature_indices[node])
            node = int(
                children_left[node]
                if values[feature_index] <= thresholds[node]
                else children_right[node]
            )
            if not 0 <= node < tree.node_count:
                raise RuntimeError(f"decision tree routed to invalid node {node}")
            decisions += 1
        prediction = self.model.classes_[node_values[node][0].argmax()]
        is_leaf = children_left[node] == children_right[node]
        return node, prediction, bool(is_leaf)

    def route(self, row: pd.Series) -> tuple[int, object, bool]:
        """Traverse this partition, returning boundary node, prediction, and leaf flag."""
        return self._route_values(row.loc[self.features].to_numpy())

    def route_array(
        self,
        row: np.ndarray,
        column_positions: dict[str, int],
    ) -> tuple[int, object, bool]:
        values = np.asarray([row[column_positions[feature]] for feature in self.features])
        return self._route_values(values)

    def deployed_node_count(self) -> int:
        tree = self.model.tree_
        stack = [(0, 0)]
        count = 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if depth < self.partition_depth and tree.children_left[node] != tree.children_right[node]:
                stack.append((int(tree.children_left[node]), depth + 1))
                stack.append((int(tree.children_right[node]), depth + 1))
        return count + sum(child.deployed_node_count() for child in self.children.values())

    def all_features(self) -> list[list[str]]:
        return [self.features] + [features for child in self.children.values() for features in child.all_features()]


class SpliDT:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _create_tree(self, max_depth: int) -> DecisionTreeClassifier:
        # The SpliDT artifact permits 4,096 leaves and uses the experiment seed
        # for every specialized subtree.
        return DecisionTreeClassifier(
            random_state=self.config.seed,
            max_depth=max_depth,
            max_leaf_nodes=min(2**max_depth, 4096),
            criterion="entropy",
            class_weight="balanced",
        )

    def _fit_subtree(
        self,
        windows: list[DatasetSplit],
        partition_index: int,
        flow_ids: list[str],
        remaining_depth: int,
    ) -> PartitionSubtree | None:
        split = windows[partition_index]
        if split.train_flow_ids is None:
            raise ValueError("SpliDT requires Flow ID values for routed partition training")
        positions = pd.Series(split.train_flow_ids.index, index=split.train_flow_ids).reindex(flow_ids).dropna().astype(int)
        if positions.empty:
            return None

        X_train = split.X_train.iloc[positions.to_numpy()].reset_index(drop=True)
        y_train = split.y_train.iloc[positions.to_numpy()].reset_index(drop=True)
        aligned_flow_ids = split.train_flow_ids.iloc[positions.to_numpy()].tolist()
        # Match the artifact: learn feature importance using the complete remaining
        # depth, choose this subtree's top-k, then retrain at the same depth.
        selector = self._create_tree(remaining_depth)
        selector.fit(X_train, y_train)
        ranked = sorted(
            zip(X_train.columns, selector.feature_importances_),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        features = [name for name, _ in ranked[: self.config.splidt.features_per_partition]]
        model = self._create_tree(remaining_depth)
        model.fit(X_train[features], y_train)
        partition_depth = self.config.splidt.partition_sizes[partition_index]
        subtree = PartitionSubtree(model=model, features=features, partition_depth=partition_depth)

        if partition_index + 1 >= len(windows) or remaining_depth <= partition_depth:
            return subtree

        routed: dict[int, list[str]] = {}
        training_values = X_train.to_numpy(copy=False)
        column_positions = {column: index for index, column in enumerate(X_train.columns)}
        for row_index, flow_id in enumerate(aligned_flow_ids):
            boundary_node, _, is_leaf = subtree.route_array(
                training_values[row_index],
                column_positions,
            )
            if not is_leaf:
                routed.setdefault(boundary_node, []).append(flow_id)
        for boundary_node, child_flows in sorted(routed.items()):
            child = self._fit_subtree(
                windows,
                partition_index + 1,
                child_flows,
                remaining_depth - partition_depth,
            )
            if child is not None:
                subtree.children[boundary_node] = child
        return subtree

    @staticmethod
    def _predict(root: PartitionSubtree, windows: list[DatasetSplit]) -> tuple[pd.Series, np.ndarray]:
        if windows[0].test_flow_ids is None:
            raise ValueError("SpliDT requires Flow ID values for routed partition inference")
        window_rows = []
        for split in windows:
            if split.test_flow_ids is None:
                raise ValueError("SpliDT requires Flow ID values for every inference window")
            positions = {
                flow_id: position for position, flow_id in enumerate(split.test_flow_ids)
            }
            values = split.X_test.to_numpy(copy=False)
            columns = {column: index for index, column in enumerate(split.X_test.columns)}
            window_rows.append((positions, values, columns))
        predictions = []
        valid_positions = []
        for position, flow_id in enumerate(windows[0].test_flow_ids):
            subtree = root
            prediction = None
            for positions, values, columns in window_rows:
                row_position = positions.get(flow_id)
                if row_position is None:
                    break
                boundary_node, prediction, is_leaf = subtree.route_array(
                    values[row_position],
                    columns,
                )
                if is_leaf or boundary_node not in subtree.children:
                    break
                subtree = subtree.children[boundary_node]
            if prediction is not None:
                valid_positions.append(position)
                predictions.append(prediction)
        return windows[0].y_test.iloc[valid_positions].reset_index(drop=True), np.asarray(predictions)

    def evaluate(self) -> tuple[PartitionSubtree, pd.Series, np.ndarray]:
        partition_count = len(self.config.splidt.partition_sizes)
        window_data = load_partition_dataset(self.config.dataset, partition_count)
        windows = [
            split_window_dataset(self.config.dataset, window_data, [partition_index])
            for partition_index in range(1, partition_count + 1)
        ]
        root_ids = windows[0].train_flow_ids
        if root_ids is None:
            raise ValueError("SpliDT training data has no Flow ID column")
        root = self._fit_subtree(windows, 0, root_ids.tolist(), self.config.splidt.max_depth)
        if root is None:
            raise ValueError("SpliDT did not train a root subtree")

        y_test, predictions = self._predict(root, windows)
        return root, y_test, predictions

    def run(self, output_dir: Path) -> ModelResult:
        partition_count = len(self.config.splidt.partition_sizes)
        root, y_test, predictions = self.evaluate()
        macro_f1 = calculate_macro_f1(y_test, predictions)
        selected_by_subtree = root.all_features()
        tree_entries = root.deployed_node_count()
        feature_entries = sum(len(features) for features in selected_by_subtree)
        resources = estimate_splidt_resources(
            TargetProfile.from_config(self.config.target),
            self.config.splidt.features_per_partition,
            feature_entries,
            tree_entries,
        )
        result = ModelResult.from_resources(
            model="SpliDT",
            dataset=self.config.dataset.name,
            target=self.config.target.name,
            seed=self.config.seed,
            macro_f1=macro_f1,
            max_depth=self.config.splidt.max_depth,
            num_features=len(set(feature for features in selected_by_subtree for feature in features)),
            test_samples=len(y_test),
            num_partitions=partition_count,
            resources=resources,
        )
        save_model_artifacts(output_dir, result, {"root": root, "features": selected_by_subtree})
        return result


def run_splidt(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return SpliDT(config).run(output_dir)
