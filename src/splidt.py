from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .config import ExperimentConfig
from .data import DatasetSplit, load_partition_dataset, split_window_dataset
from .resources import TargetProfile, estimate_splidt_resources
from .tree import ModelResult, calculate_macro_f1


@dataclass
class PartitionSubtree:
    """One routed subtree in a SpliDT partition."""

    model: DecisionTreeClassifier
    features: list[str]
    partition_depth: int
    children: dict[int, "PartitionSubtree"] = field(default_factory=dict)

    def route(self, row: pd.Series) -> tuple[int, object, bool]:
        """Traverse this partition, returning boundary node, prediction, and leaf flag."""
        tree = self.model.tree_
        node = 0
        decisions = 0
        values = row[self.features]
        while tree.children_left[node] != tree.children_right[node]:
            if decisions == self.partition_depth:
                break
            feature = self.features[int(tree.feature[node])]
            node = int(
                tree.children_left[node]
                if values[feature] <= tree.threshold[node]
                else tree.children_right[node]
            )
            decisions += 1
        prediction = self.model.classes_[tree.value[node][0].argmax()]
        is_leaf = tree.children_left[node] == tree.children_right[node]
        return node, prediction, bool(is_leaf)

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
        seed_offset: int,
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
        for row_index, flow_id in enumerate(aligned_flow_ids):
            boundary_node, _, is_leaf = subtree.route(X_train.iloc[row_index])
            if not is_leaf:
                routed.setdefault(boundary_node, []).append(flow_id)
        for child_number, (boundary_node, child_flows) in enumerate(sorted(routed.items())):
            child = self._fit_subtree(
                windows,
                partition_index + 1,
                child_flows,
                remaining_depth - partition_depth,
                seed_offset * 1000 + child_number + 1,
            )
            if child is not None:
                subtree.children[boundary_node] = child
        return subtree

    @staticmethod
    def _predict(root: PartitionSubtree, windows: list[DatasetSplit]) -> tuple[pd.Series, np.ndarray]:
        if windows[0].test_flow_ids is None:
            raise ValueError("SpliDT requires Flow ID values for routed partition inference")
        window_rows = [
            {flow_id: row for flow_id, (_, row) in zip(split.test_flow_ids, split.X_test.iterrows())}
            for split in windows
        ]
        predictions = []
        valid_positions = []
        for position, flow_id in enumerate(windows[0].test_flow_ids):
            subtree = root
            prediction = None
            for partition_index, rows in enumerate(window_rows):
                row = rows.get(flow_id)
                if row is None:
                    break
                boundary_node, prediction, is_leaf = subtree.route(row)
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
        root = self._fit_subtree(windows, 0, root_ids.tolist(), self.config.splidt.max_depth, 1)
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
        result = ModelResult(
            model="SpliDT",
            dataset=self.config.dataset.name,
            target=self.config.target.name,
            macro_f1=macro_f1,
            max_depth=self.config.splidt.max_depth,
            num_features=len(set(feature for features in selected_by_subtree for feature in features)),
            num_partitions=partition_count,
            feature_state_bits=resources.feature_state_bits,
            metadata_bits=resources.metadata_bits,
            logical_entry_bits=resources.logical_entry_bits,
            aligned_entry_bits=resources.aligned_entry_bits,
            estimated_flow_capacity=resources.estimated_flow_capacity,
            feature_table_entries=resources.feature_table_entries,
            tree_table_entries=resources.tree_table_entries,
            total_table_entries=resources.total_table_entries,
            tcam_blocks=resources.tcam_blocks,
            tcam_stages=resources.tcam_stages,
            tcam_capacity_mb=resources.tcam_capacity_mb,
            tcam_memory_mb=resources.tcam_memory_mb,
            register_words_per_flow=resources.register_words_per_flow,
        )
        save_model_outputs(output_dir, result, {"root": root, "features": selected_by_subtree})
        return result


def save_model_outputs(output_dir: Path, result: ModelResult, model_payload) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.metrics_row()]).to_csv(output_dir / "metrics.csv", index=False)
    _remove_resource_csv(output_dir)
    (output_dir / "summary.json").unlink(missing_ok=True)
    with open(output_dir / "model.pkl", "wb") as handle:
        pickle.dump(model_payload, handle)


def _remove_resource_csv(output_dir: Path) -> None:
    resource_path = output_dir / "resource.csv"
    if resource_path.exists():
        resource_path.unlink()


def run_splidt(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return SpliDT(config).run(output_dir)
