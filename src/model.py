# *****************************************************************************
#
# Copyright 2026
#   Murayyiam Parvez (Purdue University),
#   Annus Zulfiqar (University of Michigan),
#   Roman Beltiukov (University of California, Santa Barbara),
#   Shir Landau Feibish (The Open University of Israel),
#   Walter Willinger (NIKSUN Inc.),
#   Arpit Gupta (University of California, Santa Barbara),
#   Muhammad Shahbaz (University of Michigan)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *****************************************************************************


import copy
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.tree._tree import Tree

import netbeacon_tcam_rules as nb_tcam_rules
import utils

try:
    import graphviz
except ImportError:
    graphviz = None


@dataclass
class ModelConfig:
    max_depth: int = 0
    features_per_partition: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    num_partitions: int = 0
    f1_score: float = 0.0
    num_flows: int = 0
    feasible: bool = False
    total_features: int = 0
    feature_entries: int = 0
    table_entries: int = 0

    # Define greater-than comparison based on f1_score
    def __gt__(self, other):
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.f1_score > other.f1_score

    def __lt__(self, other):
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.f1_score < other.f1_score

    def __ge__(self, other):
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.f1_score >= other.f1_score

    def __le__(self, other):
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.f1_score <= other.f1_score

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.f1_score == other.f1_score


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, pred_class=None, samples=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.samples = samples
        self.pred_class = pred_class
        self.left_exit_node = 1
        self.right_exit_node = 2


class PartitionBasedDecisionTree:
    def __init__(
        self,
        max_depth,
        max_leaf_nodes,
        partition_size,
        all_features,
        feature_limit,
        all_labels,
        node_id_base: int = 0,
    ):
        self.node_id_base = node_id_base
        self.local_to_global_node_id = {}
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.partition_size = partition_size
        self.all_features = all_features
        self.all_labels = all_labels
        self.feature_limit = feature_limit
        self.top_k_features = None

        self.partition_model = DecisionTreeClassifier(
            random_state=42,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            criterion="entropy",
            class_weight="balanced",
        )
        self.next_partition_models = {}
        self.partition_tree_struct = None
        self.small_tree = None

        # the parent tree structure
        self._tree = None
        self._tree_depth = None
        self._n_nodes = None
        self._children_left = None
        self._children_right = None
        self._features = None
        self._classes = None
        self._thresholds = None
        self._values = None
        self._exit_nodes = None

        # count the total features in this and all partitions below
        self.partition_features = None

        # TCAM entries
        self.feature_table_entries = None
        self.tree_table_entries = None

    def show_tree(self):
        """Print the decision tree structure."""
        tree_rules = export_text(self.partition_model, feature_names=self.all_features)
        print(f"{tree_rules}\nTree Depth: {self._tree_depth}\n")

    def _get_parent_tree_structure(self):
        """Get the structure of the parent tree."""
        self._tree = self.partition_model.tree_
        self._tree_depth = self.partition_model.get_depth()
        self._n_nodes = self._tree.n_node_samples
        self._samples_at_each_node = {
            node_id: samples for node_id, samples in enumerate(self._n_nodes)
        }
        self._children_left = self._tree.children_left
        self._children_right = self._tree.children_right
        self._features = self._tree.feature
        self._classes = self.partition_model.classes_
        self._thresholds = self._tree.threshold
        self._values = self._tree.value

    def _get_partition_exit_nodes_and_features(self):
        if self.partition_size == 0:
            print("Error: Got partition size 0. No more partitioning needed on this branch.")
            return None
        return utils.get_partition_exit_nodes_and_features(
            partition_size=self.partition_size,
            children_left=self._children_left,
            children_right=self._children_right,
            features=self._features,
            feature_names=self.top_k_features if self.feature_limit > 0 else self.all_features,
        )

    def show_partition_exit_nodes_and_features(self, verbose=False):
        if self._exit_nodes is None:
            self._exit_nodes, self.partition_features = (
                self._get_partition_exit_nodes_and_features()
            )

        if not verbose:
            return

        if self._exit_nodes is None:
            print("Error: Partition exit nodes not found.")
            return

        for partition, roots in self._exit_nodes.items():
            print(f"Partition {partition} ({len(roots)}); exit nodes: {roots}")

        for partition, features in self.partition_features.items():
            print(f"Partition {partition} features: {features}")

        pass

    def fit_parent_tree(self, X_train, y_train):
        self.partition_model.fit(X_train, y_train)

        # feature limiting is enabled
        if self.feature_limit > 0:
            # print(f"Limiting features to {self.feature_limit}")
            # Feature importances
            feature_importances = self.partition_model.feature_importances_
            # Get feature names
            feature_names = X_train.columns
            # Get indices of top-k features
            top_k_indices = np.argsort(feature_importances)[-self.feature_limit :]
            # Get the actual top-k feature names
            self.top_k_features = [feature_names[i] for i in top_k_indices]
            # get only these features from X_train
            X_train_top_k_features = X_train[self.top_k_features]
            # fit the model again
            self.partition_model.fit(X_train_top_k_features, y_train)

        # set parent tree structure right after training it
        self._get_parent_tree_structure()
        self._exit_nodes, self.partition_features = self._get_partition_exit_nodes_and_features()

    def _generate_partition_tree(self, node_id=0):
        # It's a leaf
        if self._children_left[node_id] == self._children_right[node_id]:
            return Node(pred_class=self._classes[self._values[node_id][0].argmax()])

        # entered next partition, we exit now and no new node is created
        if node_id in self._exit_nodes[1]:
            return None

        # Create a new node with the feature and threshold
        node = Node(
            feature=self._features[node_id],
            threshold=self._thresholds[node_id],
            left=self._children_left[node_id],
            right=self._children_right[node_id],
            samples=self._samples_at_each_node[node_id],
            pred_class=self._classes[self._values[node_id][0].argmax()],
        )

        # Recursively build the left and right subtrees
        node.left = self._generate_partition_tree(self._children_left[node_id])
        node.right = self._generate_partition_tree(self._children_right[node_id])

        return node

    def generate_partition_tree(self):
        self.partition_tree_struct = self._generate_partition_tree()

    def save_tree_png(self, save_path):
        if graphviz is None:
            raise ImportError("graphviz is required to render decision tree images.")

        # Export the decision tree to DOT format
        dot_data = export_graphviz(
            self.partition_model,
            node_ids=True,
            out_file=None,
            feature_names=self.top_k_features,
            class_names=self._classes,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        # Create a graph from the DOT data and save it as a PNG
        graph = graphviz.Source(dot_data)
        # Set the DPI (e.g., 300 DPI for higher resolution)
        # graph.attr(dpi='300')
        graph.render("decision_tree", cleanup=True)

    def save_partition_tree(self, save_path):
        # save the parent tree (always after parent tree structure is set)
        self.generate_partition_tree()

        # commenting this out for now.. problematic with multiple threads
        # save the tree in a pickle file
        with open(save_path, "wb") as partition_tree_file:
            pickle.dump(self.partition_tree_struct, partition_tree_file)

    def prune_tree(self):
        """
        Prune the decision tree by removing all nodes after the exit_nodes.
        Rebuild a new smaller tree structure that includes only the unpruned nodes.
        """
        # Extract the necessary tree structure components
        children_left = self.partition_model.tree_.children_left
        children_right = self.partition_model.tree_.children_right
        feature = self.partition_model.tree_.feature
        threshold = self.partition_model.tree_.threshold
        value = self.partition_model.tree_.value

        # Create a mask to keep track of nodes to prune
        to_keep = np.ones(self.partition_model.tree_.node_count, dtype=bool)

        # Mark all descendants of the exit nodes as not to keep
        # print("Exit node: ", self._exit_nodes[1])
        for exit_node in self._exit_nodes[1]:

            def mark_subtree_for_pruning(node):
                if node == -1:
                    return
                to_keep[node] = False
                mark_subtree_for_pruning(children_left[node])
                mark_subtree_for_pruning(children_right[node])

            # Mark the current node and its descendants
            mark_subtree_for_pruning(exit_node)

        # Find how many nodes remain after pruning
        kept_node_count = int(np.sum(to_keep))

        self.small_tree = copy.deepcopy(self.partition_model)
        # in self.partition_model, turn the exit nodes to -1
        for exit_node in self._exit_nodes[1]:
            self.small_tree.tree_.children_left[exit_node] = -1
            self.small_tree.tree_.children_right[exit_node] = -1

        return to_keep

    def generate_TCAM_entries(self):
        # print("Generating TCAM entries for partition tree...")
        # preprocess self.partition_model
        partition_tree_nodes_bool = self.prune_tree()
        # get TCAM entries needed for this partition tree
        local_exit_nodes = (
            self._exit_nodes[1] if (self._exit_nodes and 1 in self._exit_nodes) else []
        )
        exit_node_ids = [self._global_id(n) for n in local_exit_nodes]  # <--- GLOBAL IDs
        # get TCAM entries needed for this partition tree
        TCAM_rules = nb_tcam_rules.get_class_flow(
            self.small_tree.tree_, partition_tree_nodes_bool, exit_node_ids=exit_node_ids
        )
        self.ft_entries, self.tt_entries = TCAM_rules
        self.feature_table_entries = len(self.ft_entries)
        self.tree_table_entries = len(self.tt_entries)

    def _global_id(self, local_id: int) -> int:
        return self.node_id_base + int(local_id)

    def predict_traditional(self, X_test):
        if self.feature_limit > 0:
            X_test_top_k_features = X_test[self.top_k_features]
            return self.partition_model.predict(X_test_top_k_features)

        return self.partition_model.predict(X_test)

    def get_data_split_at_partition_exit_nodes(self, samples):
        if self._exit_nodes is None:
            return {}

        data_split_by_partition_node = {}
        for partition_node in self._exit_nodes[1]:
            data_split_by_partition_node[partition_node] = []

        # for each sample in samples, return which node it belongs to
        # nodes are exit points for this partition
        for _, this_sample in samples.iterrows():
            # drop id and reorder the columns
            this_sample_without_id = this_sample.drop(columns=["Flow ID", "Window"], axis=1)
            available_features = (
                self.all_features if self.feature_limit == 0 else self.top_k_features
            )
            this_sample_without_id = this_sample_without_id[available_features]

            # we only look at the first partition exit
            partition_node, _, _ = utils.get_partition_exit_node(
                partition_entry_points=self._exit_nodes[1],
                children_left=self._children_left,
                children_right=self._children_right,
                feature=self._features,
                threshold=self._thresholds,
                value=self._values,
                classes=self._classes,
                sample=this_sample_without_id,
            )

            # get the 'Flow ID' and add it to the list of flows for this partition node
            if partition_node is not None:
                # None will occur when leaf node is reached.. no need for more partitioning
                data_split_by_partition_node[partition_node].append(this_sample["Flow ID"])

        return data_split_by_partition_node


class WindowBasedDecisionTree:
    def __init__(self, partition_model: PartitionBasedDecisionTree):
        self.partition_model = partition_model

    def predict_window_based(self, sample):
        window = 1
        finished = False
        # pick the first partition model
        partition_model = self.partition_model
        feature_limit = partition_model.feature_limit
        top_k_features = partition_model.top_k_features

        while not finished:
            # pick only top-k features if configuration said so.
            if feature_limit > 0:
                sample_window = sample[window][top_k_features].values
                # print(f"Window {window} features: {sample}")
            else:
                sample_window = sample[window].values

            # perform inference on this partition until an exit or leaf node is reached
            exit_node, pred_class, finished = utils.get_partition_exit_node(
                partition_entry_points=partition_model._exit_nodes[1]
                if partition_model._exit_nodes
                else [],
                children_left=partition_model._children_left,
                children_right=partition_model._children_right,
                feature=partition_model._features,
                threshold=partition_model._thresholds,
                value=partition_model._values,
                classes=partition_model._classes,
                sample=sample_window,
            )

            # if leaf node is reached, return the class
            if finished:
                return pred_class

            # update the parent_id with current exit node and move to next window
            if exit_node not in partition_model.next_partition_models:
                return pred_class

            # move to the next partition model
            partition_model = partition_model.next_partition_models[exit_node]
            if partition_model is None:
                return pred_class

            # if not none, update the feature limit and top-k features
            feature_limit = partition_model.feature_limit
            top_k_features = partition_model.top_k_features
            window += 1

            pass

    def evaluate_test_set(self, grouped_testing_dataset, train_features):
        ground_truth, predictions = [], []
        for flow, window_group in grouped_testing_dataset:
            window_group = window_group[[col for col in window_group.columns if col != "Flow ID"]]
            flow_windows = {}

            # prepare windows of this flow for inference
            label = window_group["Label"].values[0]
            for _, row in window_group.iterrows():
                # get the window id and rearrange the columns..
                window_id = row["Window"]
                flow_windows[window_id] = row[train_features]

            # perform inference on this flow
            flow_class = self.predict_window_based(sample=flow_windows)
            ground_truth.append(label)
            predictions.append(flow_class)

        return ground_truth, predictions
