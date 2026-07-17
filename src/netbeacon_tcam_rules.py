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


import os

from tree_to_table.rf import get_rf_feature_thres, get_rf_trees_table_entries
from tree_to_table.utils import *
from tree_to_table.xgb import get_xgb_feature_thres, get_xgb_trees_table_entries

current_path = os.getcwd()
dir_path = os.path.abspath(os.path.dirname(os.getcwd()))  #
from utils import get_distinct_flow_features, get_leaf_nodes, get_max_feat_threshold, get_subtrees


class CustomTreeModel:
    def __init__(self, tree, node_mask):
        """
        Initialize the custom tree model by filtering the nodes based on the node_mask.

        Args:
            tree: The original tree object from a DecisionTreeClassifier or DecisionTreeRegressor.
            node_mask: A boolean array where True indicates that the node should be included.
        """
        # Create a list of node indices based on the mask
        selected_nodes = [i for i, mask in enumerate(node_mask) if mask]

        # Create value-to-index mappings for children and features
        self.value_to_index = {node: idx for idx, node in enumerate(selected_nodes)}

        # Filter the tree arrays using the node_mask
        self.children_left = tree.children_left[selected_nodes]
        self.children_right = tree.children_right[selected_nodes]
        self.feature = tree.feature[selected_nodes]
        self.threshold = tree.threshold[selected_nodes]
        self.value = tree.value[selected_nodes]
        self.node_count = len(self.children_left)  # Number of nodes after filtering

    def get_value_from_shortened_array(self, original_value):
        """
        Access the corresponding index in the shortened array based on the original value.

        Args:
            original_value: The original node value you want to access.

        Returns:
            The index in the shortened arrays if the value exists, or None if it doesn't exist.
        """
        if original_value in self.value_to_index:
            idx = self.value_to_index[original_value]
            return {
                "children_left": self.children_left[idx],
                "children_right": self.children_right[idx],
                "feature": self.feature[idx],
                "threshold": self.threshold[idx],
                "value": self.value[idx],
            }
        else:
            return None

    def __repr__(self):
        return f"CustomTreeModel with {self.node_count} nodes"


def get_class_flow(orig_model, node_mask=None, exit_node_ids=None):
    model = CustomTreeModel(orig_model, node_mask)
    # print(model)
    pkt_flow_feat = []
    feat_table_sum = 0

    # get the distinct features of the model
    pkt_flow_feat = get_distinct_flow_features(model)

    # calculate len of flow features where it is non-negative
    pkt_flow_feat = [x for x in pkt_flow_feat if x >= 0]
    pkt_flow_feat_bit = [32] * len(pkt_flow_feat)

    max_feat_thres = {}
    for i in range(len(pkt_flow_feat)):
        max_feat_thres[pkt_flow_feat[i]] = 0

    feat_dict = get_rf_feature_thres(model, pkt_flow_feat)

    for key in feat_dict.keys():
        # adding positive values to the max_feat_thres
        if max_feat_thres[key] < len(feat_dict[key]):
            max_feat_thres[key] = len(feat_dict[key])

    # max_feat_threshold
    pkt_flow_mark_bit = get_max_feat_threshold(max_feat_thres)
    feat_key_bits = {}
    range_mark_bits = {}

    for i in range(len(pkt_flow_feat)):
        feat_key_bits[pkt_flow_feat[i]] = pkt_flow_feat_bit[i]
        range_mark_bits[pkt_flow_feat[i]] = pkt_flow_mark_bit[i]

    feat_table_data_all = {}
    feat_table_len = {}
    for i in range(len(pkt_flow_feat)):
        feat_table_data_all[pkt_flow_feat[i]] = []
        feat_table_len[pkt_flow_feat[i]] = {}
        feat_table_len[pkt_flow_feat[i]]["num_entries"] = 0
    tree_data_all = []
    tree_table_len = 0

    tree_num = 1
    feat_table_datas = get_feature_table_entries(feat_dict, feat_key_bits, range_mark_bits)
    sum_e = 0

    for key in feat_table_datas.keys():
        sum_e += len(feat_table_datas[key])
        feat_table_len[key]["num_entries"] += len(feat_table_datas[key])
        feat_table_len[key]["size"] = feat_key_bits[key]

    tree_data = get_rf_trees_table_entries(
        orig_model, pkt_flow_feat, feat_dict, range_mark_bits, tree_num
    )

    for i in range(len(pkt_flow_feat)):
        feat_table_data_all[pkt_flow_feat[i]].extend(feat_table_datas[pkt_flow_feat[i]])
    tree_data_all.extend(tree_data)
    tree_table_len += len(tree_data_all)

    # Integrate next-subtree routing for non-leaf partitions
    # print("Integrating next-subtree routing for non-leaf partitions...")

    num_exit_nodes = len(exit_node_ids) if exit_node_ids is not None else 0
    for i in range(len(tree_data_all)):
        entry = tree_data_all[i]

        if exit_node_ids is not None and i < num_exit_nodes:
            # Non-leaf partition → store both ID and class vector (for debugging or hybrid use)
            next_subtree_id = exit_node_ids[i]
            class_vector = entry[-1]
            entry[-1] = {"next_subtree_id": int(next_subtree_id), "class_probs": class_vector}
        else:
            # Leaf partition → keep the full class probability array
            entry[-1] = {"next_subtree_id": None, "class_probs": entry[-1]}

        tree_data_all[i] = entry

    return feat_table_data_all, tree_data_all


def main():
    model_file = "window_based_tree.pkl"
    with open(model_file, "rb") as file:
        loaded_model = pickle.load(file)

    print(loaded_model.tree_.node_count)
    num_tcam_entries = get_class_flow(loaded_model.tree_)
    print(num_tcam_entries)


if __name__ == "__main__":
    main()
