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

from .utils import *


# Get all the thresholds that appear in the tree
def get_rf_feature_thres(model, pkt_flow_feat):
    feat_dict = {}
    for key in pkt_flow_feat:
        feat_dict[key] = []
        for node in range(model.node_count):
            # Check if the node splits on the desired feature
            if model.feature[node] == key:
                feat_dict[key].append(int(model.threshold[node]) + 1)

    return feat_dict


# Get the model table table entries
def get_rf_trees_table_entries(model, keys, feat_dict, key_encode_bits, tree_num, pkts=None):
    tree_data = []
    tree_leaves = []  # Each row is a leaf node, recording that smallest threshold index in left subtree and smallest threshold index (negative) in right subtree on the path of that leaf node
    trees = []
    leaf_index = []
    leaf_info = []
    trees.append(len(tree_leaves))
    nodes = {}
    tree = model

    # Recursive function to build node structure and extract paths
    def traverse(node, path):
        nodes[node] = {}
        nodes[node]["path"] = [1000, 0] * len(keys)  # Initialize paths

        # If it's a leaf node
        if tree.children_left[node] == tree.children_right[node]:
            leaf_info.append(tree.value[node])
            leaf_index.append(node)
            nodes[node]["path"] = path.copy()
            tree_leaves.append(nodes[node]["path"])
            return

        # Non-leaf node: extract feature and threshold information
        feat_name = tree.feature[node]
        thre = tree.threshold[node]

        nodes[node]["info"] = [feat_name, thre]
        thre = int(float(thre)) + 1

        # Traverse left
        left_path = path.copy()
        left_path[keys.index(feat_name) * 2] = min(
            left_path[keys.index(feat_name) * 2], feat_dict[feat_name].index(int(thre)) + 1
        )
        traverse(tree.children_left[node], left_path)

        # Traverse right
        right_path = path.copy()
        right_path[keys.index(feat_name) * 2 + 1] = min(
            right_path[keys.index(feat_name) * 2 + 1], -feat_dict[feat_name].index(int(thre)) - 1
        )
        traverse(tree.children_right[node], right_path)

    # Start traversal from root node
    traverse(0, [1000, 0] * len(keys))

    # After traversal, 'nodes' will be filled with the tree structure, paths, and other information
    trees.append(len(tree_leaves))

    loop_val = []
    for i in range(len(trees))[:-1]:
        loop_val.append(range(trees[i], trees[i + 1]))
    for tup in product(*loop_val):
        flag = 0
        for f in range(len(keys)):  # Check for conflicting feature values
            a = 1000
            b = 1000
            for i in tup:
                a = min(tree_leaves[i][f * 2], a)
                b = min(tree_leaves[i][f * 2 + 1], b)
            if a + b <= 0:
                flag = 1
                break
        # Semantic conflict check can be added here
        if flag == 0:
            # print("-- ",tup,sigmoid(leafs[i]+leafs[j]))
            if pkts is None:
                tree_data.append([])  #
            else:
                tree_data.append([pkts])
            for f in range(len(keys)):
                a = 1000
                b = 1000
                for i in tup:
                    a = min(tree_leaves[i][f * 2], a)
                    b = min(tree_leaves[i][f * 2 + 1], b)
                key = keys[f]
                te = get_model_table_range_mark(key_encode_bits[key], a, b, len(feat_dict[key]))
                tree_data[-1].extend(
                    [
                        int(get_value_mask(te, key_encode_bits[key])[0], 2),
                        int(get_value_mask(te, key_encode_bits[key])[1], 2),
                    ]
                )  # The value and mask of each feature
            leaf_sum = leaf_info[tup[0]].copy()
            for i in tup[1:]:
                for j in range(len(leaf_sum)):
                    leaf_sum[j] += leaf_info[i][j]
            tree_data[-1].append(np.array(leaf_sum) / len(tup))  # classification probabilities list
            # print(tup,np.max(leaf_sum)/len(tup))
    return tree_data
