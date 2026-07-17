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


import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import yaml

try:
    from box import Box
except ImportError:
    class Box(dict):
        def __init__(self, mapping):
            super().__init__()
            for key, value in mapping.items():
                self[key] = self._box(value)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = self._box(value)

        @classmethod
        def _box(cls, value):
            if isinstance(value, dict):
                return cls(value)
            if isinstance(value, list):
                return [cls._box(item) for item in value]
            return value


def parse_yml_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", type=str)
    parsed_args = parser.parse_args()
    config_file = os.path.abspath(parsed_args.config)
    with open(config_file, "r") as yml_file:
        config = yaml.safe_load(yml_file)
    boxed_configs = Box(config)
    # add home directory and project root variables
    boxed_configs.HOME_DIRECTORY = os.path.expanduser("~")
    boxed_configs.PROJECT_ROOT = os.getcwd()
    boxed_configs.CONFIG_FILE = config_file
    boxed_configs.FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(config_file))
    boxed_configs.ARTIFACT_ROOT = os.path.dirname(boxed_configs.FRAMEWORK_ROOT)

    if "dataset" in boxed_configs and "path" in boxed_configs.dataset:
        boxed_configs.dataset.path = resolve_dataset_path(
            boxed_configs.dataset.path,
            boxed_configs.PROJECT_ROOT,
            boxed_configs.FRAMEWORK_ROOT,
            boxed_configs.ARTIFACT_ROOT,
        )
    return boxed_configs


def resolve_dataset_path(dataset_path, project_root, framework_root, artifact_root):
    if os.path.isabs(dataset_path):
        return dataset_path

    candidates = [
        os.path.abspath(os.path.join(project_root, dataset_path)),
        os.path.abspath(os.path.join(framework_root, dataset_path)),
        os.path.abspath(os.path.join(artifact_root, dataset_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def get_parameter(param_string, datatype):
    # remove brackets if present
    if "[" in param_string:
        param_string = param_string[1:-1]

    if datatype == "int":
        return int(param_string)
    return float(param_string)


def generate_bruteforce_cuts(parsed_args, depth, max_cuts):
    c1_cuts = parsed_args.bruteforce.parameter_space.c1
    c2_cuts = parsed_args.bruteforce.parameter_space.c2
    c3_cuts = parsed_args.bruteforce.parameter_space.c3
    c4_cuts = parsed_args.bruteforce.parameter_space.c4
    c5_cuts = parsed_args.bruteforce.parameter_space.c5
    c6_cuts = parsed_args.bruteforce.parameter_space.c6
    all_combinations = list(product(c1_cuts, c2_cuts, c3_cuts, c4_cuts, c5_cuts, c6_cuts))
    cuts_limit = min(max_cuts, len(all_combinations))
    return random.sample(all_combinations, cuts_limit)


def cut_to_partitions(depth, this_cut):
    # 0. process the recommended cut
    LOGICAL_PSIZE = depth / len(this_cut)
    cut_0s_and_1s = [0] * (depth - 1)
    for this_part, logical_p_cut_placement in enumerate(this_cut):
        if logical_p_cut_placement > 0:
            cut_here = int((this_part + logical_p_cut_placement) * LOGICAL_PSIZE)
            cut_here = min(cut_here, depth - 2)
            cut_0s_and_1s[cut_here] = 1

    # 1. get number of partitions with this cut
    this_cut_num_partitions = cut_0s_and_1s.count(1) + 1

    # 2. get partition sizes for this cut
    this_cut_partition_sizes = []
    current_partition_size = 1
    for do_cut in cut_0s_and_1s:
        # if not to cut, increment current partition size
        if not do_cut:
            current_partition_size += 1
            continue
        # cut found, add current partition size to partition sizes
        this_cut_partition_sizes.append(current_partition_size)
        current_partition_size = 1

    # this can happen for the last partition
    if len(this_cut_partition_sizes) < this_cut_num_partitions:
        this_cut_partition_sizes.append(current_partition_size)

    return this_cut_num_partitions, this_cut_partition_sizes


def generate_all_cuts(depth, max_cuts=10, partition_limit=8, exploration_rate=0.2):
    # generate arbitrary cuts for a tree of this depth
    cuts = []
    num_partitions_per_cut = []
    partition_sizes_per_cut = []

    cuts_limit = min(max_cuts, 2**depth)
    explorations_limit = int(exploration_rate * max_cuts)
    explorations = 0

    while True:
        # generate a new random cut
        next_cut = list(random.choices([0, 1], k=depth))

        # don't repeat cuts
        if next_cut in cuts:
            continue

        # keep partitions in limit
        # get number of partitions with this cut
        next_cut_num_partitions = next_cut.count(1) + 1
        if next_cut_num_partitions > partition_limit:
            if explorations >= explorations_limit:
                continue
            explorations += 1

        cuts.append(next_cut)
        num_partitions_per_cut.append(next_cut_num_partitions)

        # get partition sizes for this cut
        next_cut_partition_sizes = []
        current_partition_size = 1

        for do_cut in next_cut:
            # if not to cut, increment current partition size
            if not do_cut:
                current_partition_size += 1
                continue
            # cut found, add current partition size to partition sizes
            next_cut_partition_sizes.append(current_partition_size)
            current_partition_size = 1

        # this can happen for the last partition
        if len(next_cut_partition_sizes) < next_cut_num_partitions:
            next_cut_partition_sizes.append(current_partition_size)

        partition_sizes_per_cut.append(next_cut_partition_sizes)

        # upper limit is all possible cuts
        if len(cuts) == cuts_limit:
            break

    # randomly sort these cuts
    combined = list(zip(cuts, num_partitions_per_cut, partition_sizes_per_cut))
    random.shuffle(combined)
    cuts, num_partitions_per_cut, partition_sizes_per_cut = zip(*combined)

    return cuts, num_partitions_per_cut, partition_sizes_per_cut, explorations


def get_train_test_split(
    dataset_file, num_partition, max_flows=None, train_split=0.8, do_packets=False
):
    print("Generating train/test split")

    # Read the dataset
    df = pd.read_csv(dataset_file)

    # drop 5-tuple features but retain the 5-tuple
    # not needed for packets; only for phases and partitions
    if not do_packets:
        df = drop_unwanted_features(df)
    # Group by 'Flow ID' and filter groups
    grouped_df = df.groupby("Flow ID")
    if not do_packets:
        grouped_df = grouped_df.filter(lambda x: len(x) == num_partition)

    # Shuffle and split flows
    unique_flows = np.array(grouped_df["Flow ID"].unique())  # Convert to NumPy array
    np.random.shuffle(unique_flows)

    # Limit the number of flows
    if max_flows and max_flows < len(unique_flows):
        unique_flows = unique_flows[:max_flows]

    train_flows = unique_flows[: int(train_split * len(unique_flows))]
    test_flows = unique_flows[int(train_split * len(unique_flows)) :]
    print("Training flows: ", len(train_flows))
    print("Testing flows: ", len(test_flows))

    # Convert back to Series
    train_flows = pd.Series(train_flows)
    test_flows = pd.Series(test_flows)

    return train_flows, test_flows


def read_and_process_dataset(
    dataset_file,
    num_partition,
    train_flows,
    test_flows,
    do_phases=False,
    do_packets=False,
    num_jobs=16,
):
    print("Reading and processing the dataset")

    # Read the dataset
    df = pd.read_csv(dataset_file)

    # drop 5-tuple features but retain the 5-tuple
    # not needed for packets; only for phases and partitions
    if not do_packets:
        df = drop_unwanted_features(df)

    # Group by 'Flow ID' and filter groups
    grouped_df = df.groupby("Flow ID")
    print("Number of partitions: ", num_partition)
    print("Number of flows before filtering: ", len(grouped_df))

    # IIsy, per-packet models (packets)
    if do_packets:
        # these two steps are only needed for packets, not phases
        grouped_df_cp = grouped_df.filter(lambda x: len(x) > 0)
        print("Number of packets after filtering: ", len(grouped_df_cp))
        # Assign 'Packet' column to each group
        grouped_df_cp["Packet"] = grouped_df_cp.groupby("Flow ID").cumcount() + 1

    # NetBeacon (phases)
    elif do_phases:
        # these two steps are only needed for phases, not partitions
        grouped_df_cp = grouped_df.filter(lambda x: len(x) > 0)
        print("Number of phases after filtering: ", len(grouped_df_cp))
        # Assign power of 2 'Phase' column to each group
        grouped_df_cp["Phase"] = 2 ** (grouped_df_cp.groupby("Flow ID").cumcount() + 1)

    # CAP (partitions)
    else:
        # these two steps are only needed for partitions, not phases
        grouped_df_cp = grouped_df.filter(lambda x: len(x) == num_partition)
        print("Number of flows after filtering: ", len(grouped_df_cp))
        # Assign 'Window' column to each group
        grouped_df_cp["Window"] = grouped_df_cp.groupby("Flow ID").cumcount() + 1

    # reset index to ungroup the dataframe
    ungrouped_df = grouped_df_cp.reset_index(drop=True)

    print("Separating train and test datasets")
    # Filter train and test
    # (parallel): Convert to Dask DataFrame for parallel processing
    if do_packets:
        try:
            import dask.dataframe as dd
        except ImportError as exc:
            raise ImportError(
                "dask is required only when generating packet-level datasets from raw CSVs. "
                "Install the project environment from environment.yml before running src/dataset.py "
                "with packet processing enabled."
            ) from exc

        dask_df = dd.from_pandas(ungrouped_df, npartitions=num_jobs)
        train_flows_set = set(train_flows.apply(lambda x: x[0]))
        test_flows_set = set(test_flows.apply(lambda x: x[0]))
        # Use optimized filtering (parallel)
        train_df = dask_df[
            dask_df["Flow ID"].map(train_flows_set.__contains__, meta=("Flow ID", "bool"))
        ].compute()
        test_df = dask_df[
            dask_df["Flow ID"].map(test_flows_set.__contains__, meta=("Flow ID", "bool"))
        ].compute()
    # phases and partitions are not that demanding
    else:
        # filter ungrouped_df to get train and test datasets
        train_df = ungrouped_df[ungrouped_df["Flow ID"].isin(train_flows)]
        test_df = ungrouped_df[ungrouped_df["Flow ID"].isin(test_flows)]

    print("Grouping train and test datasets")
    # group train_df and test_df by 'Flow ID'
    grouped_train_df = train_df.groupby("Flow ID")
    grouped_test_df = test_df.groupby("Flow ID")

    processed_df_dict = {
        "ungrouped_train_df": train_df,
        "ungrouped_test_df": test_df,
        "grouped_train_df": grouped_train_df,
        "grouped_test_df": grouped_test_df,
    }
    return processed_df_dict


def drop_unwanted_features(df):
    return remove_nonoffloadable_features(remove_id_features(df))


def remove_id_features(df):
    # notice that DST PORT is not removed
    # because it's one of the features
    features = ["Src IP", "Src Port", "Dst IP", "Protocol", "Timestamp"]
    return df.drop(columns=features, axis=1)


def remove_nonoffloadable_features(df):
    nonoffloadable_features = [
        "Fwd Packet Length Std",
        "Bwd Packet Length Std",
        "Flow IAT Std",
        "Fwd IAT Std",
        "Bwd IAT Std",
        "Packet Length Std",
        "Packet Length Variance",
        "Fwd Bytes/Bulk Avg",
        "Fwd Packet/Bulk Avg",
        "Fwd Bulk Rate Avg",
        "Bwd Bytes/Bulk Avg",
        "Bwd Packet/Bulk Avg",
        "Bwd Bulk Rate Avg",
        "Subflow Fwd Packets",
        "Subflow Fwd Bytes",
        "Subflow Bwd Packets",
        "Subflow Bwd Bytes",
        "FWD Init Win Bytes",
        "Bwd Init Win Bytes",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
    ]

    return df.drop(columns=nonoffloadable_features, axis=1)


def read_processed_dataset(dataset_file):
    grouped_df = pd.read_pickle(dataset_file)
    return grouped_df


def get_model_save_dir(parsed_args, model_type, selector):
    save_dir = os.path.join(
        parsed_args.PROJECT_ROOT,
        "models",
        parsed_args.dataset.name,
        model_type,
        selector,
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_pickle_artifact(parsed_args, model_type, selector, payload, metadata):
    save_dir = get_model_save_dir(parsed_args, model_type, selector)
    model_path = os.path.join(save_dir, "model.pkl")
    metadata_path = os.path.join(save_dir, "metadata.json")

    with open(model_path, "wb") as model_file:
        pickle.dump(payload, model_file)

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)

    print(f"Saved {model_type} {selector} model to {save_dir}")
    return save_dir


def show_ungrouped_dataset(df):
    print(df)


def show_grouped_dataset(grouped_df, groups=3):
    for count, (group_name, group_df) in enumerate(grouped_df):
        if count > groups:
            break
        print(group_name, group_df)


def show_dataset_problems(df):
    for group_name, group_df in df:
        if len(group_df) != 4:
            print(group_name, len(group_df))


def get_filtered_samples_and_labels(
    ungrouped_dataset,
    flows=None,
    min_window=1,
    max_window=10,
    windows=[1],
    do_phase=False,
    phases=[2],
    do_packets=False,
    packets=[0],
):
    # for netbeacon datasets
    if do_phase:
        filtered_df = filter_ungrouped_dataset(
            ungrouped_dataset, flows, do_phase=do_phase, phases=phases
        )
        samples = filtered_df.drop(columns=["Label"], axis=1)
        labels = filtered_df["Label"]
        return (samples, labels)

    # for iisy datasets
    if do_packets:
        filtered_df = filter_ungrouped_dataset(
            ungrouped_dataset, flows, do_packets=do_packets, packets=packets
        )
        samples = filtered_df.drop(columns=["Label"], axis=1)
        labels = filtered_df["Label"]
        return (samples, labels)

    # for partitions datasets
    if not do_packets and not do_phase:
        filtered_df = filter_ungrouped_dataset(
            ungrouped_dataset, flows, min_window, max_window, windows
        )
        samples = filtered_df.drop(columns=["Label"], axis=1)
        labels = filtered_df["Label"]
        return (samples, labels)


def filter_ungrouped_dataset(
    df,
    flows=None,
    min_window=1,
    max_window=10,
    windows=[1],
    do_phase=False,
    phases=[2],
    do_packets=False,
    packets=[0],
):
    # phases for netbeacon? check phases
    if do_phase:
        # if flows is None, return all flows with 'Phase' in phases
        # otherwise, return rows with matching 'Flow ID' and window value
        if flows is None:
            return df[df["Phase"].isin(phases)]
        return df[(df["Flow ID"].isin(flows)) & (df["Phase"].isin(phases))]

    # packets for iisy? check packets
    if do_packets:
        # if flows is None, return all packets
        # otherwise, return rows with matching 'Flow ID'
        if flows is None:
            return df[df["Packet"].isin(packets)]
        return df[(df["Flow ID"].isin(flows)) & (df["Packet"].isin(packets))]

    # partitions? check windows
    if not do_packets and not do_phase:
        if not all(min_window <= window <= max_window for window in windows):
            print(f"Error: Window values should be between {min_window} and {max_window}")
            return None
        # if flows is None, return all flows with 'Window' in windows
        # otherwise, return rows with matching 'Flow ID' and window value
        if flows is None:
            return df[df["Window"].isin(windows)]
        return df[(df["Flow ID"].isin(flows)) & (df["Window"].isin(windows))]


def get_partition_exit_nodes_and_features(
    partition_size, children_left, children_right, features, feature_names, node_id=0, level=0
):
    # Initialize partition_roots for this call
    partition_exit_nodes = defaultdict(list)
    partition_features = defaultdict(set)

    # Check if it is a leaf node
    if children_left[node_id] == children_right[node_id]:
        return partition_exit_nodes, partition_features

    # Entering next partition, this is an exit node
    this_partition = level // partition_size
    if level % partition_size == 0:
        partition_exit_nodes[this_partition].append(node_id)

    # check for the feature of this node
    partition_features[this_partition].add(feature_names[features[node_id]])

    # Recur for left and right children and get their partition roots
    left_partition_roots, left_partition_features = get_partition_exit_nodes_and_features(
        partition_size,
        children_left,
        children_right,
        features,
        feature_names,
        children_left[node_id],
        level + 1,
    )

    right_partition_roots, right_partition_features = get_partition_exit_nodes_and_features(
        partition_size,
        children_left,
        children_right,
        features,
        feature_names,
        children_right[node_id],
        level + 1,
    )

    # Combine the partition roots from left and right subtrees
    for partition, roots in left_partition_roots.items():
        partition_exit_nodes[partition].extend(roots)

    for partition, roots in right_partition_roots.items():
        partition_exit_nodes[partition].extend(roots)

    # combine the partition features from left and right subtrees
    for partition, features in left_partition_features.items():
        partition_features[partition].update(features)

    for partition, features in right_partition_features.items():
        partition_features[partition].update(features)

    return partition_exit_nodes, partition_features


def get_partition_exit_node(
    partition_entry_points,
    children_left,
    children_right,
    feature,
    threshold,
    value,
    classes,
    sample,
    node_id=0,
):
    # Check if it is a leaf node
    if children_left[node_id] == children_right[node_id]:  # Leaf node
        # print(f"Leaf node reached. (Node ID: {node_id})")
        return None, classes[value[node_id][0].argmax()], True

    # entering next partition, exit now
    if node_id in partition_entry_points:
        return node_id, classes[value[node_id][0].argmax()], False

    # Get the feature to split on and the threshold
    feature_id = feature[node_id]
    threshold_value = threshold[node_id]

    if sample[feature_id] <= threshold_value:
        return get_partition_exit_node(
            partition_entry_points,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            classes,
            sample,
            children_left[node_id],
        )
    else:
        return get_partition_exit_node(
            partition_entry_points,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            classes,
            sample,
            children_right[node_id],
        )


def get_leaf_nodes(model):
    leaf_nodes = []
    for i in range(model.node_count):
        if model.children_left[i] == model.children_right[i]:
            leaf_nodes.append(i)

    return leaf_nodes


def get_subtrees(dict_key, model, leaf_nodes):
    subtrees = {}

    # Recursive function to get the subtree for a given node
    def traverse(node, level):
        if node not in subtrees:  # To avoid recomputing for already visited nodes
            # Initialize an empty subtree for this node
            subtrees[node] = [node]
            if level in dict_key:  # Check if there's a next level in the dict
                # add left child to the subtree
                left_child = model.children_left[node]
                # add right child to the subtree
                right_child = model.children_right[node]
                if left_child is not None and (
                    left_child in dict_key[level + 1] or left_child in leaf_nodes
                ):
                    subtrees[node].append(left_child)
                    traverse(left_child, level + 1)
                # add right child to the subtree
                if right_child is not None and (
                    right_child in dict_key[level + 1] or right_child in leaf_nodes
                ):
                    subtrees[node].append(right_child)
                    traverse(right_child, level + 1)

    # Start traversing from the root node (node 0, level 0)
    traverse(0, 0)

    return subtrees


def get_distinct_flow_features(model):
    # get distinct features of the model
    features = []
    for feature in model.feature:
        if feature not in features:
            features.append(feature)
    return features


def get_max_feat_threshold(max_feat_thres):
    max_feat_threshold = []
    for key in max_feat_thres.keys():
        max_feat_threshold.append(max_feat_thres[key])
    return max_feat_threshold
