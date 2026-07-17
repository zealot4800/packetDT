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


import json
import os
import pickle
import random
import shutil
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from subprocess import PIPE, Popen

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import model
import scenario
import switch_resources as sw_resources
import utils


class MinimalDbManager:
    def create_logging_database(self, parsed_args):
        print("Minimal build: skipping PostgreSQL logging database setup.")

    def commit_to_logging_database(self, parsed_args, model_type, iteration, model_config):
        return None


db_manager = MinimalDbManager()


def save_splidt_selected_models(parsed_args, results_path):
    samples_path = os.path.join(results_path, "output_samples.csv")
    if not os.path.exists(samples_path):
        print("No HyperMapper output_samples.csv found; skipping SpliDT model save.")
        return

    samples_df = pd.read_csv(samples_path)
    if samples_df.empty:
        print("HyperMapper output_samples.csv is empty; skipping SpliDT model save.")
        return

    samples_df["actual_f1_score"] = -1.0 * samples_df["f1_score"]
    samples_df["actual_num_flows"] = -1.0 * samples_df["num_flows"]
    samples_df = samples_df[(samples_df["actual_f1_score"] > 0) & (samples_df["actual_num_flows"] > 0)]
    if samples_df.empty:
        print("No feasible SpliDT samples found; skipping SpliDT model save.")
        return

    selected_rows = {
        "best_by_f1": samples_df.loc[samples_df["actual_f1_score"].idxmax()],
        "best_by_flows": samples_df.loc[samples_df["actual_num_flows"].idxmax()],
    }

    for selector, row in selected_rows.items():
        model_config = model.ModelConfig(
            max_depth=int(row["depth"]),
            features_per_partition=int(row["features_per_partition"]),
            c1=float(row["c1"]),
            c2=float(row["c2"]),
            c3=float(row["c3"]),
            c4=float(row["c4"]),
            c5=float(row["c5"]),
            c6=float(row["c6"]),
        )
        model_metrics = single_run_train_arbitrary_partitions(
            parsed_args,
            results_path,
            model_config,
            save_tree=True,
            save_tree_selector=selector,
        )

        save_dir = utils.get_model_save_dir(parsed_args, "splidt", selector)
        metadata_path = os.path.join(save_dir, "metadata.json")
        metadata = {
            "dataset": parsed_args.dataset.name,
            "model_type": "splidt",
            "selector": selector,
            "config": parsed_args.CONFIG_FILE,
            "max_depth": model_metrics.max_depth,
            "features_per_partition": model_metrics.features_per_partition,
            "cuts": {
                "c1": model_metrics.c1,
                "c2": model_metrics.c2,
                "c3": model_metrics.c3,
                "c4": model_metrics.c4,
                "c5": model_metrics.c5,
                "c6": model_metrics.c6,
            },
            "num_partitions": model_metrics.num_partitions,
            "f1_score": model_metrics.f1_score,
            "num_flows": model_metrics.num_flows,
            "feasible": model_metrics.feasible,
            "total_features": model_metrics.total_features,
            "feature_entries": model_metrics.feature_entries,
            "table_entries": model_metrics.table_entries,
        }
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, default=str)
        print(f"Saved splidt {selector} model metadata to {save_dir}")


def train_partitions_recursively(
    ungrouped_training_dataset,
    tree_depth,
    leaf_nodes,
    num_partitions,
    partition_sizes,
    save_partition_trees=False,
    tree_save_path=None,
    feature_limit=0,
    flows=None,
    window=1,
    node_id_base=0,
):
    # # Train partition = P
    # print(f"\nTraining partition = {window}\n")
    # print(f"Partition size = {partition_sizes[window-1]}")

    # get the training and testing dataset for first window for all flows
    X_train, y_train = utils.get_filtered_samples_and_labels(
        ungrouped_training_dataset, flows=flows, min_window=1, max_window=50, windows=[window]
    )
    # drop the 'Flow ID' and 'Window' columns
    X_train_without_id = X_train.drop(columns=["Flow ID", "Window"], axis=1)

    # get all unique features and labels
    all_features = X_train_without_id.columns
    all_labels = y_train.unique()

    # Train the first partition
    partition_dt = model.PartitionBasedDecisionTree(
        max_depth=tree_depth,
        max_leaf_nodes=leaf_nodes,
        partition_size=partition_sizes[window - 1],
        all_features=all_features,
        feature_limit=feature_limit,
        all_labels=all_labels,
        node_id_base=node_id_base,
    )

    # Replace inf values with NaN
    X_train_without_id = X_train_without_id.replace([np.inf, -np.inf], np.nan)
    # Identify rows with NaN values
    dropped_indices = X_train_without_id[X_train_without_id.isna().any(axis=1)].index
    # Drop rows with NaN and reset index
    X_train_without_id = X_train_without_id.drop(dropped_indices).reset_index(drop=True)
    X_train = X_train.drop(dropped_indices).reset_index(drop=True)
    y_train = y_train.drop(dropped_indices).reset_index(drop=True)

    if X_train_without_id.shape[0] == 0:
        print("Error: Empty training dataset. Exiting.")
        return None

    partition_dt.fit_parent_tree(X_train_without_id, y_train)
    partition_dt.show_partition_exit_nodes_and_features()

    # generate TCAM rules for this partitions
    partition_dt.generate_TCAM_entries()

    # save pkl model
    if save_partition_trees and tree_save_path is not None:
        subtree_id = node_id_base
        pkl_path = os.path.join(tree_save_path, f"subtree_{subtree_id}_model.pkl")
        json_path = os.path.join(tree_save_path, f"subtree_{subtree_id}_feature_map.json")

        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(
                {
                    "subtree_id": subtree_id,
                    "partition_model": partition_dt.partition_model,
                    "small_tree": partition_dt.small_tree,
                    "top_k_features": partition_dt.top_k_features,
                    "partition_features": dict(partition_dt.partition_features),
                    "feature_table_entries": partition_dt.ft_entries,
                    "tree_table_entries": partition_dt.tt_entries,
                },
                pkl_file,
            )

        feature_map = {}

        for fid, fname in enumerate(partition_dt.partition_features[0]):
            feature_map[f"feature_{fid}"] = fname

        with open(json_path, "w") as json_file:
            json.dump(
                {
                    "subtree_id": int(subtree_id),
                    "feature_map": {k: str(v) for k, v in feature_map.items()},
                },
                json_file,
                indent=4,
            )
        # print(f"[JSON SAVED] {json_path}")

    if partition_dt._exit_nodes is None:
        print("Error: Partition exit nodes not found. Exiting.")
        return partition_dt

    # do inference on the this window to determine split of the data
    train_data_split = partition_dt.get_data_split_at_partition_exit_nodes(X_train)

    # Train partition = P+1

    # train the second partition with the split data
    next_tree_depth = tree_depth - partition_sizes[window - 1]
    if next_tree_depth == 0:
        return partition_dt

    true_partition_flows = {
        node_id: samples for node_id, samples in enumerate(partition_dt._n_nodes)
    }

    for partition_exit_node in train_data_split.keys():
        # get training dataset for this next partition and train it
        train_partition_flows = train_data_split[partition_exit_node]
        if len(train_partition_flows) != int(true_partition_flows[partition_exit_node]):
            # issue warning but don't exit
            print(
                f"Warning: True partition flows: {true_partition_flows[partition_exit_node]}, ",
                end="",
            )
            print(f"Training partition flows: {len(train_partition_flows)}")

        this_partition_dt = train_partitions_recursively(
            ungrouped_training_dataset=ungrouped_training_dataset,
            tree_depth=next_tree_depth,
            leaf_nodes=leaf_nodes,
            num_partitions=num_partitions,
            partition_sizes=partition_sizes,
            save_partition_trees=save_partition_trees,
            tree_save_path=tree_save_path,
            feature_limit=feature_limit,
            flows=train_partition_flows,
            window=window + 1,
            node_id_base=partition_exit_node,
        )

        if this_partition_dt is None:
            print("Info: Skipping this partition model because of empty training dataset.")
            continue

        # add this partition classifier at the exit node
        partition_dt.next_partition_models[partition_exit_node] = this_partition_dt

        # add children TCAM rules to parent
        partition_dt.feature_table_entries += this_partition_dt.feature_table_entries
        partition_dt.tree_table_entries += this_partition_dt.tree_table_entries

        # add features for this partition to the parent
        partition_dt.partition_features[0].update(this_partition_dt.partition_features[0])

    return partition_dt


def single_run_train_arbitrary_partitions(
    parsed_args,
    results_path,
    model_config,
    save_tree=False,
    save_tree_selector=None,
):
    # extract model params
    max_depth = model_config.max_depth
    features_per_partition = model_config.features_per_partition
    c1 = model_config.c1
    c2 = model_config.c2
    c3 = model_config.c3
    c4 = model_config.c4
    c5 = model_config.c5
    c6 = model_config.c6

    # pick corresponding max_leaf_nodes
    leaf_nodes = min(2**max_depth, 4096)

    num_partitions, partition_sizes = utils.cut_to_partitions(
        depth=max_depth, this_cut=[c1, c2, c3, c4, c5, c6]
    )

    # early rejections (one partition is Leo)
    if num_partitions <= 1:
        return model.ModelConfig(
            max_depth=max_depth,
            features_per_partition=features_per_partition,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            c5=c5,
            c6=c6,
            num_partitions=num_partitions,
        )

    dataset_path = os.path.join(
        parsed_args.dataset.path,
        parsed_args.dataset.name,
        parsed_args.dataset.destination,
        f"dataset_df_p{num_partitions}.pkl",
    )

    read_processed_df = utils.read_processed_dataset(dataset_path)
    ungrouped_training_dataset = read_processed_df["ungrouped_train_df"]
    grouped_testing_dataset = read_processed_df["grouped_test_df"]

    tree_save_dir = None
    if save_tree:
        if save_tree_selector is not None:
            tree_save_dir = utils.get_model_save_dir(parsed_args, "splidt", save_tree_selector)
        else:
            base_dir = os.path.join(
                getattr(parsed_args, "HOME_DIRECTORY", os.path.expanduser("~")),
                getattr(parsed_args, "PROJECT_ROOT", "/models"),
            )
            tree_save_dir = os.path.join(
                base_dir,
                "models",
                f"{parsed_args.dataset.name}",
                f"d{max_depth}_np{num_partitions}_fl{features_per_partition}",
            )

        # Clean up any existing directory
        if os.path.exists(tree_save_dir):
            print(f"Existing directory found — removing: {tree_save_dir}")
            shutil.rmtree(tree_save_dir)

        # Create fresh directory
        os.makedirs(tree_save_dir, exist_ok=True)
        print(f"Trees will be saved to: {os.path.abspath(tree_save_dir)}")

    partition_classifier = train_partitions_recursively(
        ungrouped_training_dataset=ungrouped_training_dataset,
        tree_depth=max_depth,
        leaf_nodes=leaf_nodes,
        num_partitions=num_partitions,
        partition_sizes=partition_sizes,
        save_partition_trees=save_tree,
        tree_save_path=tree_save_dir,
        feature_limit=features_per_partition,
    )

    model_features = partition_classifier.partition_features[0]
    total_features = len(model_features)
    feature_entries = partition_classifier.feature_table_entries
    table_entries = partition_classifier.tree_table_entries

    # resources logic: select the model for resource estimation
    switch_model = parsed_args.switch_model.tofino1
    if parsed_args.switch_model.use_tofino2:
        switch_model = parsed_args.switch_model.tofino2

    FEATURE_WIDTH = switch_model.feature_width
    NODE_ID_WIDTH = switch_model.node_id_width
    NUM_MAU_STAGES = switch_model.num_ma_units
    # this is must for node_id (16-bit) and pkt_count (16-bit) ( always needed )
    BOOKKEEPING = switch_model.bookkeeping_stages
    # for IAT features only; include this if such a feature is being used
    IAT_BOOKKEEPING = switch_model.iat_bookkeeping_stages
    if any("IAT" in feat for feat in model_features):
        BOOKKEEPING += IAT_BOOKKEEPING

    # calculate number of stages
    feature_tcam_entries_per_key = sw_resources.calculate_allocated_tcam_entries(
        num_tcam_entries=feature_entries, width_per_key=FEATURE_WIDTH
    )

    table_tcam_entries_per_key = sw_resources.calculate_allocated_tcam_entries(
        num_tcam_entries=table_entries, width_per_key=FEATURE_WIDTH * features_per_partition
    )

    feature_table_stages = sw_resources.tcam_entries_to_stages(
        num_tcam_entries=feature_tcam_entries_per_key,
        width_per_key=NODE_ID_WIDTH + (FEATURE_WIDTH * features_per_partition),
    )

    tree_table_stage = sw_resources.tcam_entries_to_stages(
        num_tcam_entries=table_tcam_entries_per_key,
        width_per_key=NODE_ID_WIDTH + (FEATURE_WIDTH * features_per_partition),
    )

    remaining_stages = NUM_MAU_STAGES - (feature_table_stages + tree_table_stage)
    num_flows = sw_resources.stages_to_registers(
        num_features=BOOKKEEPING + features_per_partition, remaining_stages=remaining_stages
    )

    # resubmission traffic estimation
    resubmission_traffic = sw_resources.get_resubmission_traffic_Gbps(
        switch_model, num_flows, num_partitions
    )
    is_resubmission_feasible = resubmission_traffic <= switch_model.resubmission_bw_Gbps

    # drop the 'Flow ID' and 'Window' columns
    train_features = ungrouped_training_dataset.drop(columns=["Flow ID", "Window"], axis=1).columns
    # train_labels = ungrouped_training_dataset['Label'].unique()

    # generate the window-based model
    window_dt = model.WindowBasedDecisionTree(partition_classifier)
    ground_truth, predictions = window_dt.evaluate_test_set(grouped_testing_dataset, train_features)

    # get the classification report
    report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)

    result_str = f"Max Depth = {max_depth}, "
    result_str += f"Feature Limit = {features_per_partition}, "
    result_str += f"Total Features = {total_features}, "
    result_str += f"Number of Partitions = {num_partitions}, "
    result_str += f"F1 Score = {report['macro avg']['f1-score']}, "
    result_str += f"Feature Table Entries = {feature_entries}, "
    result_str += f"Tree Table Entries = {table_entries}, "
    result_str += f"Number of flows = {num_flows}, "
    result_str += f"Partition Sizes = {partition_sizes}, "
    result_str += f"Resubmission Traffic = {resubmission_traffic}, "
    result_str += f"Model Features = {model_features}\n"

    with open(os.path.join(results_path, f"results-d{max_depth}.txt"), "a") as results_file:
        results_file.write(result_str)

    f1_score = report["macro avg"]["f1-score"]

    # the design is feasible if the resubmission traffic is within the bandwidth
    # and it's able to support some flows
    is_feasible = num_flows > 0 and is_resubmission_feasible
    if not is_resubmission_feasible:
        # this allows us to encode feasibility within number of flows
        f1_score *= 0
        num_flows *= 0

    return model.ModelConfig(
        max_depth=max_depth,
        features_per_partition=features_per_partition,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        c5=c5,
        c6=c6,
        num_partitions=num_partitions,
        f1_score=f1_score,
        num_flows=num_flows,
        feasible=is_feasible,
        total_features=total_features,
        feature_entries=feature_entries,
        table_entries=table_entries,
    )


def _bruteforce_train_arbitrary_partitions(parsed_args, results_path, max_depth):
    features_per_partition = parsed_args.bruteforce.parameter_space.features_per_partition
    max_cuts = parsed_args.bruteforce.max_cuts
    # generate some cuts for this depth to test
    generated_cuts = utils.generate_bruteforce_cuts(
        parsed_args=parsed_args,
        depth=max_depth - 1,
        max_cuts=max_cuts,
    )

    # loop over all cuts to test
    for i, (c1, c2, c3, c4, c5, c6) in enumerate(generated_cuts):
        # loop through all features per partition
        for feature_limit in features_per_partition:
            model_config = model.ModelConfig(
                max_depth=max_depth,
                features_per_partition=feature_limit,
                c1=c1,
                c2=c2,
                c3=c3,
                c4=c4,
                c5=c5,
                c6=c6,
            )
            try:
                model_config = single_run_train_arbitrary_partitions(
                    parsed_args, results_path, model_config
                )
            except:
                print(f"Error: Failed to train model with depth = {max_depth}, ", end="")
                print(f"feature limit = {feature_limit}, and cuts = {c1, c2, c3, c4, c5, c6}")

            # record this result to PostgreSQL
            db_manager.commit_to_logging_database(parsed_args, "cap", i, model_config)

            pass
        pass
    pass


def bruteforce_train_arbitrary_partitions(parsed_args):
    # record results here
    time_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    results_path = os.path.join(
        parsed_args.HOME_DIRECTORY,
        parsed_args.PROJECT_ROOT,
        "results",
        f"bruteforce-{parsed_args.dataset.name}-{time_now}",
    )
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    print("Results will be saved to: ", results_path)

    all_depths = parsed_args.bruteforce.parameter_space.depths
    jobs = [(parsed_args, results_path, this_depth) for this_depth in all_depths]
    # start parallel jobs for each tree depth
    num_jobs = parsed_args.bruteforce.num_jobs
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(_bruteforce_train_arbitrary_partitions, *job) for job in jobs]
        pass
    pass


def hypermapper_train_arbitrary_partitions(parsed_args):
    # record results here
    time_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    opt_method = parsed_args.hypermapper.scenario.optimization_method
    results_path = os.path.join(
        parsed_args.HOME_DIRECTORY,
        parsed_args.PROJECT_ROOT,
        "results",
        f"hypermapper-{opt_method}-{parsed_args.dataset.name}-{time_now}",
    )
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    print("Results will be saved to: ", results_path)

    # create experiment scenario
    scenario_path, objectives = scenario.get_experiment_scenario(parsed_args, results_path)

    # Command to launch HyperMapper
    hypermapper_cmd = [sys.executable, parsed_args.hypermapper.script_path, scenario_path]
    print(hypermapper_cmd)

    # Create a subprocess and launch HyperMapper
    pipe_to_hypermapper = Popen(
        hypermapper_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding="utf-8"
    )

    iterations = 0
    num_jobs = parsed_args.hypermapper.scenario.evaluations_per_optimization_iteration

    # Check if the process is still running
    while pipe_to_hypermapper.poll() is None:
        request_from_hypermapper = pipe_to_hypermapper.stdout.readline()
        # The first line is the request in the form: Request #_of_evaluation_requests
        pipe_to_hypermapper.stdout.flush()
        if not request_from_hypermapper:
            stderr = pipe_to_hypermapper.stderr.read()
            return_code = pipe_to_hypermapper.wait()
            raise RuntimeError(
                "HyperMapper exited before sending an evaluation request "
                f"(return code {return_code}).\n{stderr}"
            )

        if "End of HyperMapper" in request_from_hypermapper:
            # This means that HyperMapper ended
            print(request_from_hypermapper)
            break
        elif "warning" in request_from_hypermapper:
            continue

        if not request_from_hypermapper.startswith("Request "):
            stderr = pipe_to_hypermapper.stderr.read()
            raise RuntimeError(
                "Unexpected HyperMapper output. Expected 'Request <n>', got: "
                f"{request_from_hypermapper!r}\n{stderr}"
            )

        print(f"Iteration {iterations}")
        print("Request from HyperMapper:", request_from_hypermapper)

        # Get the #_of_evaluation_requests
        num_of_eval_requests = int(request_from_hypermapper.split(" ")[1])
        headers = pipe_to_hypermapper.stdout.readline()
        # The second line contains the header in the form: depth,features_per_partition,partitions
        pipe_to_hypermapper.stdin.flush()
        print(headers)

        # Prepare the response string, something like so
        # response_to_hypermapper = "depth,features_per_partition,c1,c2,c3,c4,c5,c6,c7,f1_score,num_flows,feasible\n"
        response_to_hypermapper = objectives.response_header

        jobs = []
        # collect evlauations requests as jobs
        for _ in range(num_of_eval_requests):
            # Go through the rest of the eval requests
            parameters_values = pipe_to_hypermapper.stdout.readline().strip()
            print("Received parameters: ", parameters_values)

            # This is an eval request in the form: number_x1, number_x2, number_x3
            parameters_values = [x.strip() for x in parameters_values.split(",")]
            depth = utils.get_parameter(parameters_values[0], "int")
            features_per_partition = utils.get_parameter(parameters_values[1], "int")
            c1 = utils.get_parameter(parameters_values[2], "float")
            c2 = utils.get_parameter(parameters_values[3], "float")
            c3 = utils.get_parameter(parameters_values[4], "float")
            c4 = utils.get_parameter(parameters_values[5], "float")
            c5 = utils.get_parameter(parameters_values[6], "float")
            c6 = utils.get_parameter(parameters_values[7], "float")

            model_config = model.ModelConfig(
                max_depth=depth,
                features_per_partition=features_per_partition,
                c1=c1,
                c2=c2,
                c3=c3,
                c4=c4,
                c5=c5,
                c6=c6,
            )
            jobs += [(parsed_args, results_path, model_config)]

        # start parallel jobs for each requested evaluation
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            futures = [executor.submit(single_run_train_arbitrary_partitions, *job) for job in jobs]
            # Retrieve results in the order of submission
            results = [future.result() for future in futures]

        # Reply to HyperMapper with all the evaluations
        iteration_best_model_metrics = model.ModelConfig()
        for i, result in enumerate(results):
            # if a configuration is successful, we return from the result
            # otherwise, we fetch the corresponding job parameters with 0 metrics
            try:
                model_metrics = result
            except:
                # fetch corresponding job parameters and return a failed model configuration
                (_, _, model_metrics) = jobs[i]
                print(
                    "Error: Configuration Failed: ",
                    f"{model_metrics.max_depth},{model_metrics.features_per_partition},",
                    f"{model_metrics.c1},{model_metrics.c2},{model_metrics.c3},",
                    f"{model_metrics.c4},{model_metrics.c5},{model_metrics.c6}",
                )

            # response for this job
            ret_str = f"{model_metrics.max_depth}"
            ret_str += f",{model_metrics.features_per_partition}"
            ret_str += f",{model_metrics.c1}"
            ret_str += f",{model_metrics.c2}"
            ret_str += f",{model_metrics.c3}"
            ret_str += f",{model_metrics.c4}"
            ret_str += f",{model_metrics.c5}"
            ret_str += f",{model_metrics.c6}"
            ret_str += f",{-1.0 * model_metrics.f1_score}" if objectives.f1_score else ""
            ret_str += f",{-1.0 * model_metrics.num_flows}" if objectives.num_flows else ""
            ret_str += f",{model_metrics.feasible}" if objectives.feasible else ""
            ret_str += "\n"

            # Add to the response string in a csv-style
            response_to_hypermapper += ret_str

            # get iteration's best performing model
            if model_metrics > iteration_best_model_metrics:
                iteration_best_model_metrics = model_metrics

            # record this performance to PostgreSQL
            db_manager.commit_to_logging_database(parsed_args, "cap", iterations, model_metrics)

            pass

        # Reply to HyperMapper with all the evaluations
        print("\nResponse to HyperMapper:")
        print(response_to_hypermapper)
        pipe_to_hypermapper.stdin.write(response_to_hypermapper)
        pipe_to_hypermapper.stdin.flush()

        iterations += 1

        pass
    pass

    save_splidt_selected_models(parsed_args, results_path)


def main():
    parsed_args = utils.parse_yml_config()
    assert sum(parsed_args.operational_mode.values()) == 1, (
        "Error: Select just one operational mode"
    )
    assert sum([parsed_args.switch_model.use_tofino1, parsed_args.switch_model.use_tofino2]) == 1, (
        "Error: Select just one switch model"
    )

    random.seed(time.time())

    # pick the mode chosen for training
    if parsed_args.operational_mode.single_run:
        # record results here
        time_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        results_path = os.path.join(
            parsed_args.HOME_DIRECTORY,
            parsed_args.PROJECT_ROOT,
            "results",
            f"one-config-{parsed_args.dataset.name}-{time_now}",
        )
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)
        print("Results will be saved to: ", results_path)

        # train and test a single tree configuration
        model_config = model.ModelConfig(
            max_depth=parsed_args.single_run.depth,
            features_per_partition=parsed_args.single_run.features_per_partition,
            c1=parsed_args.single_run.c1,
            c2=parsed_args.single_run.c2,
            c3=parsed_args.single_run.c3,
            c4=parsed_args.single_run.c4,
            c5=parsed_args.single_run.c5,
            c6=parsed_args.single_run.c6,
        )
        try:
            model_config = single_run_train_arbitrary_partitions(
                parsed_args, results_path, model_config, save_tree=True
            )
        except:
            print(f"Error: Failed to train model with depth = {model_config.max_depth}, ", end="")
            print(f"feature limit = {model_config.features_per_partition}, ", end="")
            print(f"and cuts = {model_config.c1}, {model_config.c2}, {model_config.c3}, ", end="")
            print(f"{model_config.c4}, {model_config.c5}, {model_config.c6}")

    elif parsed_args.operational_mode.bruteforce:
        parsed_args.db_table_name = ("bruteforce" + "-" + parsed_args.dataset.name).replace(
            "-", "_"
        )
        # create the logging database for grafana dashboard
        db_manager.create_logging_database(parsed_args)

        # brute-force exploration of the space of decision trees
        bruteforce_train_arbitrary_partitions(parsed_args)

    elif parsed_args.operational_mode.hypermapper:
        parsed_args.db_table_name = (
            "hypermapper"
            + "-"
            + parsed_args.dataset.name
            + "-"
            + parsed_args.hypermapper.scenario.optimization_method
        ).replace("-", "_")
        # create the logging database for grafana dashboard
        db_manager.create_logging_database(parsed_args)

        # client-server communication with HyperMapper
        # to explore the space of decision trees
        hypermapper_train_arbitrary_partitions(parsed_args)

    else:
        return Exception("No mode selected for training")


if __name__ == "__main__":
    main()
