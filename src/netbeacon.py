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
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import netbeacon_tcam_rules as nb_tcam_rules
import switch_resources as sw_resources
import utils


@dataclass
class NetBeaconModelConfig:
    max_depth: int = 0
    f1_score: float = 0.0
    num_flows: int = 0
    total_features: int = 0
    tcam_rules: int = 0

    def __str__(self):
        # return as a YAML list
        return (
            f"- max_depth: {self.max_depth}\n"
            + f"  f1_score: {self.f1_score}\n"
            + f"  num_flows: {self.num_flows}\n"
            + f"  num_features: {self.total_features}\n"
            + f"  num_tcam_rules: {self.tcam_rules}\n"
        )


# Get the top-k offloadable features
def select_top_k_features(dtree, num_features, all_features, remove_iat=False):
    importances = dtree.feature_importances_
    # Convert to DataFrame
    features_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    # this is a special case for ISCXVPN2016 that wouldn't extend to higher flows otherwise
    if remove_iat:
        # drop any feature containing 'IAT' in it
        features_df = features_df[~features_df["Feature"].str.contains("IAT")]
    top_k_features = features_df.sort_values(by="Importance", ascending=False).head(num_features)
    return top_k_features


def train_dtree_for_one_phase(
    ungrouped_training_dataset,
    ungrouped_testing_dataset,
    max_depth,
    max_leaf_nodes,
    this_phase,
    do_top_k_features=False,
    K=None,
    use_given_features=False,
    given_features=None,
):
    # select the phase and remove the 'Flow ID' and 'Phase' columns
    # get the training and testing dataset for first window for all flows
    X_train, y_train = utils.get_filtered_samples_and_labels(
        ungrouped_training_dataset, do_phase=True, phases=[this_phase]
    )

    X_test, y_test = utils.get_filtered_samples_and_labels(
        ungrouped_testing_dataset, do_phase=True, phases=[this_phase]
    )

    # drop the 'Flow ID' and 'Phase' columns
    X_train_without_id = X_train.drop(columns=["Flow ID", "Phase"], axis=1)
    # get all unique features and labels
    all_features = X_train_without_id.columns
    all_labels = y_train.unique()
    # Replace inf values with NaN
    X_train_without_id = X_train_without_id.replace([np.inf, -np.inf], np.nan)
    # Identify rows with NaN values
    dropped_indices = X_train_without_id[X_train_without_id.isna().any(axis=1)].index
    # Drop rows with NaN and reset index
    X_train_without_id = X_train_without_id.drop(dropped_indices).reset_index(drop=True)
    # X_train = X_train.drop(dropped_indices).reset_index(drop=True)
    y_train = y_train.drop(dropped_indices).reset_index(drop=True)

    # repeat the above for the testing dataset
    X_test_without_id = X_test.drop(columns=["Flow ID", "Phase"], axis=1)
    # Replace inf values with NaN
    X_test_without_id = X_test_without_id.replace([np.inf, -np.inf], np.nan)
    # Identify rows with NaN values
    dropped_indices = X_test_without_id[X_test_without_id.isna().any(axis=1)].index
    # Drop rows with NaN and reset index
    X_test_without_id = X_test_without_id.drop(dropped_indices).reset_index(drop=True)
    # X_test = X_test.drop(dropped_indices).reset_index(drop=True)
    y_test = y_test.drop(dropped_indices).reset_index(drop=True)

    if use_given_features:
        # pick only these top-k features from train and test samples
        X_train_without_id = X_train_without_id[given_features["Feature"]]
        X_test_without_id = X_test_without_id[given_features["Feature"]]

    # Train the model
    final_dtree = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        criterion="entropy",
        class_weight="balanced",
    )
    final_dtree.fit(X_train_without_id, y_train)
    y_pred = final_dtree.predict(X_test_without_id)
    # Get classification report
    final_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # train model with top-k features only
    if do_top_k_features:
        # select top-k features
        top_k_features = select_top_k_features(
            final_dtree,
            K,
            X_train_without_id.columns,  # , remove_iat=True
        )

        # train again with just these features,
        # but don't tell it to select top-k features again
        final_report, top_k_features, final_dtree = train_dtree_for_one_phase(
            ungrouped_training_dataset,
            ungrouped_testing_dataset,
            max_depth,
            max_leaf_nodes,
            this_phase,
            use_given_features=True,
            given_features=top_k_features,
        )
        return final_report, top_k_features, final_dtree

    this_model_top_k_features = select_top_k_features(
        final_dtree,
        len(X_train_without_id.columns),
        X_train_without_id.columns,  # , remove_iat=True
    )
    return final_report, this_model_top_k_features, final_dtree


def train_netbeacon(
    ungrouped_training_dataset,
    ungrouped_testing_dataset,
    max_depth,
    max_leaf_nodes,
    num_features,
    phases,
):
    # phase dtree models
    phase_dtrees = {this_phase: None for this_phase in phases}
    feature_table_entries, tree_table_entries = 0, 0

    # train from last phase to first, backwards
    last_phase = phases[-1]
    final_report, top_k_features, final_dtree = train_dtree_for_one_phase(
        ungrouped_training_dataset,
        ungrouped_testing_dataset,
        max_depth,
        max_leaf_nodes,
        last_phase,
        do_top_k_features=True,
        K=num_features,
    )
    phase_dtrees[last_phase] = (final_report, top_k_features, final_dtree)

    # keep all nodes
    tree_nodes_bool = np.ones(final_dtree.tree_.node_count, dtype=bool)
    # get the number of entries for this tree
    this_feat_entries, this_tree_entries = nb_tcam_rules.get_class_flow(
        final_dtree.tree_, tree_nodes_bool
    )
    feature_table_entries += len(this_feat_entries)
    tree_table_entries += len(this_tree_entries)

    # now loop from second last to first phase and train on same features
    for this_phase in reversed(phases[:-1]):
        this_phase_output = train_dtree_for_one_phase(
            ungrouped_training_dataset,
            ungrouped_testing_dataset,
            max_depth,
            max_leaf_nodes,
            this_phase,
            use_given_features=True,
            given_features=top_k_features,
        )
        this_phase_final_report, this_phase_top_k_features, this_phase_final_dtree = (
            this_phase_output
        )
        phase_dtrees[this_phase] = (
            this_phase_final_report,
            this_phase_top_k_features,
            this_phase_final_dtree,
        )

        # keep all nodes
        this_tree_nodes_bool = np.ones(this_phase_final_dtree.tree_.node_count, dtype=bool)
        # get the number of entries for this tree
        this_feat_entries, this_tree_entries = nb_tcam_rules.get_class_flow(
            this_phase_final_dtree.tree_, this_tree_nodes_bool
        )
        feature_table_entries += len(this_feat_entries)
        tree_table_entries += len(this_tree_entries)

    return phase_dtrees, top_k_features, feature_table_entries, tree_table_entries


def generate_test_set_for_netbeacon(ungrouped_testing_dataset, phases):
    # select all phases from given test set
    X_test, y_test = utils.get_filtered_samples_and_labels(
        ungrouped_testing_dataset, do_phase=True, phases=phases
    )
    # select only top-k features from test set
    # Replace inf values with NaN
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    # Identify rows with NaN values
    dropped_indices = X_test[X_test.isna().any(axis=1)].index
    # Drop rows with NaN and reset index
    X_test = X_test.drop(dropped_indices).reset_index(drop=True)
    y_test = y_test.drop(dropped_indices).reset_index(drop=True)

    # generate the flow-id to phases and labels dataset
    # create a dictionary from flow_id to its label and phases
    flow_id_to_phases_and_labels = {}
    with tqdm(
        total=len(X_test), desc="Preparing dataset for testing", unit=" phase samples"
    ) as pbar:
        for i, (row_i, sample) in enumerate(X_test.iterrows()):
            flow_id = sample["Flow ID"]
            processed_sample = sample.drop(["Flow ID", "Phase"])

            # if flow_id not in the dictionary, add it and its label
            if flow_id not in flow_id_to_phases_and_labels:
                flow_id_to_phases_and_labels[flow_id] = {}
                # row_i is the index of the sample, not i
                flow_id_to_phases_and_labels[flow_id]["label"] = y_test[row_i]

            # add the phase to the flow_id dictionary, in usable format
            flow_id_to_phases_and_labels[flow_id][sample["Phase"]] = processed_sample.to_frame().T
            # this should be always true
            if flow_id_to_phases_and_labels[flow_id]["label"] != y_test[row_i]:
                print("One label is not equal to the other phases")
                # delete this flow from the dictionary
                del flow_id_to_phases_and_labels[flow_id]
            # update the progress bar with one sample completed
            pbar.update(1)

    return flow_id_to_phases_and_labels


def test_netbeacon(flow_id_to_phases_and_labels, phases, phase_dtrees):
    # loop over all unique flow-ids and predict the label at each phase
    flow_id_to_ground_truth, flow_id_to_prediction = {}, {}
    with tqdm(
        total=len(flow_id_to_phases_and_labels), desc="Testing Netbeacon", unit=" flows"
    ) as pbar:
        for flow_id, phases_and_labels in flow_id_to_phases_and_labels.items():
            # for each phase, predict the label
            flow_id_to_ground_truth[flow_id] = phases_and_labels["label"]

            for this_phase in sorted(phases)[:-1]:
                # get the sample for this phase if it exists
                if this_phase not in phases_and_labels:
                    continue

                # get the model for this phase
                _, _, phase_model = phase_dtrees[this_phase]
                # pick this model's features
                model_features = phase_model.feature_names_in_
                # get the sample for this phase
                sample = phases_and_labels[this_phase][model_features]

                prediction = phase_model.predict(sample)
                confidence = max(phase_model.predict_proba(sample)[0])
                if confidence >= 0.8:
                    # found the prediction, break the loop
                    flow_id_to_prediction[flow_id] = prediction
                    break

                pass

            # if the prediction is not found, predict at the last phase
            if flow_id not in flow_id_to_prediction:
                # get last phase in this flow
                last_phase = max([p for p in phases_and_labels.keys() if p != "label"])

                # get the model for this phase
                _, _, phase_model = phase_dtrees[last_phase]
                # pick this model's features
                model_features = phase_model.feature_names_in_
                # get the sample for this phase
                sample = phases_and_labels[last_phase][model_features]
                # get the prediction for the last phase
                prediction = phase_model.predict(sample)
                flow_id_to_prediction[flow_id] = prediction
                pass

            # update the progress bar with one flow completed
            pbar.update(1)

        pass

    ordered_flow_keys = flow_id_to_ground_truth.keys()
    ground_truth = [flow_id_to_ground_truth[flow_id] for flow_id in ordered_flow_keys]
    predictions = [flow_id_to_prediction[flow_id] for flow_id in ordered_flow_keys]

    # get classification report
    report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
    return report


def main():
    parsed_args = utils.parse_yml_config()

    # record results here
    time_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    results_path = os.path.join(
        parsed_args.HOME_DIRECTORY,
        parsed_args.PROJECT_ROOT,
        "results",
        f"netbeacon-{parsed_args.dataset.name}-{time_now}",
    )
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    print("Results will be saved to: ", results_path)

    dataset_path = os.path.join(
        parsed_args.dataset.path,
        parsed_args.dataset.name,
        parsed_args.dataset.destination,
        f"dataset_df_p1024.pkl",  # hardcoded for Netbeacon
    )

    # Initialize logs and counters
    count = 0
    total = len(parsed_args.netbeacon.depths) * len(parsed_args.netbeacon.features)
    print(f"Total experiments: {total}")

    read_processed_df = utils.read_processed_dataset(dataset_path)
    ungrouped_training_dataset = read_processed_df["ungrouped_train_df"]
    ungrouped_testing_dataset = read_processed_df["ungrouped_test_df"]

    # phases are the same across any model configuration
    phases = parsed_args.netbeacon.phases
    # same dataset is needed across all of testing so prepare it once
    testing_dataset = generate_test_set_for_netbeacon(ungrouped_testing_dataset, phases)

    max_f1_score = {"f1_score": 0, "num_flows": 0}
    max_flows = {"f1_score": 0, "num_flows": 0}
    best_f1_str, best_flows_str = "", ""
    best_f1_artifact, best_flows_artifact = None, None
    netbeacon_performance = defaultdict(NetBeaconModelConfig)

    for max_depth in parsed_args.netbeacon.depths:
        for num_features in parsed_args.netbeacon.features:
            count += 1

            # Set the maximum number of leaf nodes (based on Netbeacon's trained models)
            max_leaf_nodes = min(2**max_depth, 512)

            # Run the model and evaluate the results
            phase_dtrees, top_k_features, feature_entries, table_entries = train_netbeacon(
                ungrouped_training_dataset,
                ungrouped_testing_dataset,
                max_depth,
                max_leaf_nodes,
                num_features,
                phases,
            )

            # # test netbeacon models
            class_report = test_netbeacon(testing_dataset, phases, phase_dtrees)
            f1_score = round(class_report["macro avg"]["f1-score"], 2)

            # do all the resource estimation like netbeacon
            # resources logic: select the model for resource estimation
            switch_model = parsed_args.switch_model.tofino1
            if parsed_args.switch_model.use_tofino2:
                switch_model = parsed_args.switch_model.tofino2

            FEATURE_WIDTH = switch_model.feature_width
            NODE_ID_WIDTH = switch_model.node_id_width
            NUM_MAU_STAGES = switch_model.num_ma_units
            # this is must for node_id (16-bit) and pkt_count (16-bit) ( always needed )
            BOOKKEEPING = 0  # switch_model.bookkeeping_stages
            # for IAT features only; include this if such a feature is being used
            IAT_BOOKKEEPING = switch_model.iat_bookkeeping_stages
            if any("IAT" in feat for feat in top_k_features["Feature"].tolist()):
                BOOKKEEPING += IAT_BOOKKEEPING

            # calculate number of stages
            feature_tcam_entries_per_key = sw_resources.calculate_allocated_tcam_entries(
                num_tcam_entries=feature_entries, width_per_key=FEATURE_WIDTH
            )

            table_tcam_entries_per_key = sw_resources.calculate_allocated_tcam_entries(
                num_tcam_entries=table_entries, width_per_key=FEATURE_WIDTH * num_features
            )

            feature_table_stages = sw_resources.tcam_entries_to_stages(
                num_tcam_entries=feature_tcam_entries_per_key,
                width_per_key=NODE_ID_WIDTH + (FEATURE_WIDTH * num_features),
            )

            tree_table_stage = sw_resources.tcam_entries_to_stages(
                num_tcam_entries=table_tcam_entries_per_key,
                width_per_key=NODE_ID_WIDTH + (FEATURE_WIDTH * num_features),
            )

            remaining_stages = NUM_MAU_STAGES - (feature_table_stages + tree_table_stage)
            num_flows = sw_resources.stages_to_registers(
                num_features=BOOKKEEPING + num_features, remaining_stages=remaining_stages
            )
            num_flows = int(num_flows)

            # for a given number of flows, keep the best f1 score
            if netbeacon_performance[num_flows].f1_score < f1_score:
                netbeacon_performance[num_flows] = NetBeaconModelConfig(
                    max_depth=max_depth,
                    f1_score=f1_score,
                    num_flows=num_flows,
                    total_features=len(top_k_features["Feature"]),
                    tcam_rules=feature_entries + table_entries,
                )

            print(f"({count}/{total}) Tree Depth = {max_depth}", end="")
            print(f", Number of Features = {len(top_k_features['Feature'])}", end="")
            print(f", Macro avg f1 score = {f1_score}, Number of flows = {num_flows}")

            # don't consider infeasible solutions
            if num_flows <= 0:
                continue

            result_str = f"Max Depth = {max_depth}, "
            result_str += f"Feature Limit = {len(top_k_features)}, "
            result_str += f"Total Features = {len(top_k_features)}, "
            result_str += f"Number of Phases = {len(phases)}, "
            result_str += f"F1 Score = {f1_score}, "
            result_str += f"Feature Table Entries = {feature_entries}, "
            result_str += f"Tree Table Entries = {table_entries}, "
            result_str += f"Number of flows = {num_flows}, "
            result_str += f"Partition Sizes = {max_depth}, "
            result_str += f"Resubmission Traffic = 0, "
            result_str += f"Model Features = {top_k_features['Feature'].tolist()}\n"

            with open(os.path.join(results_path, f"results-d{max_depth}.txt"), "a") as results_file:
                results_file.write(result_str)

            # pick best model configurations
            if f1_score >= max_f1_score["f1_score"]:
                max_f1_score["f1_score"] = f1_score
                max_f1_score["num_flows"] = num_flows
                best_f1_str = result_str
                best_f1_artifact = {
                    "model": phase_dtrees,
                    "metadata": {
                        "dataset": parsed_args.dataset.name,
                        "model_type": "netbeacon",
                        "selector": "best_by_f1",
                        "config": parsed_args.CONFIG_FILE,
                        "max_depth": max_depth,
                        "f1_score": f1_score,
                        "num_flows": num_flows,
                        "num_features": len(top_k_features["Feature"]),
                        "phases": phases,
                        "feature_entries": feature_entries,
                        "tree_entries": table_entries,
                        "tcam_rules": feature_entries + table_entries,
                        "features": top_k_features["Feature"].tolist(),
                        "result": result_str.strip(),
                    },
                }

            if num_flows >= max_flows["num_flows"]:
                max_flows["f1_score"] = f1_score
                max_flows["num_flows"] = num_flows
                best_flows_str = result_str
                best_flows_artifact = {
                    "model": phase_dtrees,
                    "metadata": {
                        "dataset": parsed_args.dataset.name,
                        "model_type": "netbeacon",
                        "selector": "best_by_flows",
                        "config": parsed_args.CONFIG_FILE,
                        "max_depth": max_depth,
                        "f1_score": f1_score,
                        "num_flows": num_flows,
                        "num_features": len(top_k_features["Feature"]),
                        "phases": phases,
                        "feature_entries": feature_entries,
                        "tree_entries": table_entries,
                        "tcam_rules": feature_entries + table_entries,
                        "features": top_k_features["Feature"].tolist(),
                        "result": result_str.strip(),
                    },
                }

            pass
        pass

    print()
    print(f"Maximum F1 Score: {max_f1_score}")
    print(best_f1_str)
    print()
    print(f"Maximum Number of Flows: {max_flows}")
    print(best_flows_str)
    print()

    for _, netbeacon_config in netbeacon_performance.items():
        print(netbeacon_config)

    if best_f1_artifact is not None:
        utils.save_pickle_artifact(
            parsed_args,
            "netbeacon",
            "best_by_f1",
            {"phase_dtrees": best_f1_artifact["model"]},
            best_f1_artifact["metadata"],
        )

    if best_flows_artifact is not None:
        utils.save_pickle_artifact(
            parsed_args,
            "netbeacon",
            "best_by_flows",
            {"phase_dtrees": best_flows_artifact["model"]},
            best_flows_artifact["metadata"],
        )

    pass


if __name__ == "__main__":
    main()
