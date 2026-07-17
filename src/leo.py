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

import leo_tcam_rules
import utils


@dataclass
class LeoModelConfig:
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


def train_dtree(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes):
    # Train the model
    dtree = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        criterion="entropy",
        class_weight="balanced",
    )
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report, dtree


def train_leo(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes, num_features):
    # train decision tree with all features
    _, dtree = train_dtree(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes)

    # select top-k features
    top_k_features = select_top_k_features(dtree, num_features, X_train.columns)

    # pick only these top-k features from train and test samples
    X_train_top_k = X_train[top_k_features["Feature"]]
    X_test_top_k = X_test[top_k_features["Feature"]]

    # train again with just these features
    final_report, final_dtree = train_dtree(
        X_train_top_k, y_train, X_test_top_k, y_test, max_depth, max_leaf_nodes
    )

    return final_report, top_k_features, final_dtree


def main():
    parsed_args = utils.parse_yml_config()

    # record results here
    time_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    results_path = os.path.join(
        parsed_args.HOME_DIRECTORY,
        parsed_args.PROJECT_ROOT,
        "results",
        f"leo-{parsed_args.dataset.name}-{time_now}",
    )
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    print("Results will be saved to: ", results_path)

    dataset_path = os.path.join(
        parsed_args.dataset.path,
        parsed_args.dataset.name,
        parsed_args.dataset.destination,
        f"dataset_df_p1.pkl",  # hardcoded for Leo
    )

    # Initialize logs and counters
    count = 0
    total = len(parsed_args.leo.depths) * len(parsed_args.leo.features)
    print(f"Total number of experiments to run: {total}")

    read_processed_df = utils.read_processed_dataset(dataset_path)
    ungrouped_training_dataset = read_processed_df["ungrouped_train_df"]
    ungrouped_testing_dataset = read_processed_df["ungrouped_test_df"]

    # get the training and testing dataset for first window for all flows
    X_train, y_train = utils.get_filtered_samples_and_labels(
        ungrouped_training_dataset, windows=[1]
    )

    X_test, y_test = utils.get_filtered_samples_and_labels(ungrouped_testing_dataset, windows=[1])

    print(f"Training dataset shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing dataset shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    # drop the 'Flow ID' and 'Window' columns
    X_train_without_id = X_train.drop(columns=["Flow ID", "Window"], axis=1)
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
    X_test_without_id = X_test.drop(columns=["Flow ID", "Window"], axis=1)
    # Replace inf values with NaN
    X_test_without_id = X_test_without_id.replace([np.inf, -np.inf], np.nan)
    # Identify rows with NaN values
    dropped_indices = X_test_without_id[X_test_without_id.isna().any(axis=1)].index
    # Drop rows with NaN and reset index
    X_test_without_id = X_test_without_id.drop(dropped_indices).reset_index(drop=True)
    # X_test = X_test.drop(dropped_indices).reset_index(drop=True)
    y_test = y_test.drop(dropped_indices).reset_index(drop=True)

    max_f1_score = {"f1_score": 0, "num_flows": 0}
    max_flows = {"f1_score": 0, "num_flows": 0}
    best_f1_str, best_flows_str = "", ""
    best_f1_artifact, best_flows_artifact = None, None
    leo_performance = defaultdict(LeoModelConfig)

    for max_depth in parsed_args.leo.depths:
        for num_features in parsed_args.leo.features:
            count += 1

            # Set the maximum number of leaf nodes
            max_leaf_nodes = None
            if max_depth <= 13:
                max_leaf_nodes = int(2**max_depth)
            elif max_depth == 14:
                max_leaf_nodes = 6144
            elif max_depth == 15:
                max_leaf_nodes = 3072
            elif max_depth == 16:
                max_leaf_nodes = 2048
            else:
                max_leaf_nodes = 1024

            # Run the model and evaluate the results
            classification_reports, top_k_features, dtree = train_leo(
                X_train_without_id,
                y_train,
                X_test_without_id,
                y_test,
                max_depth,
                max_leaf_nodes,
                num_features,
            )

            # calculate the number of nodes in the tree
            num_nodes = dtree.tree_.node_count
            min_features = min(num_features, num_nodes)

            # print(f"Top k features are: {top_k_features}")

            use_iat_feature = False
            # check if IAT is in any string in top-k features
            if any("IAT" in item for item in top_k_features["Feature"].tolist()):
                # print("IAT is in the top-k features.")
                use_iat_feature = True
                min_features = min_features + 1

            # Leo can only handle a maximum of 7 features, due to its hardware limitations
            if min_features > 7:
                continue

            tcam_entries, num_flows = leo_tcam_rules.get_num_stages(
                tree_depth=max_depth, features=min_features, iat=use_iat_feature
            )

            # get the macro average f1 score from classification report
            f1_score = round(classification_reports["macro avg"]["f1-score"], 2)
            num_flows = int(num_flows)

            # for a given number of flows, keep the best f1 score
            if leo_performance[num_flows].f1_score < f1_score:
                leo_performance[num_flows] = LeoModelConfig(
                    max_depth=max_depth,
                    f1_score=f1_score,
                    num_flows=num_flows,
                    total_features=min_features,
                    tcam_rules=tcam_entries,
                )

            print(f"({count}/{total}) Tree Depth = {max_depth}", end="")
            print(f", Number of Features = {len(top_k_features)}", end="")
            print(f", Macro avg f1 score = {f1_score}, Number of flows = {num_flows}")

            # don't consider infeasible solutions
            if num_flows <= 0:
                continue

            result_str = f"Max Depth = {max_depth}, "
            result_str += f"Feature Limit = {len(top_k_features)}, "
            result_str += f"Total Features = {len(top_k_features)}, "
            result_str += f"Number of Partitions = 1, "
            result_str += f"F1 Score = {f1_score}, "
            result_str += f"Feature Table Entries = 0, "
            result_str += f"Tree Table Entries = {tcam_entries}, "
            result_str += f"Number of flows = {num_flows}, "
            result_str += f"Partition Sizes = {max_depth}, "
            result_str += f"Resubmission Traffic = 0, "
            result_str += f"Model Features = {top_k_features['Feature'].tolist()}\n"

            with open(os.path.join(results_path, f"results-d{max_depth}.txt"), "a") as results_file:
                results_file.write(result_str)

            # pick best model configurations
            if f1_score > max_f1_score["f1_score"]:
                max_f1_score["f1_score"] = f1_score
                max_f1_score["num_flows"] = num_flows
                best_f1_str = result_str
                best_f1_artifact = {
                    "model": dtree,
                    "metadata": {
                        "dataset": parsed_args.dataset.name,
                        "model_type": "leo",
                        "selector": "best_by_f1",
                        "config": parsed_args.CONFIG_FILE,
                        "max_depth": max_depth,
                        "f1_score": f1_score,
                        "num_flows": num_flows,
                        "num_features": len(top_k_features),
                        "tcam_rules": tcam_entries,
                        "features": top_k_features["Feature"].tolist(),
                        "result": result_str.strip(),
                    },
                }

            if num_flows > max_flows["num_flows"]:
                max_flows["f1_score"] = f1_score
                max_flows["num_flows"] = num_flows
                best_flows_str = result_str
                best_flows_artifact = {
                    "model": dtree,
                    "metadata": {
                        "dataset": parsed_args.dataset.name,
                        "model_type": "leo",
                        "selector": "best_by_flows",
                        "config": parsed_args.CONFIG_FILE,
                        "max_depth": max_depth,
                        "f1_score": f1_score,
                        "num_flows": num_flows,
                        "num_features": len(top_k_features),
                        "tcam_rules": tcam_entries,
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

    for _, leo_config in leo_performance.items():
        print(leo_config)

    if best_f1_artifact is not None:
        utils.save_pickle_artifact(
            parsed_args,
            "leo",
            "best_by_f1",
            {"model": best_f1_artifact["model"]},
            best_f1_artifact["metadata"],
        )

    if best_flows_artifact is not None:
        utils.save_pickle_artifact(
            parsed_args,
            "leo",
            "best_by_flows",
            {"model": best_flows_artifact["model"]},
            best_flows_artifact["metadata"],
        )

    pass


if __name__ == "__main__":
    main()
