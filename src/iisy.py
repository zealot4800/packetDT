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

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

import utils


@dataclass
class IIsyModelConfig:
    max_depth: int = 0
    f1_score: float = 0.0
    total_features: int = 0

    def __str__(self):
        # return as a YAML list
        return (
            f"- max_depth: {self.max_depth}\n"
            + f"  f1_score: {self.f1_score}\n"
            + f"  num_features: {self.total_features}\n"
        )


# Get the top-k offloadable features
def select_top_k_features(dtree, num_features, all_features):
    importances = dtree.feature_importances_
    # Convert to DataFrame
    features_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    top_k_features = features_df.sort_values(by="Importance", ascending=False).head(num_features)
    return top_k_features


def train_dtree(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes, extra):
    # Train the model
    dtree = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        criterion="entropy",
        class_weight="balanced",
    )
    dtree.fit(X_train, y_train)

    # Predict for each packet in the test set
    extra["Prediction"] = dtree.predict(X_test)

    # Ensure 'Label' is available in test_X
    extra["Label"] = y_test.values

    # Group by Flow ID and assign the most common prediction as the flow's label
    flow_predictions = extra.groupby("Flow ID")["Prediction"].apply(
        lambda x: x.value_counts().idxmax()
    )

    # Get the true labels for each Flow ID (majority vote of true labels per flow)
    flow_true_labels = extra.groupby("Flow ID")["Label"].apply(lambda x: x.value_counts().idxmax())

    report = classification_report(
        flow_true_labels, flow_predictions, output_dict=True, zero_division=0
    )
    # print(report)
    return report, dtree


def train_IIsy(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes, num_features, extra):
    # train decision tree with all features
    _, dtree = train_dtree(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes, extra)

    # select top-k features
    top_k_features = select_top_k_features(dtree, num_features, X_train.columns)

    # pick only these top-k features from train and test samples
    X_train_top_k = X_train[top_k_features["Feature"]]
    X_test_top_k = X_test[top_k_features["Feature"]]

    # train again with just these features
    final_report, final_dtree = train_dtree(
        X_train_top_k, y_train, X_test_top_k, y_test, max_depth, max_leaf_nodes, extra
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
        f"IIsy-{parsed_args.dataset.name}-{time_now}",
    )
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    print("Results will be saved to: ", results_path)

    dataset_path = os.path.join(
        parsed_args.dataset.path,
        parsed_args.dataset.name,
        parsed_args.dataset.destination,
        f"dataset_df_p0.pkl",  # hardcoded for IIsy
    )

    # Initialize logs and counters
    total = min(len(parsed_args.iisy.depths), len(parsed_args.iisy.features))
    print(f"Total number of experiments to run: {total}")

    iisy_performance = IIsyModelConfig()
    best_model_str = ""
    best_f1_artifact = None
    count = 0

    read_processed_df = utils.read_processed_dataset(dataset_path)
    train_df = read_processed_df["ungrouped_train_df"].copy()
    test_df = read_processed_df["ungrouped_test_df"].copy()

    # Encode any non-numeric packet fields while preserving the processed split.
    for df in (train_df, test_df):
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = pd.factorize(df[column])[0]

    drop_columns = ["Label", "Packet ID", "Source Port"]
    train_X = train_df.drop(columns=[col for col in drop_columns if col in train_df.columns])
    test_X = test_df.drop(columns=[col for col in drop_columns if col in test_df.columns])
    y_train = train_df["Label"]
    y_test = test_df["Label"]

    # Prepare features without 'Flow ID' for the model
    model_drop_columns = [col for col in ["Flow ID", "Packet"] if col in train_X.columns]
    X_train_without_id = train_X.drop(columns=model_drop_columns)
    X_test_without_id = test_X.drop(columns=model_drop_columns)

    print(X_train_without_id.columns)
    print(len(X_train_without_id))
    print(len(X_test_without_id))

    # zip together iisy depth and features
    iisy_configs = zip(parsed_args.iisy.depths, parsed_args.iisy.features)
    for max_depth, num_features in iisy_configs:
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
        classification_reports, top_k_features, dtree = train_IIsy(
            X_train_without_id,
            y_train,
            X_test_without_id,
            y_test,
            max_depth,
            max_leaf_nodes,
            num_features,
            test_X,
        )

        # calculate the number of nodes in the tree
        num_nodes = dtree.tree_.node_count
        min_features = min(num_features, num_nodes)

        print(f"Top k features are: {top_k_features}")

        # get the macro avg f1 score from the classification report
        f1_score = round(classification_reports["macro avg"]["f1-score"], 2)
        print(f"({count}/{total}) The depth of the tree is {max_depth}", end="")
        print(f", the number of features used are {len(top_k_features)}", end="")
        print(f" -> Macro avg f1 score: {f1_score}")

        result_str = f"Max Depth = {max_depth}, "
        result_str += f"Feature Limit = {len(top_k_features)}, "
        result_str += f"Total Features = {len(top_k_features)}, "
        result_str += f"Number of Partitions = 1, "
        result_str += f"F1 Score = {f1_score}, "
        result_str += f"Feature Table Entries = 0, "
        result_str += f"Tree Table Entries = nan, "
        result_str += f"Number of flows = nan, "
        result_str += f"Partition Sizes = {max_depth}, "
        result_str += f"Resubmission Traffic = 0, "
        result_str += f"Model Features = {top_k_features['Feature'].tolist()}\n"

        with open(os.path.join(results_path, f"results-d{max_depth}.txt"), "a") as results_file:
            results_file.write(result_str)

        # select the best model configuration
        if f1_score > iisy_performance.f1_score:
            best_model_str = result_str
            iisy_performance = IIsyModelConfig(
                max_depth=max_depth, f1_score=f1_score, total_features=len(top_k_features)
            )
            best_f1_artifact = {
                "model": dtree,
                "metadata": {
                    "dataset": parsed_args.dataset.name,
                    "model_type": "iisy",
                    "selector": "best_by_f1",
                    "config": parsed_args.CONFIG_FILE,
                    "max_depth": max_depth,
                    "f1_score": f1_score,
                    "num_features": len(top_k_features),
                    "features": top_k_features["Feature"].tolist(),
                    "result": result_str.strip(),
                },
            }

        pass

    print()
    print(f"Max score: {iisy_performance.f1_score}")
    print(best_model_str)
    print()
    print(iisy_performance)

    if best_f1_artifact is not None:
        utils.save_pickle_artifact(
            parsed_args,
            "iisy",
            "best_by_f1",
            {"model": best_f1_artifact["model"]},
            best_f1_artifact["metadata"],
        )

    pass


if __name__ == "__main__":
    main()
