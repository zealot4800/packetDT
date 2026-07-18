from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import load_phase_dataset
from .resources import TargetProfile, estimate_netbeacon_resources
from .tree import ModelResult, calculate_macro_f1, fit_tree, select_top_k_features


class NetBeacon:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, output_dir: Path) -> ModelResult:
        split = load_phase_dataset(self.config.dataset, self.config.netbeacon.phases)
        selected = select_top_k_features(split.X_train, split.y_train, self.config.netbeacon.max_features, self.config.seed)
        model = fit_tree(split.X_train[selected], split.y_train, self.config.netbeacon.max_depth, self.config.seed)
        predictions = model.predict(split.X_test[selected])
        macro_f1 = calculate_macro_f1(split.y_test, predictions)
        resources = estimate_netbeacon_resources(
            TargetProfile.from_config(self.config.target),
            len(selected),
            model.tree_.node_count,
            len(self.config.netbeacon.phases),
        )
        result = ModelResult(
            model="NetBeacon",
            dataset=self.config.dataset.name,
            target=self.config.target.name,
            macro_f1=macro_f1,
            max_depth=model.get_depth(),
            num_features=len(selected),
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
        save_model_outputs(output_dir, result, {"model": model, "features": selected})
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


def run_netbeacon(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return NetBeacon(config).run(output_dir)
