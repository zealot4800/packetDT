from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .data import load_adaflow_dataset
from .resources import TargetProfile, estimate_adaflow_resources
from .tree import ModelResult, calculate_macro_f1, extract_tree_nodes, fit_tree, save_model_artifacts, select_top_k_features, write_json


class AdaFlow:
    """Single multi-phase aggregated tree with PrioritySketch rule metadata."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, output_dir: Path) -> ModelResult:
        settings = self.config.adaflow
        split, phases = load_adaflow_dataset(
            self.config.dataset,
            settings.trigger_packet,
            settings.phase_delta,
            settings.num_dense_phases,
        )
        selected = select_top_k_features(split.X_train, split.y_train, settings.max_features, self.config.seed)
        model = fit_tree(
            split.X_train[selected],
            split.y_train,
            settings.max_depth,
            self.config.seed,
            sample_weight=split.train_sample_weights,
        )
        predictions = model.predict(split.X_test[selected])
        macro_f1 = calculate_macro_f1(split.y_test, predictions)
        resources = estimate_adaflow_resources(
            TargetProfile.from_config(self.config.target), len(selected), model.tree_.node_count, max(phases)
        )
        result = ModelResult.from_resources(
            model="AdaFlow", dataset=self.config.dataset.name, target=self.config.target.name,
            seed=self.config.seed, macro_f1=macro_f1, max_depth=model.get_depth(),
            num_features=len(selected), test_samples=len(split.y_test), num_partitions=1,
            resources=resources,
        )
        nodes = extract_tree_nodes(model, selected)
        for node, probabilities in zip(nodes, model.tree_.value):
            confidence = float(probabilities[0].max() / probabilities[0].sum())
            node["confidence"] = confidence
            node["priority"] = int(confidence <= settings.determination_threshold) if node["is_leaf"] else None
        payload = {
            "model": model, "features": selected, "phases": phases,
            "trigger_packet": settings.trigger_packet, "determination_threshold": settings.determination_threshold,
            "priority_rules": nodes,
        }
        save_model_artifacts(output_dir, result, payload)
        write_json(
            str(output_dir / "adaflow.json"),
            {key: value for key, value in payload.items() if key != "model"},
        )
        return result


def run_adaflow(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return AdaFlow(config).run(output_dir)
