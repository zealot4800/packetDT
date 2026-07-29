from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .data import load_packet_dataset
from .resources import TargetProfile, estimate_llsy_resources
from .tree import ModelResult, calculate_macro_f1, fit_tree, save_model_artifacts, select_top_k_features


class LLSY:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, output_dir: Path) -> ModelResult:
        split = load_packet_dataset(self.config.dataset, self.config.llsy.packet_index)
        selected = select_top_k_features(
            split.X_train,
            split.y_train,
            self.config.llsy.max_features,
            self.config.seed,
        )
        model = fit_tree(split.X_train[selected], split.y_train, self.config.llsy.max_depth, self.config.seed)
        predictions = model.predict(split.X_test[selected])
        macro_f1 = calculate_macro_f1(split.y_test, predictions)
        resources = estimate_llsy_resources(
            TargetProfile.from_config(self.config.target),
            len(selected),
            model.tree_.node_count,
        )
        result = ModelResult.from_resources(
            model="LLSY",
            dataset=self.config.dataset.name,
            target=self.config.target.name,
            seed=self.config.seed,
            macro_f1=macro_f1,
            max_depth=model.get_depth(),
            num_features=len(selected),
            test_samples=len(split.y_test),
            resources=resources,
        )
        save_model_artifacts(output_dir, result, {"model": model, "features": selected})
        return result


def run_llsy(config: ExperimentConfig, output_dir: Path) -> ModelResult:
    return LLSY(config).run(output_dir)
