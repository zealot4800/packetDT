from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from .config import ExperimentConfig, load_experiment_config
from .degradation import run_flow_count_metrics
from .leo import run_leo
from .llsy import run_llsy
from .netbeacon import run_netbeacon
from .splidt import run_splidt
from .statedt import run_statedt, synthetic_example
from .tree import ModelResult


Runner = Callable[[ExperimentConfig, Path], ModelResult]
RUNNERS: dict[str, Runner] = {
    "splidt": run_splidt,
    "llsy": run_llsy,
    "iisy": run_llsy,
    "netbeacon": run_netbeacon,
    "leo": run_leo,
    "statedt": run_statedt,
}

MODEL_ORDER = ["splidt", "llsy", "netbeacon", "leo", "statedt"]
MODEL_LABELS = {
    "splidt": "SpliDT",
    "llsy": "LLSY",
    "netbeacon": "NetBeacon",
    "leo": "LEO",
    "statedt": "StateDT",
}


def dataset_results_dir(config: ExperimentConfig) -> Path:
    return Path("results") / config.dataset.name


def model_output_dir(config: ExperimentConfig, model: str) -> Path:
    return dataset_results_dir(config) / ("llsy" if model == "iisy" else model)


def run_one(command: str, config: ExperimentConfig) -> ModelResult:
    model = "llsy" if command == "iisy" else command
    result = RUNNERS[command](config, model_output_dir(config, model))
    print(f"{result.model}: macro_f1={result.macro_f1}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["splidt", "llsy", "iisy", "netbeacon", "leo", "statedt", "degradation", "all", "example"])
    parser.add_argument("--config", default="configs/datasets/cic_ids_2017.yaml")
    args = parser.parse_args()

    if args.command == "example":
        synthetic_example()
        return

    config = load_experiment_config(args.config)
    if args.command == "degradation":
        output_paths = run_flow_count_metrics(config)
        print("Saved flow-count F1 metrics to:")
        for output_path in output_paths:
            print(output_path)
        return
    if args.command == "all":
        for model in MODEL_ORDER:
            try:
                run_one(model, config)
            except FileNotFoundError as exc:
                print(f"{MODEL_LABELS[model]} skipped: {exc}")
        return
    run_one(args.command, config)


if __name__ == "__main__":
    main()
