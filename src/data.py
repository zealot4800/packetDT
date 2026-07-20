from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetConfig


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_flow_ids: pd.Series | None = None
    test_flow_ids: pd.Series | None = None
    train_sample_weights: pd.Series | None = None


@dataclass
class WindowDataset:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def _dataset_file(dataset: DatasetConfig, filename: str) -> Path:
    path = dataset.directory / filename
    if not path.exists():
        raise FileNotFoundError(f"required processed dataset file is missing: {path}")
    return path


def _read_processed_pickle(path: Path) -> dict[str, pd.DataFrame]:
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise ValueError(f"processed pickle must contain a dictionary: {path}")
    required = ["ungrouped_train_df", "ungrouped_test_df"]
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"{path} missing processed split keys: {', '.join(missing)}")
    return obj


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required column(s): {', '.join(missing)}")


def prepare_features_and_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: DatasetConfig,
    metadata_columns: list[str],
) -> DatasetSplit:
    label = dataset.label_column
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    _require_columns(train_df, [label], "training dataframe")
    _require_columns(test_df, [label], "test dataframe")

    y_train = train_df[label].reset_index(drop=True)
    y_test = test_df[label].reset_index(drop=True)
    train_flow_ids = train_df[dataset.flow_id_column].astype(str).reset_index(drop=True) if dataset.flow_id_column in train_df else None
    test_flow_ids = test_df[dataset.flow_id_column].astype(str).reset_index(drop=True) if dataset.flow_id_column in test_df else None
    drop_columns = [label] + [column for column in metadata_columns if column in train_df.columns]
    X_train = train_df.drop(columns=drop_columns, errors="ignore")
    X_test = test_df.drop(columns=drop_columns, errors="ignore").reindex(columns=X_train.columns)

    for frame in [X_train, X_test]:
        for column in frame.select_dtypes(include=["object"]).columns:
            frame[column] = pd.factorize(frame[column])[0]

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    train_good = ~X_train.isna().any(axis=1)
    test_good = ~X_test.isna().any(axis=1)
    removed_train = int((~train_good).sum())
    removed_test = int((~test_good).sum())
    if removed_train or removed_test:
        print(f"Removed invalid rows: train={removed_train}, test={removed_test}")

    return DatasetSplit(
        X_train=X_train.loc[train_good].reset_index(drop=True),
        y_train=y_train.loc[train_good].reset_index(drop=True),
        X_test=X_test.loc[test_good].reset_index(drop=True),
        y_test=y_test.loc[test_good].reset_index(drop=True),
        train_flow_ids=train_flow_ids.loc[train_good].reset_index(drop=True) if train_flow_ids is not None else None,
        test_flow_ids=test_flow_ids.loc[test_good].reset_index(drop=True) if test_flow_ids is not None else None,
    )


def load_full_flow_dataset(dataset: DatasetConfig) -> DatasetSplit:
    processed = _read_processed_pickle(_dataset_file(dataset, dataset.full_flow_file))
    return prepare_features_and_labels(
        processed["ungrouped_train_df"],
        processed["ungrouped_test_df"],
        dataset,
        [dataset.flow_id_column, dataset.window_column, dataset.phase_column, dataset.packet_column],
    )


def load_partition_dataset(dataset: DatasetConfig, num_partitions: int) -> WindowDataset:
    filename = dataset.partition_file_pattern.format(num_partitions=num_partitions)
    processed = _read_processed_pickle(_dataset_file(dataset, filename))
    for key in ["ungrouped_train_df", "ungrouped_test_df"]:
        _require_columns(
            processed[key],
            [dataset.flow_id_column, dataset.window_column, dataset.label_column],
            f"{filename}:{key}",
        )
    return WindowDataset(processed["ungrouped_train_df"].copy(), processed["ungrouped_test_df"].copy())


def split_window_dataset(dataset: DatasetConfig, window_data: WindowDataset, windows: list[int]) -> DatasetSplit:
    train_df = window_data.train_df[window_data.train_df[dataset.window_column].isin(windows)].copy()
    test_df = window_data.test_df[window_data.test_df[dataset.window_column].isin(windows)].copy()
    return prepare_features_and_labels(
        train_df,
        test_df,
        dataset,
        [dataset.flow_id_column, dataset.window_column, dataset.phase_column, dataset.packet_column],
    )


def load_phase_dataset(dataset: DatasetConfig, phases: tuple[int, ...]) -> DatasetSplit:
    processed = _read_processed_pickle(_dataset_file(dataset, dataset.phase_file))
    phase_col = dataset.phase_column if dataset.phase_column in processed["ungrouped_train_df"] else dataset.window_column
    for key in ["ungrouped_train_df", "ungrouped_test_df"]:
        _require_columns(processed[key], [phase_col, dataset.label_column], f"{dataset.phase_file}:{key}")
    train_df = processed["ungrouped_train_df"][processed["ungrouped_train_df"][phase_col].isin(phases)].copy()
    test_df = processed["ungrouped_test_df"][processed["ungrouped_test_df"][phase_col].isin(phases)].copy()
    return prepare_features_and_labels(
        train_df,
        test_df,
        dataset,
        [dataset.flow_id_column, dataset.window_column, dataset.phase_column, dataset.packet_column],
    )


def load_adaflow_dataset(
    dataset: DatasetConfig,
    trigger_packet: int,
    phase_delta: int,
    num_dense_phases: int,
) -> tuple[DatasetSplit, tuple[int, ...]]:
    """Build PL-only early samples and FL-only dense samples (Algorithm 1)."""
    packet_data = _read_processed_pickle(_dataset_file(dataset, dataset.packet_file))
    phase_data = _read_processed_pickle(_dataset_file(dataset, dataset.phase_file))
    phase_col = dataset.phase_column if dataset.phase_column in phase_data["ungrouped_train_df"] else dataset.window_column
    for key in ["ungrouped_train_df", "ungrouped_test_df"]:
        _require_columns(phase_data[key], [phase_col, dataset.label_column], f"{dataset.phase_file}:{key}")
        _require_columns(packet_data[key], [dataset.label_column], f"{dataset.packet_file}:{key}")

    dense = {trigger_packet + index * phase_delta for index in range(num_dense_phases)}
    max_dense = max(dense)
    powers = {2**index for index in range(max_dense.bit_length()) if 2**index <= max_dense}
    early = set(range(1, trigger_packet))
    requested = tuple(sorted(early | dense | powers))
    train_available = set(phase_data["ungrouped_train_df"][phase_col].dropna().astype(int).unique())
    test_available = set(phase_data["ungrouped_test_df"][phase_col].dropna().astype(int).unique())
    available = train_available & test_available
    phases = tuple(phase for phase in requested if phase in available)
    if not phases:
        raise ValueError(f"none of the configured AdaFlow phases are present in {dataset.phase_file}")

    metadata = {dataset.label_column, dataset.flow_id_column, dataset.window_column, dataset.phase_column, dataset.packet_column, "Packet ID"}

    def aggregate(key: str) -> pd.DataFrame:
        packet_df = packet_data[key].copy()
        packet_position = dataset.packet_column if dataset.packet_column in packet_df else None
        if packet_position:
            packet_df = packet_df[packet_df[packet_position] < trigger_packet].copy()
        packet_features = [column for column in packet_df if column not in metadata]
        packet_df = packet_df.rename(columns={column: f"pl::{column}" for column in packet_features})
        packet_df["__adaflow_phase"] = packet_df[packet_position] if packet_position else 1
        packet_df["__adaflow_weight"] = 1.0

        flow_df = phase_data[key][phase_data[key][phase_col].isin(phases) & (phase_data[key][phase_col] >= trigger_packet)].copy()
        flow_features = [column for column in flow_df if column not in metadata]
        flow_df = flow_df.rename(columns={column: f"fl::{column}" for column in flow_features})
        flow_df["__adaflow_phase"] = flow_df[phase_col]
        flow_df["__adaflow_weight"] = flow_df[phase_col].map(
            lambda phase: max(1.0, trigger_packet * float(np.exp(float(phase) / trigger_packet - 1.0)))
        )
        combined = pd.concat([packet_df, flow_df], ignore_index=True, sort=False)
        feature_columns = [column for column in combined if column.startswith(("pl::", "fl::"))]
        # Phi in Algorithm 1: a value deliberately outside attainable feature ranges.
        combined[feature_columns] = combined[feature_columns].fillna(-1.0e30)
        return combined

    train_df = aggregate("ungrouped_train_df")
    test_df = aggregate("ungrouped_test_df")
    raw_weights = train_df["__adaflow_weight"].reset_index(drop=True)
    split = prepare_features_and_labels(
        train_df,
        test_df,
        dataset,
        [dataset.flow_id_column, dataset.window_column, dataset.phase_column, dataset.packet_column, "Packet ID", "__adaflow_phase", "__adaflow_weight"],
    )
    split.train_sample_weights = raw_weights
    return split, phases


def load_packet_dataset(dataset: DatasetConfig, packet_index: int) -> DatasetSplit:
    processed = _read_processed_pickle(_dataset_file(dataset, dataset.packet_file))
    train_df = processed["ungrouped_train_df"].copy()
    test_df = processed["ungrouped_test_df"].copy()
    if dataset.packet_column in train_df.columns:
        filtered_train = train_df[train_df[dataset.packet_column] == packet_index].copy()
        filtered_test = test_df[test_df[dataset.packet_column] == packet_index].copy()
        if filtered_train.empty or filtered_test.empty:
            first_train_packet = train_df[dataset.packet_column].min()
            first_test_packet = test_df[dataset.packet_column].min()
            print(
                f"Packet index {packet_index} is unavailable; using first observed "
                f"packet indexes train={first_train_packet}, test={first_test_packet}."
            )
            filtered_train = train_df[train_df[dataset.packet_column] == first_train_packet].copy()
            filtered_test = test_df[test_df[dataset.packet_column] == first_test_packet].copy()
        train_df = filtered_train
        test_df = filtered_test
    return prepare_features_and_labels(
        train_df,
        test_df,
        dataset,
        [dataset.flow_id_column, dataset.window_column, dataset.phase_column, dataset.packet_column, "Packet ID"],
    )
