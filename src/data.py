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
