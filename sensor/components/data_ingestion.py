"""Data ingestion with deterministic, stratified holdout creation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sensor.constant.training_pipeline import RANDOM_STATE, SCHEMA_FILE_PATH, TARGET_COLUMN
from sensor.data_access.sensor_data import SensorData
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.config_entity import DataIngestionConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils.main_utils import read_yaml_file


class DataIngestion:
    """Ingest from an explicitly supplied CSV or the configured Mongo collection."""

    def __init__(self, data_ingestion_config: DataIngestionConfig, source_path: str | None = None):
        self.data_ingestion_config = data_ingestion_config
        self.source_path = source_path
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            if self.source_path:
                dataframe = pd.read_csv(self.source_path).replace("na", pd.NA)
                logging.info("Loaded local CSV source: %s rows", len(dataframe))
            else:
                dataframe = SensorData().export_collection_as_dataframe(self.data_ingestion_config.collection_name)
                logging.info("Loaded configured MongoDB collection: %s rows", len(dataframe))
            Path(self.data_ingestion_config.feature_store_file_path).parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            return dataframe
        except Exception as exc:
            raise SensorException(exc, sys) from exc

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """Create a reproducible final holdout; it is never resampled or fitted."""
        try:
            if TARGET_COLUMN not in dataframe.columns:
                raise ValueError(f"Missing required target column {TARGET_COLUMN!r}")
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=RANDOM_STATE, stratify=dataframe[TARGET_COLUMN],
            )
            Path(self.data_ingestion_config.training_file_path).parent.mkdir(parents=True, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            logging.info("Created stratified split: train=%s test=%s", len(train_set), len(test_set))
        except Exception as exc:
            raise SensorException(exc, sys) from exc

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        dataframe = self.export_data_into_feature_store()
        drop_columns = [c for c in self._schema_config.get("drop_columns", []) if c in dataframe]
        self.split_data_as_train_test(dataframe.drop(columns=drop_columns))
        return DataIngestionArtifact(self.data_ingestion_config.training_file_path, self.data_ingestion_config.testing_file_path)
