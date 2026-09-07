"""Schema checks and numeric-feature drift reporting."""
from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

from sensor.constant.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.exception import SensorException
from sensor.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def drop_zero_std_columns(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        columns = [c for c in dataframe.columns if c != TARGET_COLUMN and dataframe[c].nunique(dropna=True) <= 1]
        return dataframe.drop(columns=columns), columns

    def validate_frame(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        expected = {next(iter(item)) for item in self._schema_config["columns"]}
        required = expected - set(self._schema_config.get("drop_columns", []))
        missing = sorted(required - set(dataframe.columns))
        features = dataframe.drop(columns=[TARGET_COLUMN], errors="ignore")
        non_numeric = features.select_dtypes(exclude="number").columns.tolist()
        return {
            "valid": not missing and TARGET_COLUMN in dataframe and dataframe[TARGET_COLUMN].dropna().nunique() == 2 and not non_numeric,
            "missing_required": missing,
            "all_null_columns": features.columns[features.isna().all()].tolist(),
            "zero_variance_columns": features.columns[features.nunique(dropna=True).le(1)].tolist(),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "non_numeric_columns": non_numeric,
            "missing_fraction": {key: float(value) for key, value in dataframe.isna().mean().items()},
        }

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        rows: list[dict[str, Any]] = []
        for column in base_df.select_dtypes(include="number").columns.intersection(current_df.columns):
            left, right = base_df[column].dropna(), current_df[column].dropna()
            if len(left) < 2 or len(right) < 2 or left.nunique() <= 1 or right.nunique() <= 1:
                rows.append({"feature": column, "test": "ks_2samp", "statistic": None, "raw_p_value": None, "adjusted_p_value": None, "drift_detected": False})
                continue
            result = ks_2samp(left, right)
            rows.append({"feature": column, "test": "ks_2samp", "statistic": float(result.statistic), "raw_p_value": float(result.pvalue)})
        tested = sorted((row for row in rows if row["raw_p_value"] is not None), key=lambda row: row["raw_p_value"])
        total = len(tested)
        corrected = [min(1.0, row["raw_p_value"] * total / (index + 1)) for index, row in enumerate(tested)]
        for index in range(total - 2, -1, -1): corrected[index] = min(corrected[index], corrected[index + 1])
        for row, p_value in zip(tested, corrected): row.update(adjusted_p_value=p_value, drift_detected=p_value < threshold)
        os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
        write_yaml_file(self.data_validation_config.drift_report_file_path, {"features": rows})
        return not any(row.get("drift_detected", False) for row in rows)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            reports = {"train": self.validate_frame(train), "test": self.validate_frame(test)}
            if not all(report["valid"] for report in reports.values()):
                raise ValueError(f"Critical schema validation failure: {reports}")
            status = self.detect_dataset_drift(train.drop(columns=[TARGET_COLUMN]), test.drop(columns=[TARGET_COLUMN]))
            return DataValidationArtifact(status, self.data_ingestion_artifact.trained_file_path, self.data_ingestion_artifact.test_file_path, None, None, self.data_validation_config.drift_report_file_path)
        except Exception as exc:
            raise SensorException(exc, sys) from exc
