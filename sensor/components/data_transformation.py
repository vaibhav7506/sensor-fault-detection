"""Training-only preprocessing and resampling."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sensor.constant.training_pipeline import RANDOM_STATE, TARGET_COLUMN
from sensor.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import SensorException
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        # Zero is a plausible sensor reading, so train-fitted median is safer than zero-fill.
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            features = [column for column in train_df.columns if column != TARGET_COLUMN]
            mapping = TargetValueMapping().to_dict()
            y_train = train_df[TARGET_COLUMN].replace(mapping)
            y_test = test_df[TARGET_COLUMN].replace(mapping)
            preprocessor = self.get_data_transformer_object().fit(train_df[features])
            x_train = preprocessor.transform(train_df[features])
            x_test = preprocessor.transform(test_df[features])
            # This is deliberately the only resampling call in the pipeline.
            x_train, y_train = SMOTETomek(random_state=RANDOM_STATE, sampling_strategy="minority").fit_resample(x_train, y_train)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, np.c_[x_train, y_train])
            # The final holdout retains its original rows and class distribution.
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, np.c_[x_test, y_test])
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            return DataTransformationArtifact(self.data_transformation_config.transformed_object_file_path, self.data_transformation_config.transformed_train_file_path, self.data_transformation_config.transformed_test_file_path)
        except Exception as exc:
            raise SensorException(exc, sys) from exc
