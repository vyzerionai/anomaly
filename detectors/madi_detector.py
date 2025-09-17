import os

import keras
import numpy as np
import pandas as pd
from absl import logging
from anomaly.utils import sample_utils

_MODEL_FILENAME = "model-multivariate-ad.keras"
_NORMALIZATION_FILENAME = "normalization_info"


class MadiDetector:
    """Anomaly detection using negative sampling and a neural net classifier."""

    def __init__(self):
        self._normalization_info = None
        self._model = None
        keras.backend.clear_session()

    def train_model(self, x_train: pd.DataFrame):
        """Train a new model and report the loss and accuracy."""
        raise NotImplementedError("This is inference only")

    def predict(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        """Given new data, predict the probability of being positive class.

        Args:
          sample_df: dataframe with features as columns, same as train().

        Returns:
          DataFrame as sample_df, with colum 'class_prob', prob of Normal class.
        """
        sample_df_normalized = sample_utils.normalize(
            sample_df, self._normalization_info
        )
        column_order = sample_utils.get_column_order(self._normalization_info)
        x = np.float32(np.matrix(sample_df_normalized[column_order]))
        y_hat = self._model.predict(x, verbose=1, steps=1)
        sample_df["class_prob"] = y_hat
        return sample_df

    def save_model(self, model_dir: str) -> None:
        """Saves the trained AD model to the model directory model_dir."""
        raise NotImplementedError("This is inference only")

    def load_model(self, model_dir: str) -> None:
        """Loads the trained AD model from the model directory model_dir."""
        model_file_path = os.path.join(model_dir, _MODEL_FILENAME)
        self._model = keras.models.load_model(model_file_path)
        logging.info("Successfully loaded model from %s" % model_file_path)
        normalization_file_path = os.path.join(model_dir, _NORMALIZATION_FILENAME)
        self._normalization_info = sample_utils.read_normalization_info(
            normalization_file_path
        )
        logging.info(
            "Sucessfully read normalization info from %s" % normalization_file_path
        )

    @property
    def model(self):
        return self._model

    @property
    def normalization_info(self):
        return self._normalization_info
