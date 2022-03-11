from abc import ABC, abstractmethod
import numpy as np


class RegressionModelBatched(ABC):

    @abstractmethod
    def fit_partial(self, source: np.ndarray, target: np.ndarray) -> float:
        """
        Partially fit model to a batch of data.
        :param source: Batch of inputs.
        :param target: Batch of targets.
        :returns: Loss on the current batch.
        """
        pass

    @abstractmethod
    def predict(self, source: np.ndarray) -> np.ndarray:
        """
        Get predictions using the fitted model.
        :param source: A set of inputs (probably a batch, if the dataset is large).
        :returns: An array of predictions.
        """
        pass


# todo: implement some batched regression models (that can use GPU)
