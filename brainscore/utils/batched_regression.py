from abc import ABC, abstractmethod
import numpy.typing as npt


class RegressionModelBatched(ABC):

    @abstractmethod
    def fit_partial(self, source: npt.ArrayLike, target: npt.ArrayLike) -> float:
        """
        Partially fit model to a batch of data.
        :param source: Batch of inputs.
        :param target: Batch of targets.
        :returns: Loss on the current batch.
        """
        pass

    @abstractmethod
    def predict(self, source: npt.ArrayLike) -> npt.ArrayLike:
        """
        Get predictions using the fitted model.
        :param source: A set of inputs (probably a batch, if the dataset is large).
        :returns: An array of predictions.
        """
        pass


# todo: implement some batched regression models (that can use GPU)
