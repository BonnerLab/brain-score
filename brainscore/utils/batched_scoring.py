from abc import ABC, abstractmethod
import numpy.typing as npt


class ScoringBatched(ABC):

    @abstractmethod
    def update(self, preds: npt.ArrayLike, target: npt.ArrayLike) -> None:
        """
        Update internal parameters needed to compute score, using a batch of predictions and targets.
        E.g. For accuracy, we would increment a running count of num_correct and num_predicted.
        :param preds: A batch of predictions.
        :param target: A batch of targets.
        """
        pass

    @abstractmethod
    def compute(self) -> npt.ArrayLike:
        """
        Called after .update() has been called on all batches. Actually computes the score on the whold dataset.
        E.g. For accuracy, we would divide num_correct / num_predicted.
        :returns: A vector of scores (one for each target variable).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset any internal variables being used in .update().
        """
        pass


# todo: implement some batched scoring functions (e.g. correlation). A lot of these can just be wrappers around
# what's already in the TorchMetrics package, which have a similar interface.
