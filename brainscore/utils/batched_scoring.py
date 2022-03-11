from abc import ABC, abstractmethod

import numpy as np
import torch


class ScoringBatched(ABC):

    @abstractmethod
    def update(self, preds: np.ndarray, target: np.ndarray) -> None:
        """
        Update internal parameters needed to compute score, using a batch of predictions and targets.
        E.g. For accuracy, we would increment a running count of num_correct and num_predicted.
        :param preds: A batch of predictions.
        :param target: A batch of targets.
        """
        pass

    @abstractmethod
    def compute(self) -> np.ndarray:
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


class PearsonrScoringBatched(ScoringBatched):
    # WARNING: this online method only becomes accurate for large sample sizes and large batches

    def __init__(self, device=None):
        self._device = device
        self._mean_x = None
        self._mean_y = None
        self._var_x = None
        self._var_y = None
        self._corr_xy = None
        self._n_prior = None

    def update(self, preds: np.ndarray, target: np.ndarray) -> None:
        self._initialize_from(preds)
        preds, target = torch.from_numpy(preds).to(self._device), torch.from_numpy(target).to(self._device)
        n_obs = len(preds)
        mx_new = (self._n_prior * self._mean_x + preds.mean(dim=0) * n_obs) / (self._n_prior + n_obs)
        my_new = (self._n_prior * self._mean_y + target.mean(dim=0) * n_obs) / (self._n_prior + n_obs)
        self._n_prior += n_obs
        self._var_x += ((preds - mx_new) * (preds - self._mean_x)).sum(dim=0)
        self._var_y += ((target - my_new) * (target - self._mean_y)).sum(dim=0)
        self._corr_xy += ((preds - mx_new) * (target - my_new)).sum(dim=0)
        self._mean_x = mx_new
        self._mean_y = my_new

    def compute(self) -> np.ndarray:
        assert self._mean_x is not None
        var_x = self._var_x / (self._n_prior - 1)
        var_y = self._var_y / (self._n_prior - 1)
        corr_xy = self._corr_xy / (self._n_prior - 1)
        r = (corr_xy / (var_x * var_y).sqrt()).squeeze()
        return r.cpu().numpy()

    def reset(self) -> None:
        self._mean_x = None
        self._mean_y = None
        self._var_x = None
        self._var_y = None
        self._corr_xy = None
        self._n_prior = None

    def _initialize_from(self, array: np.ndarray):
        if self._mean_x is None:
            self._mean_x = torch.zeros(array.shape[1]).to(self._device)
            self._mean_y = torch.zeros(array.shape[1]).to(self._device)
            self._var_x = torch.zeros(array.shape[1]).to(self._device)
            self._var_y = torch.zeros(array.shape[1]).to(self._device)
            self._corr_xy = torch.zeros(array.shape[1]).to(self._device)
            self._n_prior = torch.zeros(array.shape[1]).to(self._device)


# todo: implement some batched scoring functions. A lot of these can just be wrappers around
# what's already in the TorchMetrics package, which have a similar interface.
