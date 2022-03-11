from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn, optim


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


class LinearRegressionBatched(RegressionModelBatched):

    def __init__(self, lr=1e-4, fit_intercept=True, l1_strength=0.0, l2_strength=0.0, device=None):
        self._lr = lr
        self._fit_intercept = fit_intercept
        self._l1_strength = l1_strength
        self._l2_strength = l2_strength
        self._device = device

        self._loss_func = nn.MSELoss(reduction='sum')
        self._linear = None
        self._optimizer = None

    def fit_partial(self, source: np.ndarray, target: np.ndarray) -> float:
        self._initialize_from(source, target)
        source, target = torch.from_numpy(source).to(self._device), torch.from_numpy(target).to(self._device)

        loss = self._loss_func(self._linear(source), target)

        # L1 regularizer
        if self._l1_strength > 0:
            l1_reg = self._linear.weight.abs().sum()
            loss += self._l1_strength * l1_reg

        # L2 regularizer
        if self._l2_strength > 0:
            l2_reg = self._linear.weight.pow(2).sum()
            loss += self._l2_strength * l2_reg

        loss /= source.size(0)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def predict(self, source: np.ndarray) -> np.ndarray:
        assert self._linear is not None
        source = torch.from_numpy(source).to(self._device)
        preds = self._linear(source)
        return preds.numpy()

    def _initialize_from(self, source: np.ndarray, target: np.ndarray):
        if self._linear is None:
            self._linear = nn.Linear(source.shape[1], target.shape[1],
                                     bias=self._fit_intercept, device=self._device)
            self._optimizer = optim.Adam(self._linear.parameters(), lr=self._lr)


# todo: implement some batched regression models (that can use GPU)
