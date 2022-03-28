import torch
from scipy.stats import pearsonr

from brainscore.metrics.xarray_utils import XarrayCorrelation, Defaults as XarrayDefaults
from brainscore.metrics.transformations import TestOnlyCrossValidation


class CrossCorrelation:
    def __init__(self, stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim,
                 test_size=.8, splits=5):
        self._correlation = XarrayCorrelation(pearsonr, correlation_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        self._cross_validation = TestOnlyCrossValidation(test_size=test_size, splits=splits)
        self._neuroid_dim = neuroid_dim

    def __call__(self, source, target):
        return self._cross_validation(source, target, apply=self._correlation, aggregate=self.aggregate)

    def aggregate(self, scores):
        return scores.median(dim=self._neuroid_dim)


class Correlation:
    def __init__(self, stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim):
        self._correlation = XarrayCorrelation(pearsonr, correlation_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        self._neuroid_dim = neuroid_dim

    def __call__(self, source, target):
        correlation = self._correlation(source, target)
        return self.aggregate(correlation)

    def aggregate(self, scores):
        return scores.median(dim=self._neuroid_dim)


def pairwise_corrcoef(
    x: torch.Tensor,
    y: torch.Tensor = None,
    return_diagonal: bool = True,
    device: torch.device = None,
    batch_size: int = None,  # TODO implement batching
) -> torch.Tensor:
    """Compute the pairwise correlation between columns of tensors x (and y, optionally).

    :param x: first tensor, shape (n_samples, n_features_x)
    :type x: torch.Tensor
    :param y: first tensor, shape (n_samples, n_features_y), defaults to None
    :type y: torch.Tensor, optional
    :param return_diagonal: whether to return correlations only for corresponding columns (if y is passed), defaults to True
    :type return_diagonal: bool, optional
    :param device: CPU or GPU, defaults to None
    :type device: torch.device, optional
    :return: correlations
    :rtype: torch.Tensor
    """
    if device is not None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
    n_features_x, n_features_y = x.shape[0], None
    if y is not None:
        n_features_y = y.shape[0]
        if return_diagonal:
            assert n_features_x == n_features_y, "diagonal does not exist: x and y have different shapes"
        x = torch.concat((x, y), dim=0).to(device)
        y = None
    x = torch.corrcoef(x)
    if n_features_y is not None:
        x = x[:n_features_x, n_features_x:]
    if return_diagonal:
        x = torch.diag(x)
    return x
