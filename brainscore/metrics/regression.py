from typing import Optional, Dict

import scipy.stats
import numpy as np
import torch
import xarray as xr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import scale

from brainio.assemblies import walk_coords
from brainscore.metrics.mask_regression import MaskRegression
from brainscore.metrics.transformations import CrossValidation, CrossValidationLazy
from brainscore.utils.batched_regression import LinearRegressionBatched
from brainscore.utils.batched_scoring import PearsonrScoringBatched
from .xarray_utils import XarrayCorrelationEfficient, XarrayRegression, XarrayCorrelation, XarrayRegressionLazy, XarrayRegressionScoreBatched, Defaults, map_target_to_source


class CrossRegressedCorrelation:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class CrossRegressedCorrelationLazy:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidationLazy(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source, target_train, target_test):
        self.regression.fit(source, target_train)
        stimulus_dim = Defaults.expected_dims[0]
        stimulus_coord = Defaults.stimulus_coord
        source_test = source.isel({stimulus_dim: map_target_to_source(source, target_test, stimulus_coord)})
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class ScaledCrossRegressedCorrelation:
    def __init__(self, *args, **kwargs):
        self.cross_regressed_correlation = CrossRegressedCorrelation(*args, **kwargs)
        self.aggregate = self.cross_regressed_correlation.aggregate

    def __call__(self, source, target):
        scaled_values = scale(target, copy=True)
        target = target.__class__(scaled_values, coords={
            coord: (dims, value) for coord, dims, value in walk_coords(target)}, dims=target.dims)
        return self.cross_regressed_correlation(source, target)


class SingleRegression():
    def __init__(self):
        self.mapping = []

    def fit(self, X, Y):
        X = X.values
        Y = Y.values
        n_stim, n_neuroid = X.shape
        _, n_neuron = Y.shape
        r = np.zeros((n_neuron, n_neuroid))
        for neuron in range(n_neuron):
            r[neuron, :] = pearsonr(X, Y[:, neuron:neuron+1])
        self.mapping = np.nanargmax(r, axis=1)

    def predict(self, X):
        X = X.values
        Ypred = X[:, self.mapping]
        return Ypred


class TorchLinearRegression:
    def __init__(self, device: torch.device = None):
        if torch.device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.betas = None

    def fit(self, x, y):
        x = self._cast_to_torch(x)
        y = self._cast_to_torch(y)
        self.betas, _, _, _ = torch.linalg.lstsq(x, y)

    def predict(self, x):
        if self.betas is None:
            raise RuntimeError("model has not been fit")
        else:
            x = self._cast_to_torch(x)
            return torch.matmul(x, self.betas).cpu().numpy()

    def _cast_to_torch(self, x):
        if isinstance(x, xr.DataArray):
            x = x.values
        return torch.from_numpy(x).float().to(self.device)


def mask_regression():
    regression = MaskRegression()
    regression = XarrayRegression(regression)
    return regression


def pls_regression(regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = PLSRegression(**regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def linear_regression(xarray_kwargs=None):
    regression = LinearRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def linear_regression_efficient(xarray_kwargs=None):
    regression = TorchLinearRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegressionLazy(regression, **xarray_kwargs)
    return regression


def ridge_regression(xarray_kwargs=None):
    regression = Ridge()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def pearsonr_correlation(xarray_kwargs=None):
    xarray_kwargs = xarray_kwargs or {}
    return XarrayCorrelation(scipy.stats.pearsonr, **xarray_kwargs)


def pearsonr_correlation_efficient(xarray_kwargs=None, backend="torch", device: torch.device = None):
    xarray_kwargs = xarray_kwargs or {}
    if backend == "torch":
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)
        def _torch_corrcoef(prediction, target):
            n_half = target.sizes["neuroid"]
            return (
                torch.diag(
                    torch.corrcoef(
                        torch.concat(
                            (
                                torch.from_numpy(prediction.values.transpose()).to(device),
                                torch.from_numpy(target.values.transpose()).to(device),
                            ),
                            dim=0,
                        )
                    )[:n_half, n_half:]
                )
                .cpu()
                .numpy()
            )
        _correlation = _torch_corrcoef
    return XarrayCorrelationEfficient(_correlation, **xarray_kwargs)


def single_regression(xarray_kwargs=None):
    regression = SingleRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def pearsonr(x, y):
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True)
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True)

    r = ((xm/normxm)*(ym/normym)).sum(axis=0)

    return r


#############################################################################
########################## Batched regression ###############################
#############################################################################


class CrossRegressedCorrelationBatched:
    def __init__(self,
                 regression_score: Optional[XarrayRegressionScoreBatched] = None,
                 crossvalidation_kwargs: Dict = None):
        regression_score = regression_score or linear_regression_pearsonr_batched()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidationLazy(**crossvalidation_kwargs)
        self.regression_score = regression_score

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source, target_train, target_test):
        self.regression_score.fit(source, target_train)
        score = self.regression_score.score(source, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


def linear_regression_pearsonr_batched(regression_kwargs=None, corr_kwargs=None, xarray_kwargs=None):
    if regression_kwargs is None:
        regression_kwargs = {}
    if corr_kwargs is None:
        corr_kwargs = {}
    if xarray_kwargs is None:
        xarray_kwargs = {}
    regression = LinearRegressionBatched(**regression_kwargs)
    scoring = PearsonrScoringBatched(**corr_kwargs)
    regression_score = XarrayRegressionScoreBatched(regression=regression, scoring=scoring, **xarray_kwargs)
    return regression_score
