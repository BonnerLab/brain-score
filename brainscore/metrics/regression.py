from typing import Optional, Dict, Union, Tuple

import scipy.stats
import numpy as np
import torch
import xarray as xr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import scale

from brainio.assemblies import walk_coords, NeuroidAssembly
from brainscore.metrics import Metric, Score
from brainscore.metrics.correlation import pairwise_corrcoef
from brainscore.metrics.mask_regression import MaskRegression
from brainscore.metrics.transformations import CrossValidation, CrossValidationLazy
from brainscore.utils.batched_regression import LinearRegressionBatched
from brainscore.utils.batched_scoring import PearsonrScoringBatched
from .xarray_utils import XarrayRegression, XarrayCorrelation, XarrayRegressionScoreBatched, map_target_to_source


class CrossRegressedCorrelation(Metric):
    def __init__(self, regression: XarrayRegression, correlation: XarrayCorrelation, lazy=False, crossvalidation_kwargs=None) -> None:
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        if lazy:
            self.cross_validation = CrossValidationLazy(**crossvalidation_kwargs)
        else:
            self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source: NeuroidAssembly, target: NeuroidAssembly) -> Score:
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train: NeuroidAssembly, target_train: NeuroidAssembly, source_test: NeuroidAssembly, target_test: NeuroidAssembly) -> Score:
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores: Score) -> Score:
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


class TestSetCorrelation(Metric):
    # TODO add tests
    def __init__(
        self,
        regression: XarrayRegression,
        correlation: XarrayCorrelation,
        stimulus_dim: str = "presentation",
        stimulus_coord: str = "image_id",
        train_coord: str = "",
        test_coord: str = "",
    ) -> None:
        self.regression = regression
        self.correlation = correlation
        self._stimulus_dim = stimulus_dim
        self._stimulus_coord = stimulus_coord
        self._train_coord = train_coord
        self._test_coord = test_coord

    def __call__(self, source: NeuroidAssembly, target: NeuroidAssembly) -> Score:
        mask_train, mask_test = self._create_masks(target)
        target_train = self._filter(target, mask_train)
        self.regression.fit(source, target_train)

        target_test = self._filter(target, mask_test)
        source_test = source.isel({self._stimulus_dim: map_target_to_source(source, target_test, self._stimulus_dim)})
        prediction = self.regression.predict(source_test)

        score = self.correlation(prediction, target_test)
        aggregated_score = self.aggregate(score)
        aggregated_score.attrs["raw"] = score
        return aggregated_score

    def _create_masks(self, target: NeuroidAssembly) -> Tuple[np.ndarray, np.ndarray]:
        if self._train_coord:
            mask_train = target[self._train_coord].astype(bool).values
            if self._test_coord:
                mask_test = target[self._test_coord].astype(bool).values
                assert ~np.any(mask_train & mask_test), "train and test data overlap"
            else:
                mask_test = ~mask_train
        else:
            if self._test_coord:
                mask_test = target[self._test_coord].astype(bool).values
                mask_train = ~mask_test
            else:
                mask_train = np.full(target.sizes[self._stimulus_dim], True, dtype=bool)
                mask_test = np.full(target.sizes[self._stimulus_dim], True, dtype=bool)
        return mask_train, mask_test

    def _filter(self, x: NeuroidAssembly, mask: np.ndarray) -> NeuroidAssembly:
        return x.isel({self._stimulus_dim: mask})

    def aggregate(self, scores: Score) -> Score:
        return scores.median(dim='neuroid')


class SingleRegression():
    def __init__(self) -> None:
        self.mapping = []

    def fit(self, X: NeuroidAssembly, Y: NeuroidAssembly) -> None:
        X = X.values
        Y = Y.values
        n_stim, n_neuroid = X.shape
        _, n_neuron = Y.shape
        r = np.zeros((n_neuron, n_neuroid))
        for neuron in range(n_neuron):
            r[neuron, :] = pearsonr(X, Y[:, neuron:neuron+1])
        self.mapping = np.nanargmax(r, axis=1)

    def predict(self, X: NeuroidAssembly) -> np.ndarray:
        X = X.values
        Ypred = X[:, self.mapping]
        return Ypred


class LinearRegressionPytorch:
    def __init__(
        self,
        fit_intercept: bool = True,
        device: Union[torch.device, str] = None
    ) -> None:
        self.fit_intercept = fit_intercept
        self.device_ = device
        self.n_features_in_ = None
        self.coef_ = None
        self.intercept_ = None
        self._residues = None
        self.rank_ = None
        self.singular_ = None

    def fit(self, x: xr.DataArray, y: xr.DataArray) -> None:
        x = self._cast_to_torch(x)
        n_samples_, self.n_features_in_ = x.shape

        y = self._cast_to_torch(y)
        if self.fit_intercept:
            x = torch.cat([x, torch.ones(n_samples_, device=self.device_).unsqueeze(1)], dim=1)

        self.coef_, self._residues, self.rank_, self.singular_ = torch.linalg.lstsq(x, y)

        if self.fit_intercept:
            self.intercept_ = self.coef_[-1, :]
        else:
            self.intercept_ = torch.zeros(self.n_features_in_)

        self.coef_ = self.coef_[:-1, :].transpose_(0, 1)

    def predict(self, x: xr.DataArray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model has not been fit")
        else:
            x = self._cast_to_torch(x)
            x = torch.matmul(x, self.coef_.transpose_(0, 1))
            x += self.intercept_
            return x.cpu().numpy()

    def _cast_to_torch(self, x: xr.DataArray) -> torch.Tensor:
        return torch.from_numpy(x.values).float().to(self.device_)


class RidgeRegressionPytorch:
    # adapted from https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
    def __init__(
        self,
        regularization: float = 1,
        fit_intercept: bool = True,
        device: Union[torch.device, str] = None
    ):
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.device_ = device

    def fit(self, x: xr.DataArray, y: xr.DataArray) -> None:
        x = self._cast_to_torch(x)
        y = self._cast_to_torch(y)

        assert x.shape[0] == y.shape[0], "number of X and y rows don't match"
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1, device=self.device_), x], dim=1)

        lhs = x.T @ x
        rhs = x.T @ y
        if self.regularization == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.regularization * torch.eye(lhs.shape[0], device=self.device)
            self.w, _ = torch.lstsq(rhs, lhs + ridge)

    def predict(self, x: torch.tensor) -> None:
        x = self._cast_to_torch(x)
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1, device=self.device_), x], dim=1)
        return (x @ self.w).cpu().numpy()

    def _cast_to_torch(self, x: xr.DataArray) -> torch.Tensor:
        return torch.from_numpy(x.values).float().to(self.device_)


def mask_regression():
    regression = MaskRegression()
    regression = XarrayRegression(regression)
    return regression


def pls_regression(regression_kwargs=None, xarray_kwargs=None) -> XarrayRegression:
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = PLSRegression(**regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def linear_regression(xarray_kwargs=None, backend: str = "sklearn", torch_kwargs=None) -> XarrayRegression:
    if backend == "sklearn":
        regression = LinearRegression()
    elif backend == "pytorch":
        torch_kwargs = torch_kwargs or {}
        regression = LinearRegressionPytorch(**torch_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def ridge_regression(xarray_kwargs=None, backend: str = "sklearn", sklearn_kwargs=None, torch_kwargs=None) -> XarrayRegression:
    # TODO figure out why the code doesn't seem to allow for varying the regularization strength
    if backend == "sklearn":
        regression = Ridge(**sklearn_kwargs)
        print(regression.alpha)
    elif backend == "pytorch":
        torch_kwargs = torch_kwargs or {}
        regression = RidgeRegressionPytorch(**torch_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def pearsonr_correlation(xarray_kwargs=None, parallel=True, corrcoef_kwargs=None) -> XarrayCorrelation:
    xarray_kwargs = xarray_kwargs or {}
    corrcoef_kwargs = corrcoef_kwargs or {}

    if parallel:
        def corrcoef(prediction: NeuroidAssembly, target: NeuroidAssembly, **corrcoef_kwargs) -> np.ndarray:
            return (
                pairwise_corrcoef(
                    x=torch.from_numpy(prediction.values.transpose()),
                    y=torch.from_numpy(target.values.transpose()),
                    return_diagonal=True,
                    **corrcoef_kwargs,
                )
                .cpu()
                .numpy()
            )
        return XarrayCorrelation(corrcoef, parallel=True, **xarray_kwargs)
    else:
        return XarrayCorrelation(scipy.stats.pearsonr, parallel=False, **xarray_kwargs)


def single_regression(xarray_kwargs=None) -> XarrayRegression:
    regression = SingleRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True)
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True)

    r = ((xm / normxm) * (ym / normym)).sum(axis=0)

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

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression_score.fit(source_train, target_train)
        score = self.regression_score.score(source_test, target_test)
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
