import numpy as np
import xarray as xr
import pytest
from brainio.assemblies import NeuroidAssembly
from pytest import approx

from brainscore.metrics.regression import CrossRegressedCorrelation, CrossRegressedCorrelationBatched, \
    pls_regression, linear_regression, pearsonr_correlation, ridge_regression, linear_regression_pearsonr_batched


class TestCrossRegressedCorrelation:
    def test_small(self):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'image_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', ['some_region'] * 25)},
                                   dims=['presentation', 'neuroid'])
        metric = CrossRegressedCorrelation(regression=pls_regression(), correlation=pearsonr_correlation())
        score = metric(source=assembly, target=assembly)
        assert score.sel(aggregation='center') == approx(1, abs=.00001)


class TestRegression:
    @pytest.mark.parametrize('regression_ctr', [pls_regression, linear_regression, ridge_regression])
    def test_small(self, regression_ctr):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'image_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', [None] * 25)},
                                   dims=['presentation', 'neuroid'])
        regression = regression_ctr()
        regression.fit(source=assembly, target=assembly)
        prediction = regression.predict(source=assembly)
        assert all(prediction['image_id'] == assembly['image_id'])
        assert all(prediction['neuroid_id'] == assembly['neuroid_id'])


class TestCrossRegressedCorrelationBatched:
    def test(self):
        source = NeuroidAssembly(
            (np.arange(300 * 25) + np.random.standard_normal(300 * 25)).reshape((300, 25)),
            coords={'image_id': ('presentation', np.arange(300)),
                    'object_name': ('presentation', ['a', 'b', 'c'] * 100),
                    'neuroid_id': ('neuroid', np.arange(25)),
                    'region': ('neuroid', ['some_region'] * 25)},
            dims=['presentation', 'neuroid'])
        target_rep1 = source.copy(deep=True)
        target_rep1['repetition'] = ('presentation', [0] * source.sizes['presentation'])
        target_rep2 = source.copy(deep=True)
        target_rep2['repetition'] = ('presentation', [1] * source.sizes['presentation'])
        target = xr.concat([target_rep1, target_rep2], dim='presentation')

        metric = CrossRegressedCorrelationBatched(
            linear_regression_pearsonr_batched(regression_kwargs={'lr': 0.1})
        )
        score = metric(source=source, target=target)

        # Test convergence
        assert score.sel(aggregation='center') == approx(1, abs=.00001)

        # Test dimensions and shapes
        assert score.dims == ('aggregation',)
        assert score.shape == (2,)
        assert score.raw.dims == ('split', 'neuroid')
        assert score.raw.shape == (10, 25)
