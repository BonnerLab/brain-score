import numpy as np
import xarray as xr
import scipy.stats
from pytest import approx
from sklearn.linear_model import LinearRegression

from brainio.assemblies import NeuroidAssembly
from brainscore.utils.batched_regression import LinearRegressionBatched
from brainscore.utils.batched_scoring import PearsonrScoringBatched
from brainscore.metrics.xarray_utils import XarrayRegression, XarrayCorrelation, XarrayRegressionScoreBatched


class TestXarrayRegression:
    def test_fitpredict_alignment(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10),
                                         coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10)},
                                         dims=['presentation', 'neuroid'])
        target = jumbled_source.sortby(['image_id', 'neuroid_id'])
        regression = XarrayRegression(LinearRegression())
        regression.fit(jumbled_source, target)
        prediction = regression.predict(jumbled_source)
        # do not test for alignment of metadata - it is only important that the data is well-aligned with the metadata.
        np.testing.assert_array_almost_equal(prediction.sortby(['image_id', 'neuroid_id']).values,
                                             target.sortby(['image_id', 'neuroid_id']).values)

    def test_neuroid_single_coord(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10),
                                         coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid_id', list(reversed(range(10))))},
                                         dims=['presentation', 'neuroid_id']).stack(neuroid=['neuroid_id'])
        target = jumbled_source.sortby(['image_id', 'neuroid_id'])
        regression = XarrayRegression(LinearRegression())
        regression.fit(jumbled_source, target)
        prediction = regression.predict(jumbled_source)
        assert set(prediction.dims) == {'presentation', 'neuroid'}
        assert len(prediction['neuroid_id']) == 10


class TestXarrayCorrelation:
    def test_dimensions(self):
        prediction = NeuroidAssembly(np.random.rand(500, 10),
                                     coords={'image_id': ('presentation', list(range(500))),
                                             'image_meta': ('presentation', [0] * 500),
                                             'neuroid_id': ('neuroid', list(range(10))),
                                             'neuroid_meta': ('neuroid', [0] * 10)},
                                     dims=['presentation', 'neuroid'])
        correlation = XarrayCorrelation(lambda a, b: (1, 0))
        score = correlation(prediction, prediction)
        np.testing.assert_array_equal(score.dims, ['neuroid'])
        np.testing.assert_array_equal(score.shape, [10])

    def test_correlation(self):
        prediction = NeuroidAssembly(np.random.rand(500, 10),
                                     coords={'image_id': ('presentation', list(range(500))),
                                             'image_meta': ('presentation', [0] * 500),
                                             'neuroid_id': ('neuroid', list(range(10))),
                                             'neuroid_meta': ('neuroid', [0] * 10)},
                                     dims=['presentation', 'neuroid'])
        correlation = XarrayCorrelation(lambda a, b: (1, 0))
        score = correlation(prediction, prediction)
        assert all(score == approx(1))

    def test_alignment(self):
        jumbled_prediction = NeuroidAssembly(np.random.rand(500, 10),
                                             coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                     'image_meta': ('presentation', [0] * 500),
                                                     'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                     'neuroid_meta': ('neuroid', [0] * 10)},
                                             dims=['presentation', 'neuroid'])
        prediction = jumbled_prediction.sortby(['image_id', 'neuroid_id'])
        correlation = XarrayCorrelation(scipy.stats.pearsonr)
        score = correlation(jumbled_prediction, prediction)
        assert all(score == approx(1))

    def test_neuroid_single_coord(self):
        prediction = NeuroidAssembly(np.random.rand(500, 10),
                                     coords={'image_id': ('presentation', list(range(500))),
                                             'image_meta': ('presentation', [0] * 500),
                                             'neuroid_id': ('neuroid_id', list(range(10)))},
                                     dims=['presentation', 'neuroid_id']).stack(neuroid=['neuroid_id'])
        correlation = XarrayCorrelation(lambda a, b: (1, 0))
        score = correlation(prediction, prediction)
        np.testing.assert_array_equal(score.dims, ['neuroid'])
        assert len(score['neuroid']) == 10


class TestXarrayRegressionScoreBatched:
    def test_fit_convergence(self):
        jumbled_source, target = self.get_jumbled_data()
        regression_score = self.get_regression_scoring(regression_kwargs={'lr': 0.1})
        regression_score.fit(jumbled_source, target)
        prediction = regression_score.predict(jumbled_source)

        np.testing.assert_array_almost_equal(prediction.sortby(['image_id', 'neuroid_id']).values,
                                             target.sortby(['image_id', 'neuroid_id']).values)

    def test_fitpredict_alignment(self):
        jumbled_source, target = self.get_jumbled_data()
        regression_score = self.get_regression_scoring(regression_kwargs={'lr': 0.1})
        regression_score.fit(jumbled_source, target)
        prediction = regression_score.predict(jumbled_source)

        assert prediction.dims == target.dims
        assert (prediction['presentation'] == jumbled_source['presentation']).all()
        assert (prediction['neuroid'] == target['neuroid']).all()

    def test_correlation_convergence(self):
        jumbled_source, target = self.get_jumbled_data()
        regression_score = self.get_regression_scoring(regression_kwargs={'lr': 0.1})
        regression_score.fit(jumbled_source, target)
        score = regression_score.score(jumbled_source, target)

        np.testing.assert_allclose(score.values, 1.0, atol=0.01)

    def test_correlation_alignment(self):
        jumbled_source, target = self.get_jumbled_data()
        regression_score = self.get_regression_scoring(regression_kwargs={'lr': 0.1})
        regression_score.fit(jumbled_source, target)
        score = regression_score.score(jumbled_source, target)

        assert score.dims == ('neuroid',)
        assert (score['neuroid'] == target['neuroid']).all()

    def test_duplicate_targets(self):
        jumbled_source, target = self.get_jumbled_data()
        target_rep1 = target
        target_rep1['repetition'] = ('presentation', [0] * target_rep1.sizes['presentation'])
        target_rep2 = target.isel(presentation=slice(0, 10))
        target_rep2['repetition'] = ('presentation', [1] * target_rep2.sizes['presentation'])
        target = xr.concat([target_rep1, target_rep2], dim='presentation')
        regression_score = self.get_regression_scoring(regression_kwargs={'lr': 0.1})
        regression_score.fit(jumbled_source, target)
        score = regression_score.score(jumbled_source, target)

        np.testing.assert_allclose(score.values, 1.0, atol=0.01)

    def get_jumbled_data(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10),
                                         coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10)},
                                         dims=['presentation', 'neuroid'])
        target = jumbled_source.sortby(['image_id', 'neuroid_id'])
        return jumbled_source, target


    def get_regression_scoring(self, regression_kwargs=None, scoring_kwargs=None, **kwargs):
        if regression_kwargs is None:
            regression_kwargs = {}
        if scoring_kwargs is None:
            scoring_kwargs = {}
        return XarrayRegressionScoreBatched(regression=LinearRegressionBatched(**regression_kwargs),
                                            scoring=PearsonrScoringBatched(**scoring_kwargs),
                                            batch_size=50,
                                            eval_batch_size=100,
                                            **kwargs)
