from typing import Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader

from brainio.assemblies import NeuroidAssembly, array_is_element, walk_coords
from brainscore.metrics import Score
from brainscore.utils.batched_regression import RegressionModelBatched
from brainscore.utils.batched_scoring import ScoringBatched


class Defaults:
    expected_dims = ('presentation', 'neuroid')
    stimulus_coord = 'image_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'


class XarrayRegression:
    """
    Adds alignment-checking, un- and re-packaging, and comparison functionality to a regression.
    """

    def __init__(self, regression, expected_dims=Defaults.expected_dims, neuroid_dim=Defaults.neuroid_dim,
                 neuroid_coord=Defaults.neuroid_coord, stimulus_coord=Defaults.stimulus_coord):
        self._regression = regression
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None

    def fit(self, source, target):
        source, target = self._align(source), self._align(target)
        stimulus_dim = self._expected_dims[0]
        source = source.isel({stimulus_dim: map_target_to_source(source, target, self._stimulus_coord)})

        self._regression.fit(source, target)

        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def predict(self, source):
        source = self._align(source)
        predicted_values = self._regression.predict(source)
        prediction = self._package_prediction(predicted_values, source=source)
        return prediction

    def _package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        dims = source.dims
        # if there is only one neuroid coordinate, it would get discarded and the dimension would be used as coordinate.
        # to avoid this, we can build the assembly first and then stack on the neuroid dimension.
        neuroid_level_dim = None
        if len(self._target_neuroid_values) == 1:  # extract single key: https://stackoverflow.com/a/20145927/2225200
            (neuroid_level_dim, _), = self._target_neuroid_values.items()
            dims = [dim if dim != self._neuroid_dim else neuroid_level_dim for dim in dims]
        for target_coord, target_value in self._target_neuroid_values.items():
            # this might overwrite values which is okay
            coords[target_coord] = (neuroid_level_dim or self._neuroid_dim), target_value
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=dims)
        if neuroid_level_dim:
            prediction = prediction.stack(**{self._neuroid_dim: [neuroid_level_dim]})

        return prediction

    def _align(self, assembly):
        assert set(assembly.dims) == set(self._expected_dims), \
            f'Expected {set(self._expected_dims)}, but got {set(assembly.dims)}'
        return assembly.transpose(*self._expected_dims)


class XarrayCorrelation:
    def __init__(self, correlation, correlation_coord=Defaults.stimulus_coord, neuroid_coord=Defaults.neuroid_coord):
        self._correlation = correlation
        self._correlation_coord = correlation_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target):
        # align
        prediction = prediction.sortby([self._correlation_coord, self._neuroid_coord])
        target = target.sortby([self._correlation_coord, self._neuroid_coord])
        assert np.array(prediction[self._correlation_coord].values == target[self._correlation_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()
        # compute correlation per neuroid
        neuroid_dims = target[self._neuroid_coord].dims
        assert len(neuroid_dims) == 1
        correlations = []
        for i, coord_value in enumerate(target[self._neuroid_coord].values):
            target_neuroids = target.isel(**{neuroid_dims[0]: i})  # `isel` is about 10x faster than `sel`
            prediction_neuroids = prediction.isel(**{neuroid_dims[0]: i})
            r, p = self._correlation(target_neuroids, prediction_neuroids)
            correlations.append(r)
        # package
        result = Score(correlations,
                       coords={coord: (dims, values)
                               for coord, dims, values in walk_coords(target) if dims == neuroid_dims},
                       dims=neuroid_dims)
        return result


#############################################################################
##################### Batched regression/correlation ########################
#############################################################################


class XarrayRegressionScoreBatched:

    def __init__(self,
                 regression: RegressionModelBatched,
                 scoring: ScoringBatched,
                 batch_size: int = 2048,
                 eval_batch_size: int = 2048,
                 max_epochs: int = 100,
                 shuffle=True,
                 expected_dims=Defaults.expected_dims,
                 neuroid_dim=Defaults.neuroid_dim,
                 stimulus_coord=Defaults.stimulus_coord,
                 random_seed: Optional[int] = None):
        # todo: add support for stopping based on a patience parameter tracking the progression of training losses
        self._regression = regression
        self._scoring = scoring
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._max_epochs = max_epochs
        self._shuffle = shuffle
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._stimulus_coord = stimulus_coord
        self._stimulus_dim = self._expected_dims[0]
        self._random_seed = random_seed

        self._fitted = False
        self._epoch_training_loss = None
        self._target_neuroid_values = None

    def fit(self, source: NeuroidAssembly, target: NeuroidAssembly) -> None:
        self._check_dims(source), self._check_dims(target)
        target_to_source_idx = map_target_to_source(source, target, self._stimulus_coord)

        if self._shuffle and self._random_seed is not None:
            np.random.seed(self._random_seed)

        epoch_training_loss = []
        indices = np.arange(len(target[self._stimulus_coord]))
        for _ in range(self._max_epochs):
            if self._shuffle:
                np.random.shuffle(indices)

            batch_losses = []
            for i in range(0, len(indices), self._batch_size):
                target_batch_indices = indices[i:i + self._batch_size]
                source_batch_indices = target_to_source_idx[target_batch_indices]
                target_batch = target.isel({self._stimulus_dim: target_batch_indices})
                source_batch = source.isel({self._stimulus_dim: source_batch_indices})

                batch_loss = self._regression.fit_partial(source_batch.values, target_batch.values)
                batch_losses.append(batch_loss)

            batch_losses = np.array(batch_losses)
            epoch_training_loss.append(batch_losses.mean())

        self._fitted = True
        self._epoch_training_loss = np.array(epoch_training_loss)
        self._target_neuroid_values = target[self._neuroid_dim]

    def predict(self, source: NeuroidAssembly) -> NeuroidAssembly:
        self._check_dims(source)
        preds = self._regression.predict(source.values)
        preds = NeuroidAssembly(preds,
                                dims=self._expected_dims,
                                coords={
                                    **{coord: (dims, values)
                                       for coord, dims, values in walk_coords(source[self._stimulus_dim])},
                                    **{coord: (dims, values)
                                       for coord, dims, values in walk_coords(self._target_neuroid_values)}
                                })
        return preds


    def score(self, source: NeuroidAssembly, target: NeuroidAssembly) -> Score:
        self._check_dims(source), self._check_dims(target)
        self._check_target_alignment(target)
        target_to_source_idx = map_target_to_source(source, target, self._stimulus_coord)

        self._scoring.reset()
        for i in range(0, source.sizes[self._stimulus_dim], self._eval_batch_size):
            source_batch_indices = target_to_source_idx[i:i + self._eval_batch_size]
            target_batch = target.isel({self._stimulus_dim: slice(i, i + self._eval_batch_size)})
            source_batch = source.isel({self._stimulus_dim: source_batch_indices})

            preds = self._regression.predict(source_batch.values)
            self._scoring.update(preds, target_batch.values)

        scores = self._scoring.compute()
        scores = Score(scores,
                       coords={coord: (dims, values)
                               for coord, dims, values in walk_coords(target) if dims == (self._neuroid_dim,)},
                       dims=[self._neuroid_dim])

        return scores

    @property
    def regression(self):
        return self._regression

    @property
    def epoch_training_loss(self):
        assert self._fitted
        return self._epoch_training_loss

    def _check_dims(self, assembly: NeuroidAssembly) -> None:
        # Don't do any transposes if the assembly isn't aligned to the expected dims,
        # because lazily-loaded assemblies would be entirely loaded into memory.
        # Just throw an error instead.
        assert assembly.dims == self._expected_dims, \
            f'Expected {self._expected_dims}, but got {assembly.dims}'
        stimulus_dim = assembly[self._stimulus_coord].dims
        assert len(stimulus_dim) == 1 and stimulus_dim[0] == self._stimulus_dim, \
            f'Expected stimulus coord {self._stimulus_coord} to be along ' \
            f'the first dimension in expected dims {self._expected_dims}, ' \
            f'but it was along {stimulus_dim}'

    def _check_stimulus_alignment(self, source: NeuroidAssembly, target: NeuroidAssembly) -> None:
        # Don't re-order assemblies if they are not aligned along their stimulus_coord,
        # because lazily-loaded assemblies would be entirely loaded into memory.
        # Just throw an error instead.
        assert source.dims == target.dims and \
               (source[self._stimulus_coord].values == target[self._stimulus_coord].values).all(), \
            f'Source and target data assemblies do not align along their ' \
            f'stimulus coordinate: {self._stimulus_coord}'

    def _check_target_alignment(self, target: NeuroidAssembly) -> None:
        # Make sure the target's neuroid coordinates align with those from training.
        # Don't re-order them otherwise, because lazily-loaded assemblies would be
        # entirely loaded in memory. Just throw an error instead.
        assert (target[self._neuroid_dim].values == self._target_neuroid_values.values).all()


def map_target_to_source(source: NeuroidAssembly, target: NeuroidAssembly, stimulus_coord: str) -> np.ndarray:
    assert len(np.unique(source[stimulus_coord])) == len(
        source[stimulus_coord]
    ), f'Source assembly has duplicate samples along the {stimulus_coord} coordinate'
    assert np.all(
        np.isin(target[stimulus_coord], source[stimulus_coord])
    ), 'Not all targets have corresponding sources'

    index_map = []
    for target_sample in target[stimulus_coord]:
        source_index = np.where(source[stimulus_coord] == target_sample)
        source_index = source_index[0].item()
        index_map.append(source_index)
    index_map = np.array(index_map)

    return index_map
