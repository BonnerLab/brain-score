from typing import Callable, Optional

import numpy as np

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
        self._stimulus_dim = self._expected_dims[0]
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None

    def fit(self, source, target):
        source, target = self._align(source), self._align(target)
        source = source.isel({self._stimulus_dim: map_target_to_source(source, target, self._stimulus_dim)})
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
    def __init__(
        self,
        correlation: Callable,
        correlation_coord: str = Defaults.stimulus_coord,
        neuroid_coord: str = Defaults.neuroid_coord,
        parallel: bool = False,
    ):
        self._correlation = correlation
        self._correlation_coord = correlation_coord
        self._neuroid_coord = neuroid_coord
        self._parallel = parallel

    def __call__(self, prediction: NeuroidAssembly, target: NeuroidAssembly) -> Score:
        """Compute the correlation metric on the prediction and target assemblies"""
        try:
            self._check_assemblies(prediction, target)
        except AssertionError:
            prediction = prediction.sortby([self._correlation_coord, self._neuroid_coord])
            target = target.sortby([self._correlation_coord, self._neuroid_coord])
            self._check_assemblies(prediction, target)

        neuroid_dims = target[self._neuroid_coord].dims
        if self._parallel:
            correlations = self._correlation(prediction, target)
        else:
            correlations = [
                self._correlation(  # `isel` is about 10x faster than `sel`
                    target.isel({neuroid_dims[0]: i_neuroid}),
                    prediction.isel({neuroid_dims[0]: i_neuroid}),
                )[0]  # extract only r values, not p
                for i_neuroid in range(len(target[self._neuroid_coord]))
            ]
        # package
        return Score(
            correlations,
            coords={
                coord: (dims, values)
                for coord, dims, values in walk_coords(target) if dims == neuroid_dims
            },
            dims=neuroid_dims,
        )

    def _check_assemblies(self, prediction: NeuroidAssembly, target: NeuroidAssembly) -> None:
        """Check for alignment of prediction and target assemblies."""
        assert len(target[self._neuroid_coord].dims) == 1
        assert len(target[self._correlation_coord].dims) == 1
        assert np.array(prediction[self._correlation_coord].values == target[self._correlation_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()


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
        target_to_source_idx = map_target_to_source(source, target, self._stimulus_dim)

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
        target_to_source_idx = map_target_to_source(source, target, self._stimulus_dim)

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


def map_target_to_source(source: NeuroidAssembly, target: NeuroidAssembly, stimulus_dim: str) -> np.ndarray:
    """Generate the indices along stimulus_dim in the source assembly that correspond to the same samples in the target assembly."""

    # align the source and target according to all their shared levels along the stimulus_dimension
    if stimulus_dim in source.indexes:
        source = source.reset_index(stimulus_dim)
    if stimulus_dim in target.indexes:
        target = target.reset_index(stimulus_dim)
    unshared_levels = list(set(source[stimulus_dim].coords) ^ set(target[stimulus_dim].coords))
    shared_levels = list(set(source[stimulus_dim].coords) & set(target[stimulus_dim].coords))
    # FIXME If there's no `time_bin`` dimension, `time_bin` is automatically treated as a coord along all existing dimensions.
    # This causes it to be present in `shared_levels` and messes up setting the index because it contains a non-hashable numpy array.
    # Right now, I'm just dropping it if the data is 2D (no temporal dim) but I have no idea how it'll behave with temporal data.
    if "time_bin" not in source.dims and "time_bin" in source.coords:
        source = source.drop_vars("time_bin")
        if "time_bin" in target.coords:
            target = target.drop_vars("time_bin")
        if "time_bin" in shared_levels:
            shared_levels.remove("time_bin")
    source = source.drop_vars(unshared_levels, errors="ignore").set_index({stimulus_dim: shared_levels})
    target = target.drop_vars(unshared_levels, errors="ignore").set_index({stimulus_dim: shared_levels})
    assert source.indexes[stimulus_dim].names == target.indexes[stimulus_dim].names, "source and target don't have levels in the same order along the stimulus dimension"

    # if the assemblies are already aligned, don't worry about duplicate samples in source
    if np.all(source[stimulus_dim].values == target[stimulus_dim].values):
        return np.arange(source.sizes[stimulus_dim])

    assert len(np.unique(source[stimulus_dim])) == len(
        source[stimulus_dim]
    ), f'Source assembly has duplicate samples along the {stimulus_dim} dimension'

    # tuples of immutables don't compare equal even when their hashes are equal, for some reason, so explicitly use hash to create dict
    source_val_to_index = {hash(val): idx for idx, val in enumerate(source.indexes[stimulus_dim].values)}
    try:
        index_map = np.array([source_val_to_index[hash(target_sample)]
                              for target_sample in target[stimulus_dim].values])
    except KeyError as e:
        print('Not all targets have corresponding sources')
        raise e

    return index_map
