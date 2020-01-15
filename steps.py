from typing import Callable

import numpy as np
from neuraxle.base import NonFittableMixin, BaseStep, Identity, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin
from neuraxle.union import FeatureUnion

from plotting import plot_predictions


class MeanStdNormalizer(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        InputAndOutputTransformerMixin.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        di, eo = data_inputs
        mean = np.mean(di, axis=0) + 0.00001
        stddev = np.std(di, axis=0) + 0.00001
        di = (di - mean) / (2.5 * stddev)

        if eo is not None:
            eo = (eo - mean) / (2.5 * stddev)

        return di, eo


class WindowTimeSeries(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    def __init__(self, window_size_past, window_size_future):
        BaseStep.__init__(self)
        InputAndOutputTransformerMixin.__init__(self)
        NonFittableMixin.__init__(self)
        self.window_size_past = window_size_past
        self.window_size_future = window_size_future

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs

        new_data_inputs = []
        new_expected_outputs = []
        for i in range(len(data_inputs) - self.window_size_past - self.window_size_future):
            new_data_inputs.append(data_inputs[i: i + self.window_size_past])
            new_expected_outputs.append(
                data_inputs[i + self.window_size_past: i + self.window_size_past + self.window_size_future])

        return np.array(new_data_inputs), np.array(new_expected_outputs)


class ToNumpy(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        InputAndOutputTransformerMixin.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs
        return np.array(data_inputs), np.array(expected_outputs)


class PlotPredictionsWrapper(FeatureUnion):
    def __init__(self, wrapped, max_plotted_predictions=None):
        if max_plotted_predictions is None:
            max_plotted_predictions = 10

        FeatureUnion.__init__(self, [
            Identity(),
            wrapped
        ], joiner=PlotPredictionsJoiner(plot_predictions, max_plotted_predictions), n_jobs=1)


class PlotPredictionsJoiner(NonFittableMixin, BaseStep):
    def __init__(self, plotting_function: Callable, max_plotted_predictions, enabled=False):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(self)
        self.max_plotted_predictions = max_plotted_predictions
        self.enabled = enabled
        self.plotting_function = plotting_function

    def set_max_plotted_predictions(self, max_plotted_predictions):
        self.max_plotted_predictions = max_plotted_predictions

    def toggle_plotting(self):
        self.enabled = not self.enabled

    def transform(self, data_inputs):
        raise NotImplementedError('must be used inside a pipeline')

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        past = data_container.data_inputs[0].data_inputs
        predicted = data_container.data_inputs[1].data_inputs
        expected = data_container.expected_outputs

        if self.enabled:
            self._plot_predictions(expected, past, predicted)

        data_container.set_data_inputs(predicted)

        return data_container

    def _plot_predictions(self, expected, past, predicted):
        i = 0
        for past_sequence, expected_sequence, predicted_sequence in zip(past, expected, predicted):
            if i > self.max_plotted_predictions:
                break

            self.plotting_function(past_sequence, expected_sequence, predicted_sequence)
            i += 1
