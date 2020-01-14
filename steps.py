import numpy as np
from neuraxle.base import NonFittableMixin, BaseStep
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin


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
            new_expected_outputs.append(data_inputs[i + self.window_size_past: i + self.window_size_past + self.window_size_future])

        return np.array(new_data_inputs), np.array(new_expected_outputs)


class ToNumpy(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        InputAndOutputTransformerMixin.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs
        return np.array(data_inputs), np.array(expected_outputs)
