import numpy as np
from neuraxle.base import NonFittableMixin, BaseStep
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin


class MeanStdNormalizer(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        InputAndOutputTransformerMixin.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs
        mean = np.expand_dims(np.average(data_inputs, axis=1) + 0.00001, axis=1)
        stddev = np.expand_dims(np.std(data_inputs, axis=1) + 0.00001, axis=1)
        data_inputs = (data_inputs - mean) / (2.5 * stddev)

        if expected_outputs is not None:
            expected_outputs = (expected_outputs - mean) / (2.5 * stddev)

        return data_inputs, expected_outputs


class WindowTimeSeries(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    def __init__(self, window_size):
        BaseStep.__init__(self)
        InputAndOutputTransformerMixin.__init__(self)
        NonFittableMixin.__init__(self)
        self.window_size = window_size

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs

        new_data_inputs = []
        new_expected_outputs = []
        for i in range(len(data_inputs) - self.window_size * 2):
            new_data_inputs.append(data_inputs[i: i + self.window_size])
            new_expected_outputs.append(data_inputs[i + self.window_size: i + self.window_size * 2])

        return np.array(new_data_inputs), np.array(new_expected_outputs)
