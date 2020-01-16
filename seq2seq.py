from logging import warning
from typing import List

import tensorflow as tf
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.metaopt.random import ValidationSplitWrapper
from neuraxle.metrics import MetricsWrapper
from neuraxle.pipeline import Pipeline, MiniBatchSequentialPipeline
from neuraxle.steps.data import EpochRepeater, DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.loop import ForEachDataInput
from sklearn.metrics import mean_squared_error
from tensorflow_core.python.client import device_lib
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import GRUCell, RNN, Dense
from tensorflow_core.python.training.adam import AdamOptimizer

from datasets import generate_data
from datasets import metric_3d_to_2d_wrapper
from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from plotting import plot_metrics
from steps import MeanStdNormalizer, ToNumpy, PlotPredictionsWrapper


def create_model(step: Tensorflow2ModelStep) -> tf.keras.Model:
    """
    Create a TensorFlow v2 sequence to sequence (seq2seq) encoder-decoder model.

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :return: TensorFlow v2 Keras model
    """
    # shape: (batch_size, seq_length, input_dim)
    encoder_inputs = Input(
        shape=(None, step.hyperparams['input_dim']),
        batch_size=None,
        dtype=tf.dtypes.float32,
        name='encoder_inputs'
    )

    last_encoder_outputs, last_encoders_states = _create_encoder(step, encoder_inputs)
    decoder_outputs = _create_decoder(step, last_encoder_outputs, last_encoders_states)

    return Model(encoder_inputs, decoder_outputs)


def _create_encoder(step: Tensorflow2ModelStep, encoder_inputs: Input) -> (tf.Tensor, List[tf.Tensor]):
    """
    Create an encoder RNN using GRU Cells. GRU cells are similar to LSTM cells.

    :param step: The base Neuraxle step for TensorFlow v2 (class Tensorflow2ModelStep)
    :param encoder_inputs: encoder inputs layer of shape (batch_size, seq_length, input_dim)
    :return: (last encoder outputs, last stacked encoders states)
                last_encoder_outputs shape: (batch_size, hidden_dim)
                last_encoder_states shape: (layers_stacked_count, batch_size, hidden_dim)
    """
    encoder = RNN(cell=_create_stacked_rnn_cells(step), return_sequences=False, return_state=True)

    last_encoder_outputs_and_states = encoder(encoder_inputs)
    # last_encoder_outputs shape: (batch_size, hidden_dim)
    # last_encoder_states shape: (layers_stacked_count, batch_size, hidden_dim)

    # refer to: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable#output_shape_2
    last_encoder_outputs, *last_encoders_states = last_encoder_outputs_and_states
    return last_encoder_outputs, last_encoders_states


def _create_decoder(
        step: Tensorflow2ModelStep, last_encoder_outputs: tf.Tensor,last_encoders_states: List[tf.Tensor]
) -> tf.Tensor:
    """
    Create a decoder RNN using GRU cells.

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :param last_encoders_states: last encoder states tensor
    :param last_encoder_outputs: last encoder output tensor
    :return: decoder output
    """
    decoder_lstm = RNN(cell=_create_stacked_rnn_cells(step), return_sequences=True, return_state=False)

    last_encoder_output = tf.expand_dims(last_encoder_outputs, axis=1)
    # last encoder output shape: (batch_size, 1, hidden_dim)

    replicated_last_encoder_output = tf.repeat(
        input=last_encoder_output,
        repeats=step.hyperparams['window_size_future'],
        axis=1
    )
    # replicated last encoder output shape: (batch_size, window_size_future, hidden_dim)

    decoder_outputs = decoder_lstm(replicated_last_encoder_output, initial_state=last_encoders_states)
    # decoder outputs shape: (batch_size, window_size_future, hidden_dim)

    decoder_dense = Dense(step.hyperparams['output_dim'])
    # decoder outputs shape: (batch_size, window_size_future, output_dim)

    return decoder_dense(decoder_outputs)


def _create_stacked_rnn_cells(step: Tensorflow2ModelStep) -> List[GRUCell]:
    """
    Create a `layers_stacked_count` amount of GRU cells and stack them on top of each other.
    They have a `hidden_dim` number of neuron layer size.

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :return: list of gru cells
    """
    cells = []
    for _ in range(step.hyperparams['layers_stacked_count']):
        cells.append(GRUCell(step.hyperparams['hidden_dim']))

    return cells


def create_loss(step: Tensorflow2ModelStep, expected_outputs: tf.Tensor, predicted_outputs: tf.Tensor) -> tf.Tensor:
    """
    Create model loss.

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :param expected_outputs: expected outputs of shape (batch_size, window_size_future, output_dim)
    :param predicted_outputs: expected outputs of shape (batch_size, window_size_future, output_dim)
    :return: loss (a tf Tensor that is a float)
    """
    l2 = step.hyperparams['lambda_loss_amount'] * sum(
        tf.reduce_mean(tf.nn.l2_loss(tf_var))
        for tf_var in step.model.trainable_variables
    )

    output_loss = sum(
        tf.reduce_mean(tf.nn.l2_loss(pred - expected))
        for pred, expected in zip(predicted_outputs, expected_outputs)
    ) / float(len(predicted_outputs))

    return output_loss + l2


def create_optimizer(step: TensorflowV1ModelStep) -> AdamOptimizer:
    """
    Create a TensorFlow 2 Optimizer: here the AdamOptimizer.

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :return: optimizer
    """
    return AdamOptimizer(learning_rate=step.hyperparams['learning_rate'])


def main(chosen_device):
    exercice_number = 1
    print('exercice {}\n=================='.format(exercice_number))

    data_inputs, expected_outputs = generate_data(
        # See: https://github.com/guillaume-chevalier/seq2seq-signal-prediction/blob/master/datasets.py
        exercice_number=exercice_number,
        n_samples=None,
        window_size_past=None,
        window_size_future=None
    )

    print('data_inputs shape: {} => (n_samples, window_size_past, input_dim)'.format(data_inputs.shape))
    print('expected_outputs shape: {} => (n_samples, window_size_future, output_dim)'.format(expected_outputs.shape))

    sequence_length = data_inputs.shape[1]
    input_dim = data_inputs.shape[2]
    output_dim = expected_outputs.shape[2]

    batch_size = 100
    epochs = 3
    validation_size = 0.15
    max_plotted_validation_predictions = 10

    seq2seq_pipeline_hyperparams = HyperparameterSamples({
        'hidden_dim': 100,
        'layers_stacked_count': 2,
        'lambda_loss_amount': 0.0003,
        'learning_rate': 0.006,
        'window_size_future': sequence_length,
        'output_dim': output_dim,
        'input_dim': input_dim
    })
    feature_0_metric = metric_3d_to_2d_wrapper(mean_squared_error)
    metrics = {'mse': feature_0_metric}

    signal_prediction_pipeline = Pipeline([
        ForEachDataInput(MeanStdNormalizer()),
        ToNumpy(),
        PlotPredictionsWrapper(Tensorflow2ModelStep(
            # See: https://github.com/Neuraxio/Neuraxle-TensorFlow
            create_model=create_model,
            create_loss=create_loss,
            create_optimizer=create_optimizer,
            expected_outputs_dtype=tf.dtypes.float32,
            data_inputs_dtype=tf.dtypes.float32,
            print_loss=True
        ).set_hyperparams(seq2seq_pipeline_hyperparams))
    ]).set_name('SignalPrediction')

    pipeline = Pipeline([EpochRepeater(
        ValidationSplitWrapper(
            MetricsWrapper(Pipeline([
                TrainOnlyWrapper(DataShuffler()),
                MiniBatchSequentialPipeline([
                    MetricsWrapper(
                        signal_prediction_pipeline,
                        metrics=metrics,
                        name='batch_metrics'
                    )
                ], batch_size=batch_size)
            ]), metrics=metrics,
                name='epoch_metrics',
                print_metrics=True
            ),
            test_size=validation_size,
            scoring_function=feature_0_metric
        ), epochs=epochs)])

    pipeline, outputs = pipeline.fit_transform(data_inputs, expected_outputs)

    plot_metrics(pipeline=pipeline, exercice_number=exercice_number)
    plot_predictions(data_inputs, expected_outputs, pipeline, max_plotted_validation_predictions)


def plot_predictions(data_inputs, expected_outputs, pipeline, max_plotted_predictions):
    _, _, data_inputs_validation, expected_outputs_validation = \
        pipeline.get_step_by_name('ValidationSplitWrapper').split(data_inputs, expected_outputs)

    pipeline.apply('toggle_plotting')
    pipeline.apply('set_max_plotted_predictions', max_plotted_predictions)

    signal_prediction_pipeline = pipeline.get_step_by_name('SignalPrediction')
    signal_prediction_pipeline.transform_data_container(DataContainer(
        data_inputs=data_inputs_validation,
        expected_outputs=expected_outputs_validation
    ))


def choose_tf_device():
    """
    Choose a TensorFlow device (e.g.: GPU if available) to compute on.
    """
    tf.debugging.set_log_device_placement(True)
    devices = [x.name for x in device_lib.list_local_devices()]
    print('You can use the following tf devices: {}'.format(devices))
    try:
        chosen_device = [d for d in devices if 'gpu' in d.lower()][0]
    except:
        warning(
            "No GPU device found. Please make sure to do `Runtime > Change Runtime Type` and select GPU for Python 3.")
        chosen_device = devices[0]
    print('Chosen Device: {}'.format(chosen_device))
    return chosen_device


if __name__ == '__main__':
    chosen_device = choose_tf_device()
    main(chosen_device)
