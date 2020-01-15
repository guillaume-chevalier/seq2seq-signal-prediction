from typing import Callable

import numpy as np
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
from tensorflow_core.python.training.rmsprop import RMSPropOptimizer

from data_loading import generate_data
from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from plotting import plot_metrics
from steps import MeanStdNormalizer, ToNumpy, PlotPredictionsWrapper


def create_model(step: Tensorflow2ModelStep):
    # shape: (batch_size, seq_length, input_dim)
    encoder_inputs = Input(shape=(None, step.hyperparams['input_dim']), dtype=tf.dtypes.float32)
    # TODO: why is this 2D here whereas the comment above is 3D? I'd have expected a 3D Input placeholder.
    #       This needs an explanation or at least I need to know to explain it.
    # TODO: this might be a bug

    # shape: (batch_size, seq_length, output_dim)
    decoder_inputs = Input(shape=(None, step.hyperparams['output_dim']), dtype=tf.dtypes.float32)

    last_encoder_outputs, last_encoders_states = create_encoder(step, encoder_inputs)
    decoder_outputs = create_decoder(step, last_encoder_outputs, last_encoders_states)

    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def create_encoder(step: Tensorflow2ModelStep, encoder_inputs):
    encoder = RNN(create_stacked_rnn_cells(step), return_sequences=False, return_state=True)
    last_encoder_outputs_and_states = encoder(encoder_inputs)

    last_encoder_outputs, *last_encoders_states = last_encoder_outputs_and_states
    return last_encoder_outputs, last_encoders_states


def create_decoder(step: Tensorflow2ModelStep, last_encoder_outputs, last_encoders_states):
    decoder_lstm = RNN(create_stacked_rnn_cells(step), return_sequences=True, return_state=False)

    last_encoder_output = tf.expand_dims(last_encoder_outputs, axis=1)
    replicated_last_encoder_output = tf.repeat(input=last_encoder_output,
                                               repeats=step.hyperparams['window_size_future'], axis=1)
    decoder_outputs = decoder_lstm(replicated_last_encoder_output, initial_state=last_encoders_states)

    decoder_dense = Dense(step.hyperparams['output_dim'])

    return decoder_dense(decoder_outputs)


def create_stacked_rnn_cells(step: Tensorflow2ModelStep):
    cells = []
    for _ in range(step.hyperparams['layers_stacked_count']):
        cells.append(GRUCell(step.hyperparams['hidden_dim']))

    return cells


def create_loss(step: Tensorflow2ModelStep, expected_outputs, predicted_outputs):
    l2 = step.hyperparams['lambda_loss_amount'] * sum(
        tf.reduce_mean(tf.nn.l2_loss(tf_var))
        for tf_var in step.model.trainable_variables
    )

    output_loss = sum(
        tf.reduce_mean(tf.nn.l2_loss(pred - expected))
        for pred, expected in zip(predicted_outputs, expected_outputs)
    ) / float(len(predicted_outputs))

    return output_loss + l2


def create_optimizer(step: TensorflowV1ModelStep):
    return RMSPropOptimizer(
        learning_rate=step.hyperparams['learning_rate'],
        decay=step.hyperparams['lr_decay'],
        momentum=step.hyperparams['momentum']
    )


seq2seq_pipeline_hyperparams = HyperparameterSamples({
    'hidden_dim': 32,
    'layers_stacked_count': 2,
    'lambda_loss_amount': 0.003,
    'learning_rate': 0.006,
    'lr_decay': 0.92,
    'momentum': 0.5,
    'window_size_future': 40,
    'output_dim': 2,
    'input_dim': 2
})


def metric_2d_to_3d_wrapper(metric_fun: Callable):
    def metric(data_inputs, expected_outputs):
        return metric_fun(np.array(data_inputs)[..., 0], np.array(expected_outputs)[..., 0])

    return metric


def main():
    exercice_number = 1

    data_inputs, expected_outputs = generate_data(exercice_number=exercice_number)

    print('exercice {}\n=================='.format(1))
    tf.debugging.set_log_device_placement(True)
    print('You can use the following devices: {}'.format(get_avaible_devices()))

    print('data_inputs shape: {} => (batch_size, sequence_length, input_dim)'.format(data_inputs.shape))
    print('expected_outputs shape: {} => (batch_size, sequence_length, input_dim)'.format(expected_outputs.shape))

    sequence_length = data_inputs.shape[1]
    input_dim = data_inputs.shape[2]
    output_dim = expected_outputs.shape[2]

    batch_size = 10
    epochs = 50
    validation_size = 0.15

    metrics = {'mse': metric_2d_to_3d_wrapper(mean_squared_error)}

    signal_prediction_pipeline = Pipeline([
        ForEachDataInput(MeanStdNormalizer()),
        ToNumpy(),
        PlotPredictionsWrapper(Tensorflow2ModelStep(
            create_model=create_model,
            create_loss=create_loss,
            create_optimizer=create_optimizer,
            expected_outputs_dtype=tf.dtypes.float32,
            data_inputs_dtype=tf.dtypes.float32,
            print_loss=True,
            # device_name='/device:XLA_GPU:0'
        ).set_hyperparams(seq2seq_pipeline_hyperparams).update_hyperparams(HyperparameterSamples({
            'window_size_future': sequence_length,
            'input_dim': input_dim,
            'output_dim': output_dim
        })))
    ]).set_name('SignalPrediction')

    pipeline = Pipeline([EpochRepeater(
        ValidationSplitWrapper(
            MetricsWrapper(
                Pipeline([
                    TrainOnlyWrapper(DataShuffler()),
                    MiniBatchSequentialPipeline([
                        MetricsWrapper(
                            signal_prediction_pipeline,
                            metrics=metrics,
                            name='batch_metrics'
                        )
                    ], batch_size=batch_size)
                ]), metrics=metrics, name='epoch_metrics'),
            test_size=validation_size,
            scoring_function=metric_2d_to_3d_wrapper(mean_squared_error),
        ), epochs=epochs, fit_only=False)])

    pipeline, outputs = pipeline.fit_transform(data_inputs, expected_outputs)

    plot_metrics(pipeline=pipeline, exercice_number=exercice_number)
    plot_predictions(data_inputs, expected_outputs, pipeline)


def plot_predictions(data_inputs, expected_outputs, pipeline):
    _, _, data_inputs_validation, expected_outputs_validation = \
        pipeline.get_step_by_name('ValidationSplitWrapper').split(data_inputs, expected_outputs)

    signal_prediction_pipeline = pipeline.get_step_by_name('SignalPrediction')
    signal_prediction_pipeline.apply('toggle_plotting')
    signal_prediction_pipeline.apply('set_max_plotted_predictions', 10)

    signal_prediction_pipeline.transform_data_container(DataContainer(
        data_inputs=data_inputs_validation,
        expected_outputs=expected_outputs_validation
    ))


def get_avaible_devices():
    return [x.name for x in device_lib.list_local_devices()]


if __name__ == '__main__':
    main()
    # TODO: how to import external code files: https://stackoverflow.com/questions/48905127/importing-py-files-in-google-colab
