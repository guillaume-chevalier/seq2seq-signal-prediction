import math
from typing import Callable

import numpy as np
import tensorflow as tf
from neuraxle.api import DeepLearningPipeline
from neuraxle.base import ExecutionContext, DEFAULT_CACHE_FOLDER
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEachDataInput
from sklearn.metrics import mean_squared_error
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import GRUCell, RNN, Dense
from tensorflow_core.python.training.rmsprop import RMSPropOptimizer

from data_loading import generate_data
from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from plotting import plot_predictions, plot_metrics
from steps import MeanStdNormalizer, ToNumpy


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


class SignalPredictionPipeline(Pipeline):
    HYPERPARAMS = HyperparameterSamples({
        'lambda_loss_amount': 0.003,
        'output_dim': 2,
        'input_dim': 2,
        'hidden_dim': 12,
        'layers_stacked_count': 2,
        'learning_rate': 0.006,
        'lr_decay': 0.92,
        'momentum': 0.5,
        'window_size_future': 40
    })

    def __init__(self, window_size_future, input_dim, output_dim):
        super().__init__([
            ForEachDataInput(MeanStdNormalizer()),
            ToNumpy(),
            Tensorflow2ModelStep(
                create_model=create_model,
                create_loss=create_loss,
                create_optimizer=create_optimizer,
                expected_outputs_dtype=tf.dtypes.float32,
                data_inputs_dtype=tf.dtypes.float32,
                print_loss=True
            ).set_hyperparams(self.HYPERPARAMS).update_hyperparams(HyperparameterSamples({
                'window_size_future': window_size_future,
                'input_dim': input_dim,
                'output_dim': output_dim
            }))
        ])


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
    print('data_inputs shape: {} => (batch_size, sequence_length, input_dim)'.format(data_inputs.shape))
    print('expected_outputs shape: {} => (batch_size, sequence_length, input_dim)'.format(expected_outputs.shape))

    sequence_length = data_inputs.shape[1]
    input_dim = data_inputs.shape[2]
    output_dim = expected_outputs.shape[2]

    batch_size = 10
    epochs = 5
    validation_size = 0.15

    metrics = {'mse': metric_2d_to_3d_wrapper(mean_squared_error)}

    pipeline = DeepLearningPipeline(
        # TODO: use this rather than the DeepLearningPipeline for now, and add a slide for this to compare the
        #       DL Wrapper to full expanded pipeline definition as follow:
        # EpochRepeater(
        #   ValidationSplitWrapper([
        #       TrainOnlyWrapper(Shuffled()),
        #       MiniBatchSequentialPipeline([pipeline!]
        #   ])
        # )
        Pipeline([
            ForEachDataInput(MeanStdNormalizer()),
            ToNumpy(),
            Tensorflow2ModelStep(
                create_model=create_model,
                create_loss=create_loss,
                create_optimizer=create_optimizer,
                expected_outputs_dtype=tf.dtypes.float32,
                data_inputs_dtype=tf.dtypes.float32,
                print_loss=True
            ).set_hyperparams(seq2seq_pipeline_hyperparams).update_hyperparams(HyperparameterSamples({
                'window_size_future': sequence_length,
                'input_dim': input_dim,
                'output_dim': output_dim
            }))
        ]),
        validation_size=0.15,
        batch_size=batch_size,
        batch_metrics=metrics,
        shuffle_in_each_epoch_at_train=True,
        n_epochs=epochs,
        epochs_metrics=metrics,
        scoring_function=metric_2d_to_3d_wrapper(mean_squared_error)
    )

    pipeline, outputs = pipeline.fit_transform(data_inputs, expected_outputs)

    validation_index = math.floor(len(data_inputs) * (1 - validation_size))
    data_inputs_validation = data_inputs[validation_index:]
    expected_outputs_validation = expected_outputs[validation_index:]

    pipeline.set_train(is_train=False)
    pipeline.mutate('transform', 'transform')
    predicted_outputs_validation_data_container = pipeline.transform(
        DataContainer(current_ids=None, data_inputs=data_inputs_validation, expected_outputs=expected_outputs_validation),
        ExecutionContext(DEFAULT_CACHE_FOLDER)
    )

    for i in range(10):
        plot_predictions(
            data_inputs=data_inputs_validation[i],
            expected_outputs=predicted_outputs_validation_data_container.expected_outputs[i],
            predicted_outputs=predicted_outputs_validation_data_container.data_inputs[i],
            exercice_number=exercice_number
        )

    plot_metrics(pipeline=pipeline, exercice_number=exercice_number)


if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)
    # TODO: try GPU
    # with tf.device('/device:GPU:0'):
    main()
    # TODO: prediction charts, too. Use the chart methods of the original repo. And fix english comments that were french in the chart maybe.

    # TODO: how to import external code files: https://stackoverflow.com/questions/48905127/importing-py-files-in-google-colab
