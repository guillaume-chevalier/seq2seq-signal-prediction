from typing import Callable

import numpy as np
import tensorflow as tf
from neuraxle.api import DeepLearningPipeline
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEachDataInput
from sklearn.metrics import mean_squared_error
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import GRUCell, RNN, Dense
from tensorflow_core.python.training.rmsprop import RMSPropOptimizer

from data_loading import fetch_data, generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3
from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from plotting import plot_metric
from steps import MeanStdNormalizer, ToNumpy


def create_model(step: Tensorflow2ModelStep):
    # shape: (batch_size, seq_length, input_dim)
    encoder_inputs = Input(shape=(None, step.hyperparams['input_dim']), dtype=tf.dtypes.float32)

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

    def __init__(self):
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
            ).set_hyperparams(self.HYPERPARAMS)
        ])


def metric_2d_to_3d_wrapper(metric_fun: Callable, index_column_for_metric=0):
    def metric(data_inputs, expected_outputs):
        return metric_fun(np.array(data_inputs)[..., 0], np.array(expected_outputs)[..., 0])

    return metric


# EpochRepeater(
#   ValidationSplitWrapper([
#       TrainOnlyWrapper(Shuffled()),
#       MiniBatchSequentialPipeline([pipeline!]
#   ])
# )

def main():
    batch_size = 100
    epochs = 50

    pipeline = DeepLearningPipeline(
        SignalPredictionPipeline(),
        validation_size=0.15,
        batch_size=batch_size,
        batch_metrics={'mse': metric_2d_to_3d_wrapper(mean_squared_error)},
        shuffle_in_each_epoch_at_train=True,
        n_epochs=epochs,
        epochs_metrics={'mse': metric_2d_to_3d_wrapper(mean_squared_error)},
        scoring_function=metric_2d_to_3d_wrapper(mean_squared_error)
    )

    data_inputs, expected_outputs = fetch_data(window_size_past=40, window_size_future=SignalPredictionPipeline.HYPERPARAMS['window_size_future'])
    # data_inputs, expected_outputs = generate_x_y_data_v1(batch_size=5, sequence_length=10)
    # data_inputs, expected_outputs = generate_x_y_data_v2(batch_size=5, sequence_length=15)
    # data_inputs, expected_outputs = generate_x_y_data_v3(batch_size=5, sequence_length=30)

    pipeline.get_step_by_name("Tensorflow2ModelStep").update_hyperparams({
        'window_size_future': 40
    })

    pipeline, outputs = pipeline.fit_transform(data_inputs, expected_outputs)

    mse_train = pipeline.get_epoch_metric_train('mse')
    mse_validation = pipeline.get_epoch_metric_validation('mse')

    plot_metric(mse_train, mse_validation, xlabel='epoch', ylabel='mse', title='Model Mean Squared Error')

    loss_train = pipeline.get_step_by_name('Tensorflow2ModelStep').train_losses
    loss_test = pipeline.get_step_by_name('Tensorflow2ModelStep').test_losses
    plot_metric(loss_train, loss_test, xlabel='batch', ylabel='l2_loss', title='Model L2 Loss')


if __name__ == '__main__':
    main()
