import numpy as np
import tensorflow as tf
from neuraxle.api import DeepLearningPipeline
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import GRUCell, RNN, Dense

from datasets import generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3, generate_x_y_data_v4
from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from plotting import plot_metric

GO_TOKEN = -1.


def create_model(step: Tensorflow2ModelStep):
    # shape: (batch_size, seq_length, input_dim)

    # shape: (batch_size, seq_length, input_dim)

    # shape: (batch_size)
    encoder_state = create_encoder(step)
    decoder_outputs = create_decoder(step, encoder_state)

    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def create_encoder(step: Tensorflow2ModelStep):
    encoder_inputs = Input(shape=(None, step.hyperparams['input_dim']))
    encoder = RNN(create_stacked_rnn_cells(step), return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)

    return encoder_outputs_and_states[1:]


def create_decoder(step: Tensorflow2ModelStep, encoder_states):
    decoder_inputs = Input(shape=(None, step.hyperparams['output_dim']))

    decoder_lstm = RNN(create_stacked_rnn_cells(step), return_sequences=True, return_state=True)

    decoder_outputs_and_states = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_dense = Dense(step.hyperparams['output_dim'])
    return decoder_dense(decoder_outputs)


def create_stacked_rnn_cells(step: Tensorflow2ModelStep):
    cells = []
    for _ in range(step.hyperparams['layers_stacked_count']):
        cells.append(GRUCell(step.hyperparams['hidden_dim']))

    return cells


def create_loss(step: TensorflowV1ModelStep):
    # L2 loss prevents this overkill neural network to overfit the data
    l2 = step.hyperparams['lambda_loss_amount'] * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    output_loss = tf.reduce_mean(tf.nn.l2_loss(step['output'] - step['expected_outputs']))

    return output_loss + l2


def create_optimizer(step: TensorflowV1ModelStep):
    return tf.train.RMSPropOptimizer(
        learning_rate=step.hyperparams['learning_rate'],
        decay=step.hyperparams['lr_decay'],
        momentum=step.hyperparams['momentum']
    )


def to_numpy_metric_wrapper(metric_fun):
    def metric(data_inputs, expected_outputs):
        return metric_fun(np.array(data_inputs)[..., 0], np.array(expected_outputs)[..., 0])

    return metric


class SignalPredictionPipeline(Pipeline):
    BATCH_SIZE = 5
    LAMBDA_LOSS_AMOUNT = 0.003
    OUTPUT_DIM = 2
    INPUT_DIM = 2
    HIDDEN_DIM = 12
    LAYERS_STACKED_COUNT = 2
    LEARNING_RATE = 0.1
    LR_DECAY = 0.92
    MOMENTUM = 0.5
    OUTPUT_SIZE = 5
    EPOCHS = 20

    def __init__(self):
        super().__init__([
            Tensorflow2ModelStep(
                create_model=create_model,
                create_loss=create_loss,
                create_optimizer=create_optimizer
            ).set_hyperparams(HyperparameterSamples({
                'batch_size': self.BATCH_SIZE,
                'lambda_loss_amount': self.LAMBDA_LOSS_AMOUNT,
                'output_dim': self.OUTPUT_DIM,
                'output_size': self.OUTPUT_SIZE,
                'input_dim': self.INPUT_DIM,
                'hidden_dim': self.HIDDEN_DIM,
                'layers_stacked_count': self.LAYERS_STACKED_COUNT,
                'learning_rate': self.LEARNING_RATE,
                'lr_decay': self.LR_DECAY,
                'momentum': self.MOMENTUM
            })),
        ])


if __name__ == '__main__':
    exercise = 4
    # We choose which data function to use below, in function of the exericse.
    if exercise == 1:
        generate_x_y_data = generate_x_y_data_v1
    if exercise == 2:
        generate_x_y_data = generate_x_y_data_v2
    if exercise == 3:
        generate_x_y_data = generate_x_y_data_v3
    if exercise == 4:
        generate_x_y_data = generate_x_y_data_v4

    pipeline = DeepLearningPipeline(
        SignalPredictionPipeline(),
        validation_size=0.15,
        batch_size=SignalPredictionPipeline.BATCH_SIZE,
        batch_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        shuffle_in_each_epoch_at_train=True,
        n_epochs=SignalPredictionPipeline.EPOCHS,
        epochs_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        scoring_function=to_numpy_metric_wrapper(mean_squared_error)
    )

    data_inputs, expected_outputs = generate_x_y_data(isTrain=True, batch_size=5)
    for i in range(10):
        di, eo = generate_x_y_data(isTrain=True, batch_size=5)
        data_inputs = np.concatenate([data_inputs, di])
        expected_outputs = np.concatenate([expected_outputs, eo])

    pipeline, outputs = pipeline.fit_transform(data_inputs, expected_outputs)

    mse_train = pipeline.get_epoch_metric_train('mse')
    mse_validation = pipeline.get_epoch_metric_validation('mse')

    plot_metric(mse_train, mse_validation, xlabel='epoch', ylabel='mse', title='Model Mean Squared Error')

    loss = pipeline.get_step_by_name('TensorflowV1ModelStep').loss
    plot_metric(loss, xlabel='batch', ylabel='l2_loss', title='Model L2 Loss')
