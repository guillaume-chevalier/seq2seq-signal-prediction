import numpy as np
import tensorflow as tf
from neuraxle.api import DeepLearningPipeline
from neuraxle.hyperparams.space import HyperparameterSamples
from sklearn.metrics import mean_squared_error

from datasets import generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3, generate_x_y_data_v4
from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep

GO_TOKEN = -1.


def create_graph(step: TensorflowV1ModelStep):
    # shape: (batch_size, seq_length, input_dim)
    data_inputs = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, step.hyperparams['input_dim']],
        name='data_inputs'
    )

    # shape: (batch_size, seq_length, input_dim)
    expected_outputs = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, step.hyperparams['output_dim']],
        name='expected_outputs'
    )

    # shape: (batch_size)
    target_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='expected_outputs_length')

    encoder_state = create_encoder(step, data_inputs)

    decoder_cell = create_stacked_rnn(step)
    decoder_outputs_training = create_training_decoder(step, encoder_state, decoder_cell)
    decoder_outputs_inference = create_inference_decoder(step, encoder_state, decoder_cell)

    return decoder_outputs_training, decoder_outputs_inference


def create_encoder(step, data_inputs):
    encoder_cell = create_stacked_rnn(step)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=data_inputs, dtype=tf.float32)

    return encoder_state


def create_training_decoder(step: TensorflowV1ModelStep, encoder_state, decoder_cell):
    go_tokens = tf.constant(GO_TOKEN, shape=[step.hyperparams['batch_size'], 1, step.hyperparams['output_dim']])
    inputs = tf.concat([go_tokens, step['expected_outputs']], axis=1)

    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=inputs,
        sequence_length=step['expected_outputs_length']
    )

    output = create_decoder_outputs(step, helper, encoder_state, decoder_cell)

    return output


def create_inference_decoder(step: TensorflowV1ModelStep, encoder_state, decoder_cell):
    start_inputs = tf.constant(GO_TOKEN, shape=[step.hyperparams['batch_size'], step.hyperparams['output_dim']])

    helper = tf.contrib.seq2seq.InferenceHelper(
        sample_fn=lambda x: x,
        sample_shape=[step.hyperparams['input_dim']],
        sample_dtype=tf.dtypes.float32,
        start_inputs=start_inputs,
        end_fn=lambda sample_ids: False,
    )

    output = create_decoder_outputs(step, helper, encoder_state, decoder_cell)

    return output


def create_decoder_outputs(step, helper, encoder_state, decoder_cell):
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=helper,
        initial_state=encoder_state,
        output_layer=tf.layers.Dense(units=step.hyperparams['output_dim'])
    )

    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        impute_finished=True,
        maximum_iterations=step.hyperparams['output_size']
    )

    return decoder_outputs.rnn_output


def create_stacked_rnn(step: TensorflowV1ModelStep):
    cells = []
    for _ in range(step.hyperparams['layers_stacked_count']):
        cells.append(tf.contrib.rnn.GRUCell(step.hyperparams['hidden_dim']))

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell


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


def create_feed_dict(step: TensorflowV1ModelStep, data_inputs, expected_outputs):
    expected_outputs_length = []
    for expected_output in expected_outputs:
        expected_outputs_length.append(len(expected_output))

    # shape: (batch_size)
    expected_outputs_length = np.array(expected_outputs_length, dtype=np.int32)

    return {
        step['expected_outputs_length']: expected_outputs_length
    }


def to_numpy_metric_wrapper(metric_fun):
    def metric(data_inputs, expected_outputs):
        return metric_fun(np.array(data_inputs)[..., 0], np.array(expected_outputs)[..., 0])

    return metric


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

    EPOCHS = 150
    BATCH_SIZE = 5

    sample_x, sample_y = generate_x_y_data(isTrain=True, batch_size=BATCH_SIZE)
    output_dim = input_dim = sample_x.shape[-1]

    pipeline = DeepLearningPipeline(
        TensorflowV1ModelStep(
            create_graph=create_graph,
            create_loss=create_loss,
            create_optimizer=create_optimizer,
            create_feed_dict=create_feed_dict
        ).set_hyperparams(HyperparameterSamples({
            'batch_size': BATCH_SIZE,
            'lambda_loss_amount': 0.003,
            'output_dim': output_dim,
            'output_size': 5,
            'input_dim': input_dim,
            'hidden_dim': 12,
            'layers_stacked_count': 2,
            'learning_rate': 0.007,
            'lr_decay': 0.92,
            'momentum': 0.5
        })),
        validation_size=0.15,
        batch_size=BATCH_SIZE,
        batch_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        shuffle_in_each_epoch_at_train=True,
        n_epochs=EPOCHS,
        epochs_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        scoring_function=to_numpy_metric_wrapper(mean_squared_error)
    )

    X, Y = generate_x_y_data(isTrain=True, batch_size=BATCH_SIZE)
    pipeline, outputs = pipeline.fit_transform(data_inputs=X, expected_outputs=Y)
