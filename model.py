import numpy as np
import tensorflow as tf
# Backward compatibility for TensorFlow's version 0.12:
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import MiniBatchSequentialPipeline, Pipeline, Joiner
from neuraxle.steps.data import EpochRepeater

from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep
from seq2seq import generate_x_y_data

try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except:
    print("TensorFlow's version : 0.12")


def create_graph(step: TensorflowV1ModelStep):
    # Encoder: inputs
    encoder_inputs = [
        tf.placeholder(tf.float32, shape=(None, step.hyperparams['input_dim']), name="inp_{}".format(t))
        for t in range(step.hyperparams['seq_length'])
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, step.hyperparams['output_dim']),
                       name="expected_sparse_output_".format(t))
        for t in range(step.hyperparams['seq_length'])
    ]

    # Give a "GO" token to the decoder.
    # Note: we might want to fill the encoder with zeros or its own feedback rather than with "+ encoder_inputs[:-1]"
    dec_inp = [tf.zeros_like(encoder_inputs[0], dtype=np.float32, name="GO")] + encoder_inputs[:-1]

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(step.hyperparams['layers_stacked_count']):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(step.hyperparams['hidden_dim']))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def.
    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        encoder_inputs,
        dec_inp,
        cell
    )

    # For reshaping the output dimensions of the seq2seq RNN:
    w_out = tf.Variable(tf.random_normal([step.hyperparams['hidden_dim'], step.hyperparams['output_dim']]),
                        name='w_out')
    b_out = tf.Variable(tf.random_normal([step.hyperparams['output_dim']]), name='b_out')

    # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
    tf.Variable(1.0, name="output_scalefactor")
    return tf.Variable(
        [step['output_scalefactor'] * (tf.matmul(i, step['w_out']) + step['b_out']) for i in dec_outputs])


def create_loss(step: TensorflowV1ModelStep):
    # Loss, optimizer and evaluation
    # L2 loss prevents this overkill neural network to overfit the data
    l2 = step.hyperparams['lambda_loss_amount'] * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    return tf.reduce_mean(tf.nn.l2_loss(step['output'])) + l2


def create_optimizer(step: TensorflowV1ModelStep):
    return tf.train.RMSPropOptimizer(
        learning_rate=step.hyperparams['learning_rate'],
        decay=step.hyperparams['lr_decay'],
        momentum=step.hyperparams['momentum']
    )


if __name__ == '__main__':
    EPOCHS = 150
    BATCH_SIZE = 5

    sample_x, sample_y = generate_x_y_data(isTrain=True, batch_size=BATCH_SIZE)
    seq_length = sample_x.shape[0]
    output_dim = input_dim = sample_x.shape[-1]

    model_step = TensorflowV1ModelStep(
        create_graph=create_graph,
        create_loss=create_loss,
        create_optimizer=create_optimizer
    ).set_hyperparams(HyperparameterSamples({
        'output_dim': output_dim,
        'input_dim': input_dim,
        'seq_length': seq_length,
        'hidden_dim': 12,
        'layers_stacked_count': 2,
        'learning_rate': 0.007,
        'lr_decay': 0.92,
        'momentum': 0.5,
        'lambda_l2_reg': 0.003
    }))

    pipeline = Pipeline([
        EpochRepeater(
            MiniBatchSequentialPipeline([
                model_step,
                Joiner(batch_size=BATCH_SIZE)
            ]),
            epochs=EPOCHS
        )
    ])

    X, Y = generate_x_y_data(isTrain=True, batch_size=BATCH_SIZE)
    pipeline, outputs = pipeline.fit_transform(data_inputs=X, expected_outputs=Y)
