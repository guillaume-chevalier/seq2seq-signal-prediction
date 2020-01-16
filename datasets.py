import json
import random
import urllib
from typing import Callable

import math
import numpy as np


def generate_data(
        exercice_number,
        window_size_past=None,
        window_size_future=None,
        n_samples=None
):
    if exercice_number == 1:
        return generate_data_v1(n_samples, window_size_past)

    if exercice_number == 2:
        return generate_data_v2(n_samples, window_size_past)

    if exercice_number == 3:
        return generate_data_v3(n_samples, window_size_past)

    if exercice_number == 4:
        return generate_data_v4(n_samples, window_size_future, window_size_past)


def generate_data_v1(n_samples, sequence_length):
    """
    Data for exercise 1.

    returns: tuple (X, Y)
        X is a sine and a cosine from 0.0*pi to 1.5*pi
        Y is a sine and a cosine from 1.5*pi to 3.0*pi
    Therefore, Y follows X. There is also a random offset
    commonly applied to X an Y.

    The returned arrays are of shape:
        (seq_length, batch_size, output_dim)
        Therefore: (10, batch_size, 2)

    For this exercise, let's ignore the "isTrain"
    argument and test on the same data.
    """
    if n_samples is None:
        n_samples = 1000
    if sequence_length is None:
        sequence_length = 10

    batch_x = []
    batch_y = []
    for _ in range(n_samples):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, sequence_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, sequence_length * 2))
        x1 = sig1[:sequence_length]
        y1 = sig1[sequence_length:]
        x2 = sig2[:sequence_length]
        y2 = sig2[sequence_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    return batch_x, batch_y


def generate_data_v2(n_samples, sequence_length):
    """
    Similar the the "v1" function, but here we generate a signal with
    2 frequencies chosen randomly - and this for the 2 signals. Plus,
    the lenght of the examples is of 15 rather than 10.
    So we have 30 total values for past and future.
    """
    if n_samples is None:
        n_samples = 10000
    if sequence_length is None:
        sequence_length = 15

    return generate_data_two_freqs(n_samples, seq_length=sequence_length)


def generate_data_v3(n_samples, sequence_length):
    """
    Similar to the "v2" function, but here we generate a signal
    with noise in the X values.
    """
    if n_samples is None:
        n_samples = 10000
    if sequence_length is None:
        sequence_length = 30

    x, y = generate_data_two_freqs(n_samples, seq_length=sequence_length)
    noise_amount = random.random() * 0.15 + 0.10
    x = x + noise_amount * np.random.randn(n_samples, sequence_length, 1)

    avg = np.average(x)
    std = np.std(x) + 0.0001
    x = x - avg
    y = y - avg
    x = x / std / 2.5
    y = y / std / 2.5

    return x, y


def generate_data_two_freqs(batch_size, seq_length):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        )

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sig1 = amp_rand * np.cos(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        ) + sig1

        x1 = sig1[:seq_length]
        y1 = sig1[seq_length:]

        x_ = np.array([x1])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    return batch_x, batch_y


def load_currency(currency):
    """
    Return the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by :
    # https://github.com/Levino/coindesk-api-node
    req = urllib.request.Request(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2017-12-01&currency={}".format(
            currency),
        method="GET",
        headers={'content-type': 'application/json'}
    )
    response = urllib.request.urlopen(req)
    data = json.loads(response.read())

    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values[1000:]

    return kept_values


def generate_data_v4(n_samples, window_size_future, window_size_past):
    if n_samples is None:
        n_samples = 2000
    if window_size_past is None:
        window_size_past = 40
    if window_size_future is None:
        window_size_future = 40

    data_inputs_usd = load_currency("USD")
    data_inputs_eur = load_currency("EUR")

    data_inputs_usd = np.expand_dims(np.array(data_inputs_usd), axis=1)
    data_inputs_eur = np.expand_dims(np.array(data_inputs_eur), axis=1)

    data_inputs = np.concatenate((data_inputs_usd, data_inputs_eur), axis=1)
    data_inputs = data_inputs[:n_samples]

    return window_time_series(
        data_inputs=data_inputs,
        window_size_past=window_size_past,
        window_size_future=window_size_future
    )


def window_time_series(data_inputs, window_size_past, window_size_future):
    new_data_inputs = []
    new_expected_outputs = []

    for i in range(len(data_inputs) - window_size_past - window_size_future):
        new_data_inputs.append(data_inputs[i: i + window_size_past])
        new_expected_outputs.append(
            data_inputs[i + window_size_past: i + window_size_past + window_size_future])

    return np.array(new_data_inputs), np.array(new_expected_outputs)


def metric_3d_to_2d_wrapper(metric_fun: Callable):
    def metric(data_inputs, expected_outputs):
        # We keep axis 0 for evaluation on USD only (or on first dim only when we have multidim output).
        return metric_fun(np.array(data_inputs)[..., 0], np.array(expected_outputs)[..., 0])

    return metric
