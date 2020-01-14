import math
import random

import numpy as np
import requests
import matplotlib.pyplot as plt

from steps import WindowTimeSeries


def fetch_data(window_size_past, window_size_future):
    data_inputs_usd = load_currency("USD")
    data_inputs_eur = load_currency("EUR")

    data_inputs_usd = np.expand_dims(np.array(data_inputs_usd), axis=1)
    data_inputs_eur = np.expand_dims(np.array(data_inputs_eur), axis=1)
    data_inputs = np.concatenate((data_inputs_usd, data_inputs_eur), axis=1)

    return WindowTimeSeries(window_size_past=window_size_past, window_size_future=window_size_future).transform(
        (data_inputs, None))


def generate_x_y_data_v1(batch_size, sequence_length=10):
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

    batches = []
    for _ in range(batch_size):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, sequence_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, sequence_length * 2))
        data_inputs = np.array([sig1, sig2])
        data_inputs = data_inputs.T

        batches.append(data_inputs)

    batches = np.array(batches)
    # shape: (batch_size, seq_length, output_dim)

    batches = np.array(batches).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batches


def generate_x_y_data_v2(batch_size, sequence_length=15):
    """
    Similar the the "v1" function, but here we generate a signal with
    2 frequencies chosen randomly - and this for the 2 signals. Plus,
    the lenght of the examples is of 15 rather than 10.
    So we have 30 total values for past and future.
    """
    return generate_x_y_data_two_freqs(batch_size, seq_length=sequence_length)


def generate_x_y_data_v3(batch_size, sequence_length=30):
    """
    Similar to the "v2" function, but here we generate a signal
    with noise in the X values.
    """
    data_inputs = generate_x_y_data_two_freqs(batch_size, seq_length=sequence_length)
    noise_amount = random.random() * 0.15 + 0.10
    data_inputs = data_inputs + noise_amount * np.random.randn(sequence_length, batch_size, 1)

    return data_inputs


def generate_x_y_data_two_freqs(batch_size, seq_length):
    batches = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        ))

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sig1 = amp_rand * np.cos(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )) + sig1

        x1 = sig1

        x_ = np.array([x1])
        x_ = x_.T

        batches.append(x_)

    # shape: (batch_size, seq_length, output_dim)

    batches = np.array(batches).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batches


def load_currency(currency):
    """
    Return the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by :
    # https://github.com/Levino/coindesk-api-node
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2020-01-01&currency={}".format(
            currency
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values[1000:]

    return kept_values
