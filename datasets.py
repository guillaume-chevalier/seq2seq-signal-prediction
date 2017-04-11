__author__ = "Guillaume Chevalier"
__license__ = "MIT"
__version__ = "2017-03"

import numpy as np
import requests

import random
import math

def generate_x_y_data_v1(isTrain, batch_size):
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
    seq_length = 10

    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        rand = random.random()*2*math.pi

        sig1 = np.sin(np.linspace(0.0*math.pi+rand, 3.0*math.pi+rand, seq_length*2))
        sig2 = np.cos(np.linspace(0.0*math.pi+rand, 3.0*math.pi+rand, seq_length*2))
        x1 = sig1[:seq_length]
        y1 = sig1[seq_length:]
        x2 = sig2[:seq_length]
        y2 = sig2[seq_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y

def generate_x_y_data_two_freqs(isTrain, batch_size, seq_length):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random()*2*math.pi
        freq_rand = (random.random()-0.5)/1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
                seq_length/15.0*freq_rand*0.0*math.pi+offset_rand,
                seq_length/15.0*freq_rand*3.0*math.pi+offset_rand,
                seq_length*2
            )
        )

        offset_rand = random.random()*2*math.pi
        freq_rand = (random.random()-0.5)/1.5 * 15 + 0.5
        amp_rand = random.random()*1.2

        sig1 = amp_rand * np.cos(np.linspace(
                seq_length/15.0*freq_rand*0.0*math.pi+offset_rand,
                seq_length/15.0*freq_rand*3.0*math.pi+offset_rand,
                seq_length*2
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

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y

def generate_x_y_data_v2(isTrain, batch_size):
    """
    Similar the the "v1" function, but here we generate a signal with
    2 frequencies chosen randomly - and this for the 2 signals. Plus, 
    the lenght of the examples is of 15 rather than 10. 
    So we have 30 total values for past and future. 
    """
    return generate_x_y_data_two_freqs(isTrain, batch_size, seq_length=15)


def generate_x_y_data_v3(isTrain, batch_size):
    """
    Similar to the "v2" function, but here we generate a signal
    with noise in the X values. Plus, 
    the lenght of the examples is of 30 rather than 10. 
    So we have 60 total values for past and future. 
    """
    seq_length = 30
    x, y = generate_x_y_data_two_freqs(isTrain, batch_size, seq_length=seq_length)
    noise_amount = random.random() * 0.15 + 0.10
    x = x + noise_amount*np.random.randn(seq_length, batch_size, 1)

    avg = np.average(x)
    std = np.std(x) + 0.0001
    x = x - avg
    y = y - avg
    x = x / std / 2.5
    y = y / std / 2.5

    return x, y

def loadCurrency(curr, window_size):
    """
    Returns the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by : https://github.com/Levino/coindesk-api-node
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2017-03-03&currency={}".format(
            curr
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values[1000:]

    X = []
    Y = []
    for i in range(len(kept_values)-window_size*2):
        X.append(kept_values[i:i+window_size])
        Y.append(kept_values[i+window_size:i+window_size*2])

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    return X, Y

def normalize(X, Y=None):
    """
    Normalises X et Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean   = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(    X, axis=1) + 0.00001, axis=1)
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5*stddev)
    if Y is not None:
        assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5*stddev)
        return X, Y
    return X

def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples. 
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
    assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    return X_out, Y_out

X_train = []
Y_train = []
X_test = []
Y_test = []
def generate_x_y_data_v4(isTrain, batch_size):
    """
    Returns financial data for the bitcoin. 
    Features are USD and EUR, in the internal dimension. 
    We normalize X and Y data according to the X only to not 
    spoil the predictions we ask for. 
    
    For every window (window or seq_length), Y is the prediction following X.
    Train and test data are separated according to the 80/20 rule. 
    Therefore, the 20 percent of the test data are the most 
    recent historical bitcoin values. Every example in X contains
    40 points of USD and then EUR data in the feature axis/dimension. 
    It is to be noted that the returned X and Y has the same shape 
    and are in a tuple. 
    """
    # 40 pas values for encoder, 40 after for decoder's predictions.
    seq_length = 40

    global Y_train
    global X_train
    global X_test
    global Y_test
    # First load, with memoization:
    if len(Y_test) == 0:
        # API call:
        X_usd, Y_usd = loadCurrency("USD", window_size=seq_length)
        X_eur, Y_eur = loadCurrency("EUR", window_size=seq_length)

        # All data, aligned:
        X = np.concatenate((X_usd, X_eur), axis=2)
        Y = np.concatenate((Y_usd, Y_eur), axis=2)
        X, Y = normalize(X, Y)

        # Split 80-20:
        X_train = X[:int(len(X)*0.8)]
        Y_train = Y[:int(len(Y)*0.8)]
        X_test = X[int(len(X)*0.8):]
        Y_test = Y[int(len(Y)*0.8):]

    if isTrain:
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    else:
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)
