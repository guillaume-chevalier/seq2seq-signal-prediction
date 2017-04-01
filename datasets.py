
import numpy as np

import random
import math

def generate_x_y_data_v1(isTrain, batch_size):
    """
    Donnees pour l'exercice 1.

    retour: tuple (X, Y)
        X est un sinus et un cosinus ayant pour domaine 0.0*pi a 1.5*pi
        Y est un sinus et un cosinus ayant pour domaine 1.5*pi a 3.0*pi
    Donc, un signal en X et sa prolongation en Y.

    Les 2 tableaux retournes sont un de la taille:
        (seq_length, batch_size, output_dim)
        donc: (10, batch_size, 2)

    Pour cet exercice, on ignore l'argument "isTrain",
    car on va prendre les meme donnees de test que pour l'entrainement.
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
    Similaire a la fonction "v1", ici on genere un signal
    avec 2 frequences choisies au hasard, et cela pour les 2 signaux.
    """
    return generate_x_y_data_two_freqs(isTrain, batch_size, seq_length=15)


def generate_x_y_data_v3(isTrain, batch_size):
    """
    Similaire a la fonction "v1", ici on genere un signal
    avec 2 frequences choisies au hasard, et cela pour les 2 signaux.
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

def generate_x_y_data_v4(isTrain, batch_size):
    pass
