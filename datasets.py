
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

def generate_x_y_data_v2(isTrain, batch_size):
    """
    Similaire a la fonction "v1", ici on genere un signal
    avec 2 frequences choisies au hasard, et cela pour les 2 signaux.
    """
    seq_length = 15

    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random()*2*math.pi
        freq_rand = (random.random()-0.5)/1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
                freq_rand*0.0*math.pi+offset_rand,
                freq_rand*3.0*math.pi+offset_rand,
                seq_length*2
            )
        )

        offset_rand = random.random()*2*math.pi
        freq_rand = (random.random()-0.5)/1.5 * 15 + 0.5
        amp_rand = random.random()*1.2

        sig1 = amp_rand * np.cos(np.linspace(
                freq_rand*0.0*math.pi+offset_rand,
                freq_rand*3.0*math.pi+offset_rand,
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

def generate_x_y_data_v3(isTrain, batch_size):
    """
    Signal plus complique qu'a la version #2, et dans lequel il y a
    du hasard:
    """
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand1 = random.random()*2*math.pi
        offset_rand2 = random.random()*2
        freq_rand1 = (random.random()-0.5)/0.5 * 5 + 0.5
        # freq_rand2 = (random.random()-0.5)/0.5 * 5 + 0.5
        rand_ = (random.random()-0.5)/0.5 * 5 + 0.5

        sample_rate = 50  # 50 Hz resolution
        signal_lenght = 2*sample_rate  # 1 second

        # Generate a random x(t) signal with waves and noise.
        t  = np.linspace(-3+offset_rand1, 6+offset_rand1, signal_lenght)
        # t2 = np.linspace(-1+offset_rand2, 9+offset_rand2, signal_lenght)
        # g = 30*( np.sin((t2/10)**2) )
        x  = 0.30*np.cos(2*np.pi*5.25*t*freq_rand1 - 0.2)
        # x += 0.28*np.sin(2*np.pi*1.50*t*freq_rand2 + 1.0)
        # x += 0.10*np.sin(2*np.pi*5.85*g + 1.0)
        x += 0.10*np.sin(2*np.pi*10.0*t)
        # x += 0.04*np.sin(2*np.pi*20.0*t)
         #x += 0.15*np.cos(2*np.pi*135.0*(t2/5.0-1)**2)

        # Normalize
        x = x - np.average(x)
        x = x / (2.5*np.std(x) + 0.00001)
        x = np.array(list(reversed(x.tolist())))

        # plt.figure(figsize=(11, 9))
        # plt.plot(x)
        # plt.title("Signal")
        # plt.show()

        y = x[int(signal_lenght/2):]
        x = x[:int(signal_lenght/2)]

        x = x + 0.04*np.random.randn(len(x))

        x = x.reshape((len(x), 1))
        y = y.reshape((len(x), 1))

        batch_x.append(x)
        batch_y.append(y)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y

def generate_x_y_data_v4(isTrain, batch_size):
    pass
