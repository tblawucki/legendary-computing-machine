#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import configparser
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dense, TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


def create_model(input_sequences, output_sequences):
    rnn_cell = LSTM
    n_neurons = 100
    n_features = 1
    activation='tanh'
    n_epochs = 500
    batch_size = 32
    model = Sequential()
    n_steps = input_sequences.shape[1]
    model.add(rnn_cell(n_neurons,
                       input_shape=(n_steps, n_features),
                       return_sequences=True,
                       stateful=False,
                       activation=activation))
    model.add(rnn_cell(n_neurons,
                       input_shape=(n_steps, n_features),
                       return_sequences=True,
                       stateful=False,
                       activation=activation))
    model.add(TimeDistributed(Dense(1)))
    
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto', restore_best_weights=True)
    model.fit(x=input_sequences,
              y=output_sequences,
              validation_split = 0.1,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[es])
    return model

def sine_function(X):
    return np.sin(X) + 0.2 * np.cos(0.3 * X + np.random.random()) + np.sin(5 * X) + np.random.random()

def simple_sine_function(X):
    return np.sin(X * np.random.uniform(0.5, 1.5)) * np.random.uniform(0.5, 1) + np.random.uniform(-0.5, 0.5)

def line_function(X):
    return 0*X + np.random.random()

def trend_function(X):
    return X * np.random.uniform(-5, 5)

def random_noise(X):
    return np.random.uniform(size=X.ravel().shape[0])

def generate_dataset(time, functions, n_sequences, offset=10):
    sequences = []
    label_sequences = []
    for i in range(n_sequences):
#        time = time + np.random.uniform()*10
        function = np.random.choice(functions)
        sequence = function(time)
        sequences.append(sequence[:-offset])
        label_sequences.append(sequence[offset:])
    return np.array(sequences).reshape(n_sequences, -1, 1), \
           np.array(label_sequences).reshape(n_sequences, -1, 1)
           


if __name__ == '__main__':
    config = configparser.ConfigParser()
    try:
        config.read('./config.cnf')
    except:
        print('No config file!')
        sys.exit(-1)

    #configuration sections
    default_section = config['DEFAULT']
    forecasting_section = config['FORECASTING']
    
    
    n_steps = int(forecasting_section['n_steps'])
    output_len = int(forecasting_section['forecast_horizon'])
    n_sequences = int(forecasting_section['n_sequences'])
    functions = [simple_sine_function, sine_function, line_function, random_noise]
    time = np.linspace(0, 10, n_steps + output_len)
    X, y = generate_dataset(time, functions, n_sequences)
    
    sf = create_model(X, y)

    
    for i in range(20):
        Xt, yt = generate_dataset(time, functions, 1)
        yp = sf.predict(Xt)
        plt.figure()
        plt.plot((yt).ravel())
        plt.plot(yp.ravel())
        
    
    plt.figure()
    for i in range(20):
        plt.plot(X[i].ravel())