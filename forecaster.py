#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


def create_forecaseter_model(input_sequences, output_sequences, rnn_cell,
                             n_layers, n_neurons, batch_size, n_steps,
                             n_features, activation, output_length,
                             loss_metrics, n_epochs, model_name, *args, **kwargs):
    rnn_cell = LSTM if rnn_cell == 'LSTM' else GRU
    model = Sequential()
    for i in range(n_layers -1):
        model.add(rnn_cell(n_neurons,input_shape=(n_steps,
                                                  n_features),
                                   return_sequences=True,
                                   stateful=False,
                                   activation=activation))
    model.add(rnn_cell(n_neurons,
                               input_shape=(n_steps,
                                            n_features),
                               return_sequences=False,
                               stateful=False,
                               activation=activation))
    model.add(Dense(output_length))
    model.compile(optimizer='adam', loss=loss_metrics)
    model.summary()
    model.fit(x=input_sequences,
              y=output_sequences,
              validation_split = 0.1,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=False)
    print(f'SAVING MODEL TO "./models/forecaster_{model_name}.pkl"')
    model.save(f'./models/forecaster_{model_name}.pkl')
    return model

class SKU_Forecaster:
    def __init__(self, *args, **kwargs):
        self.rnn_cell = kwargs.get('rnn_cell', 'LSTM')
        self.n_layers = int(kwargs.get('n_layers'))
        self.n_epochs = int(kwargs.get('n_epochs'))
        self.n_neurons = int(kwargs.get('n_neurons'))
        self.n_steps = int(kwargs.get('n_steps'))
        self.batch_size = int(kwargs.get('batch_size'))
        self.activation = kwargs.get('activation', 'tanh')
        self.output_length = int(kwargs.get('forecast_horizon'))
        self.forecast_column = kwargs.get('forecast_column')
        self.loss_metrics = kwargs.get('loss_metrics', ['mse']).split(';')
        self.sku_path = kwargs.get('sku_path')
        self.cold_start = True if kwargs.get('cold_start') == 'True' else False 
        self.trained = False

    def _load_datasets(self):
        datasets = []
        labels = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding='cp1252',
                                               sep=';')
            n_splits = df.shape[0] // self.n_steps
            trim = df.shape[0] % self.n_steps
#            df = df[trim:]
            for split_idx in range(n_splits):
                chunk = df[split_idx * self.n_steps : \
                           (split_idx + 1) * self.n_steps]
                chunk_label = df[(split_idx + 1) * self.n_steps : \
                                 (split_idx + 1) * self.n_steps + self.output_length]
                datasets.append(chunk.values)
                labels.append(chunk_label[self.forecast_column].values)
        return np.array(datasets, dtype=np.float64), \
               np.array(labels, dtype=np.float64)
        

    def train(self, X, y, model_name='0'):
        self.trained = True
        self.input_sequences = X
        self.output_sequences = y
        parameters_set = { 'rnn_cell':self.rnn_cell,
                           'n_layers':self.n_layers,
                           'n_epochs':self.n_epochs,
                           'n_neurons':self.n_neurons,
                           'n_steps':self.n_steps,
                           'n_features':self.input_sequences.shape[2],
                           'batch_size':self.batch_size,
                           'activation':self.activation,
                           'output_length':self.output_length,
                           'loss_metrics':self.loss_metrics[0],
                           'input_sequences':self.input_sequences,
                           'output_sequences':self.output_sequences,
                           'model_name':model_name}
        model_files = os.listdir('./models/')
        model_exists = f'forecaster_{model_name}' in ''.join(model_files)
        if not self.cold_start and model_exists:
            print('MODEL EXISTS, LOADING...')
            forecaster_file = f'forecaster_{model_name}.pkl'
            self.forecaster = load_model(os.path.join('./models/', forecaster_file))
        else:
            print('TRANING MODEL...')
            self.forecaster = create_forecaseter_model(**parameters_set)
        return self.forecaster
        
    def predict(self, X):
        if not self.trained:
            print('Model not trained yet')
            return None
        prediction = self.forecaster.predict(X)
        return prediction