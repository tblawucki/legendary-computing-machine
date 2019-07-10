#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, TimeDistributed
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

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
                               return_sequences=True,
                               stateful=False,
                               activation=activation))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss=loss_metrics)
    model.summary()
    
    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto', restore_best_weights=True)
    model.fit(x=input_sequences,
              y=output_sequences,
              validation_split = 0.1,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=False,
              callbacks=[])
    print(f'SAVING MODEL TO "./models/forecaster_{model_name}.pkl"')
    model.save(f'./models/forecaster_{model_name}.pkl')
    
    plot_model(model, to_file=f'./architecture/forecaster_{model_name}_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')

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
        self.encoding = kwargs.get('encoding', 'utf8')
        self.full_dataset = True if kwargs.get('full_dataset') == 'True' else False
        self._load_datasets = self._load_datasets_full if self.full_dataset else self._load_datasets_partial


    def _load_datasets_partial(self):
        datasets = []
        labels = []
        columns = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding=self.encoding,
                                               sep=';')
            n_splits = df.shape[0] // self.n_steps
            trim = df.shape[0] % self.n_steps
#            df = df[trim:]
            for split_idx in range(n_splits):
                chunk = df[split_idx * self.n_steps : \
                           (split_idx + 1) * self.n_steps]
                chunk_label = df[split_idx * self.n_steps + self.output_length : \
                                 (split_idx + 1) * self.n_steps + self.output_length]
                datasets.append(chunk.values)
                labels.append(chunk_label[self.forecast_column].values)
        columns = df.columns
        print(type(datasets), len(datasets), datasets[-1])
        return np.array(datasets, dtype=np.float64), \
               np.array(labels, dtype=np.float64), \
               columns
    
    def _load_datasets_full(self):
        datasets = []
        labels = []
        columns = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding=self.encoding,
                                               sep=';')
            n_splits = df.shape[0] // self.n_steps
            trim = df.shape[0] % self.n_steps
#            df = df[trim:]
#            for offset in range(n_splits * (self.n_steps - 1) - trim - self.output_length):
            for offset in range((n_splits -1) * self.n_steps):
                chunk = df[offset : offset + self.n_steps]
                chunk_label = df[offset + self.output_length : offset + self.n_steps + self.output_length]
                datasets.append(chunk.values)
                labels.append(chunk_label[self.forecast_column].values)
        columns = df.columns
        return np.array(datasets, dtype=np.float64), \
               np.array(labels, dtype=np.float64), \
               columns

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
            hist = self.forecaster.history.history
            loss = hist['loss']
            val_loss = hist['val_loss']
            plt.figure(figsize=(10, 7))
            plt.plot(loss, label='training_loss')
            plt.plot(val_loss, label='validation_loss')
            plt.legend()
            plt.title(f'Forecaster {model_name} loss')
            plt.savefig(f'./loss/forecaster_{model_name}_loss.png')
        return self.forecaster
        
    def predict(self, X):
        if not self.trained:
            print('Model not trained yet')
            return None
        prediction = self.forecaster.predict(X)
        return prediction