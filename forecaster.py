#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, TimeDistributed
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import talos
from utils import print, ConfigurationError

def create_forecaseter_model(X_train, y_train, X_val=None, y_val=None, params=None, *args, **kwargs):
    keras.backend.clear_session()   
    input_sequences  = X_train
    output_sequences = y_train
    rnn_cell         = LSTM if params['rnn_cell'] == 'LSTM' else GRU
    n_layers         = int(params['n_layers'])
    n_neurons        = [int(p) for p in params['n_neurons'].split('|')]
    batch_size       = int(params['batch_size'])
    n_steps          = int(params['n_steps'])
    n_features       = int(params['n_features'])
    n_epochs         = int(params['n_epochs'])
    activation       = params['activation']
    loss_metrics     = params['loss_metrics']
    scan             = params['scan']
    model_name       = [''] if scan else params['model_name']
    early_stopping   = params['early_stopping'] == 'True'
    
    if len(n_neurons) == 1:
        n_neurons = [n_neurons[0] for _ in range(n_layers)]

    model = Sequential()
    for i in range(n_layers):
        model.add(rnn_cell(n_neurons[i], 
                           input_shape=(n_steps, n_features),
                           return_sequences=True,
                           stateful=False,
                           activation=activation))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss=loss_metrics)
    model.summary()
    
    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto', restore_best_weights=True)
    callbacks = [] if not early_stopping else [es]
    hist = model.fit(x=input_sequences,
              y=output_sequences,
              validation_split = 0.1,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=False,
              callbacks=callbacks)
    if scan:
        return hist, model
    else:
        print(f'SAVING MODEL {model_name} TO "./models/forecaster_{model_name}.pkl"')
        model.save(f'./models/forecaster_{model_name}.pkl')
        plot_model(model, to_file=f'./architecture/forecaster_{model_name}_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')
        return model

class SKU_Forecaster:
    def __init__(self, *args, **kwargs):
        self.forecast_column = kwargs.get('forecast_column')
        self.sku_path = kwargs.get('sku_path')

        self.rnn_cell = kwargs.get('rnn_cell', 'LSTM').split(';')
        self.output_length    = [int(p) for p in kwargs['forecast_horizon'].split(';')][0]
        self.n_layers         = [int(p) for p in kwargs['n_layers'].split(';')]
#        self.n_neurons        = [int(p) for p in kwargs['n_neurons'].split(';')]
        self.n_neurons      =   [p for p in kwargs['n_neurons'].split(';')]
        self.batch_size       = [int(p) for p in kwargs['batch_size'].split(';')]
        self.n_steps          = [int(p) for p in kwargs['n_steps'].split(';')][0]
#        self.n_features       = [int(p) for p in kwargs['n_features'].split(';')][0]
        self.n_epochs         = [int(p) for p in kwargs['n_epochs'].split(';')]
        self.activation       = kwargs['activation'].split(';')
        self.loss_metrics     = kwargs['loss_metrics'].split(';')[0]
        self.early_stopping   = [kwargs['early_stopping'] == 'True']

        self.cold_start = kwargs.get('cold_start') == 'True' 
        self.encoding = kwargs.get('encoding', 'utf8')
        self.full_dataset = kwargs.get('full_dataset') == 'True'
        self._load_datasets = self._load_datasets_full if self.full_dataset else self._load_datasets_partial
        self.trained = False


    def _load_datasets_partial(self):
        datasets = []
        labels = []
        columns = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding=self.encoding,
                                               sep=';')
            n_splits = df.shape[0] // self.n_steps
#            trim = df.shape[0] % self.n_steps
#            df = df[trim:]
            for split_idx in range(n_splits):
                chunk = df[split_idx * self.n_steps : \
                           (split_idx + 1) * self.n_steps]
                chunk_label = df[split_idx * self.n_steps + self.output_length : \
                                 (split_idx + 1) * self.n_steps + self.output_length]
                datasets.append(chunk.values)
                labels.append(chunk_label[self.forecast_column].values)
        columns = df.columns
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
#            trim = df.shape[0] % self.n_steps
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
        params = { 'rnn_cell':self.rnn_cell,
                   'n_layers':self.n_layers,
                   'n_epochs':self.n_epochs,
                   'n_neurons':self.n_neurons,
                   'n_steps':[self.n_steps],
                   'n_features':[self.input_sequences.shape[2]],
                   'batch_size':self.batch_size,
                   'activation':self.activation,
                   'output_length':[self.output_length],
                   'loss_metrics':[self.loss_metrics],
                   'model_name':['talos_scan_model'], 
                   'early_stopping':self.early_stopping,
                   'scan':[True]}
        
        model_files = os.listdir('./models/')
        model_exists = f'forecaster_{model_name}' in ''.join(model_files)
        if not self.cold_start and model_exists:
            print('MODEL EXISTS, LOADING...')
            forecaster_file = f'forecaster_{model_name}.pkl'
            self.forecaster = load_model(os.path.join('./models/', forecaster_file))
        else:
            print('TRANING MODEL...')
            results = talos.Scan(X, y, params=params, model=create_forecaseter_model)
            best_params = results.data.sort_values(by=['val_loss'], ascending=True).iloc[0].to_dict()
            best_params = {key:str(value) for key,value in best_params.items()}
            best_params['scan'] = False
            best_params['model_name'] = model_name
            self.forecaster = create_forecaseter_model(X, y, params=best_params)
            
#            self.forecaster = results.best_model(metric='val_loss')
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
    
    
    
if __name__ == '__main__':
    import configparser
    import sys
    
    config = configparser.ConfigParser()
    try:
        config.read('./test_config.cnf')
    except:
        print('No config file!')
        sys.exit(-1)

    #configuration sections
    forecasting_section = config['FORECASTING']
    
    sf = SKU_Forecaster(**forecasting_section)
    X, y, columns = sf._load_datasets()
    y = y.reshape(*y.shape, 1)
    forecaster = sf.train(X, y, model_name='TestModelForecaster')
