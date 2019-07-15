#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:05:09 2019

@author: tom
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import configparser
import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
# =============================================================================
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU
# from tensorflow.keras.layers import Dense, TimeDistributed
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.backend import clear_session
# =============================================================================
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense, TimeDistributed
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.backend import clear_session
# =============================================================================
# sklearn option
# =============================================================================
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
# =============================================================================
# talos option
# =============================================================================
import talos

#%%


# Reset Keras Session
def reset_keras():
    from keras.backend.tensorflow_backend import set_session
    from keras.backend.tensorflow_backend import clear_session
    from keras.backend.tensorflow_backend import get_session
    import tensorflow
    from numba import cuda

    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        print('No model found')

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))
#    cuda.select_device(0)
#    cuda.close()
#    cuda.detect()

def create_model(x_train, y_train, x_val, y_val, params):
    clear_session()
    rnn_cell = LSTM
    n_features = 1
    activation = params['activation']
    model = Sequential()
    n_steps = x_train.shape[1]
    for i in range(params['n_layers']):
        model.add(rnn_cell(params['n_neurons'],
                       input_shape=(n_steps, n_features),
                       return_sequences=True,
                       stateful=False,
                       activation=activation))
    model.add(TimeDistributed(Dense(1)))
    
    model.compile(optimizer=params['optimizer'], loss='mse')
    model.summary()
    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto', restore_best_weights=True)
    out = model.fit(x=x_train,
              y=y_train,
              validation_split = 0.1,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              shuffle=True,
              callbacks=[es])
#    print(out)
    return out, model

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
           
#%%

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
    functions = [simple_sine_function, sine_function, line_function]
    time = np.linspace(0, 10, n_steps + output_len)
    X, y = generate_dataset(time, functions, n_sequences)
    
    params = {  'n_layers':[2,4],
                'n_neurons':[50, 200, 500],
                'activation': ['elu', 'relu', 'tanh'],
                'optimizer': ['Adam', 'Adagrad', 'sgd', 'rmsprop', 'Adadelta'],
                'batch_size': [32],
                'epochs': [400]
            }
    hist = talos.Scan(X, y, model=create_model, params=params, print_params=True)
    hist.x = np.zeros(30)
    hist.y = np.zeros(30)
    r = talos.Reporting(hist)
    r_df = r.data
    talos.Deploy(hist, 'predictor', metric='val_loss', asc=True)
    sf = talos.Restore('predictor.zip').model
    
    for i in range(20):
        Xt, yt = generate_dataset(time, functions, 1)
        yp = sf.predict(Xt)
        plt.figure()
        plt.plot((yt).ravel())
        plt.plot(yp.ravel())
        
    
    plt.figure()
    for i in range(20):
        plt.plot(X[i].ravel())
        
        
writer = pd.ExcelWriter('talos_exp.xlsx')
r_df.to_excel(writer)
writer.close()

reset_keras()
