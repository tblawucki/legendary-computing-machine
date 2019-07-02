#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete app for SKU preprocessing, clustering and forecasting
@author: t.blawucki@smartgeometries.pl
@company: SmartGeometries SP. Z.O.O
"""
import configparser
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from preprocessor import SKU_Preprocessor
from clusterer import SKU_Clusterer
from forecaster import SKU_Forecaster
import sklearn.metrics as metrics
# =============================================================================
# TEST SERIES GENERATOR
# =============================================================================

def sine_function(X):
    return np.sin(X) + 0.2 * np.cos(0.3 * X + np.random.random()) + np.sin(5 * X) + np.random.random()

def simple_sine_function(X):
    return np.sin(X + np.random.random()) * np.random.uniform(0.5, 1)

def line_function(X):
    return 0*X + np.random.random()

def random_noise(X):
    return np.random.uniform(size=X.ravel().shape[0])

def generate_dataset(time, functions, n_sequences, offset=10):
    sequences = []
    classes = []
    label_sequences = []
    for i in range(n_sequences):
        time = time + np.random.random()*10
        function = np.random.choice(functions)
        sequence = function(time)
        sequences.append(sequence[:-offset])
        label_sequences.append(sequence[-offset:])
    return np.array(sequences).reshape(n_sequences, -1, 1), \
           np.array(label_sequences).reshape(n_sequences, -1)

# =============================================================================
# 4. Rescale/destandarize result of prediction
# =============================================================================

# =============================================================================
# 5. Measure error of prediction for each sample
#   a) Mean Square Error
#   b) Root Mean Square Error
#   c) Mean Absolute Error
#   d) Mean Absolute Precentage Error
# =============================================================================


if __name__ == '__main__':
    config = configparser.ConfigParser()
    try:
        config.read('./test_config.cnf')
    except:
        print('No config file!')

    #configuration sections
    default_section = config['DEFAULT']
    preprocessing_section = config['PREPROCESSING']
    clustering_section = config['CLUSTERING']
    forecasting_section = config['FORECASTING']

# =============================================================================
#     Training Section
# =============================================================================
#%% BEGIN OF TRAINING SECTION
    #preprocessing
    sp = SKU_Preprocessor(**{**default_section, **preprocessing_section})
    sp.run()

    #clustering
    sc = SKU_Clusterer(**clustering_section)
    dsts = sc._load_datasets()
    sc.train()

    #forecasting
    sf = SKU_Forecaster(**forecasting_section)
    X, y, columns = sf._load_datasets()
    X_filtered = np.array([sc.filter_dataset(pd.DataFrame(_X, columns = columns)).values for _X in X])
    cluster, cluster_idxs = sc.cluster(X_filtered)
    
    classes = set(cluster_idxs)
    forecasters = {}
    
    for cl in classes:
        mask = cluster_idxs == cl
        
        X_cl = X[mask]
        y_cl = y[mask]
        forecaster = sf.train(X_cl, y_cl, model_name=str(cl))
        forecasters[cl] = forecaster
#%%END OF TRAINING SECTION
# =============================================================================
#     Evaluation Section
# =============================================================================
#%% BEGIN OF EVALUATION SECTION
    # SKU file preparation
    for i in range(10):
        sku_files = os.listdir(default_section['sku_path'])
        chosen_sku = sku_files[i]
        sku = pd.read_csv(os.path.join(default_section['sku_path'], chosen_sku),
                    encoding='cp1252',
                    sep=';')
        n_features = len(sku.columns)
        preprocessor = SKU_Preprocessor(**preprocessing_section)
        sku_prep = preprocessor._remove_columns(sku)
        sku_prep = preprocessor._sanitize_dataset(sku_prep)
        sku_prep = preprocessor.fit_transform(sku_prep, 'test_sku')
        
        
        # assign series to cluster
        sc = SKU_Clusterer(**clustering_section)
        if not sc.load_models():
            print('\nSKU_Clusterer not trained, Aborting...\n')
            sys.exit(-1)
        # choose sequence for forecasting
        input_sequence = sku_prep[262:312]
        label_sequence = sku_prep[312:324][forecasting_section['forecast_column']]
        
        input_sequence_filtered = sc.filter_dataset(input_sequence).values
        _, sequence_class = sc.cluster(input_sequence_filtered.reshape(-1, *input_sequence_filtered.shape))
        sequence_class = sequence_class[0]
        print(f'class of sequence: {sequence_class}')
        
        # get apropriate forecaster for cluster class
        forecaster = forecasters[sequence_class]
        forecast = forecaster.predict(input_sequence.values.reshape(-1, *input_sequence.shape))
    #    plt.plot(forecast.ravel())
    #    plt.plot(label_sequence.ravel())
        
        # inverse transform output to the original space
        original = sku[262:312].reset_index(drop=True)
        label = sku[312:324][forecasting_section['forecast_column']].reset_index(drop=True)
        rescaled = preprocessor.inverse_transform(input_sequence, 'test_sku')
    #    for c in sku.columns:
    #        rescaled[c] += original[c][:1]
    #    for c in sku.columns:
    #        plt.figure()
    #        plt.plot(rescaled[c], label='rescaled')
    #        plt.plot(original[c], label='original')
            
            
        seq_with_forecast = np.append(input_sequence[forecasting_section['forecast_column']], forecast)
        seq_with_forecast_repeated = np.array(np.repeat([seq_with_forecast], n_features, axis=0)).T
        _df = pd.DataFrame(seq_with_forecast_repeated, columns=sku.columns)
        rescaled = preprocessor.inverse_transform(_df, 'test_sku')[forecasting_section['forecast_column']]
        
        const = original[forecasting_section['forecast_column']][0] - rescaled.ravel()[0]
        plt.figure()
        rescaled += const
        plt.plot(rescaled.ravel(), label='predicted_output')
        plt.plot(original[forecasting_section['forecast_column']])
        plt.plot(list(range(50, 62)), label, label='original_output')
        plt.legend()
        plt.title(chosen_sku)
        plt.savefig(f'./screens/{chosen_sku}.png')
        #Error measurement
        mae = metrics.mean_absolute_error(label, rescaled[-12:])
        mse = metrics.mean_squared_error(label, rescaled[-12:])
        mmae = metrics.median_absolute_error(label, rescaled[-12:])
        print(f'SKU: {chosen_sku}')
        error_dict = {'Mean Squared Error':mse,
                      'Mean Absolute Error': mae,
                      'Median Absolute Error': mmae}
        print(error_dict)
        
        plt.figure()
        plt.plot(original[forecasting_section['forecast_column']])
#%% END OF EVALUATION SECION
# =============================================================================
#     Test Section
# =============================================================================
    
#    pred = forecaster.predict(X[0].reshape(-1, 50, 14))
#    plt.figure()
#    plt.plot(pred.ravel(), label='prediction')
#    plt.plot(y[0], label='true')
#    plt.legend()
    
# =============================================================================

#    n_steps = int(forecasting_section['n_steps'])
#    forecast_horizon = int(forecasting_section['forecast_horizon'])
#    seq_len = n_steps + forecast_horizon
#    n_sequences = 100
#    functions = [simple_sine_function, sine_function]
#    #functions = [random_noise, sine_function, simple_sine_function]
#    T = np.linspace(0, 12, num=seq_len)
#    X_train, y_train = generate_dataset(T, functions, n_sequences, offset=forecast_horizon)

#    plt.plot(X_train[0, :, :])
#    plt.plot(list(range(50, 60)), y_train[0, :, :])

#    
#    plt.figure(figsize=(10, 6))
#    plt.plot(sf.forecaster.history.history['loss'])
#    plt.plot(sf.forecaster.history.history['val_loss'])
#    
#    
#    X_test, y_test = generate_dataset(T, functions, 5, offset=forecast_horizon)
#    pred = sf.predict(X_test)
#    
#    plt.figure()
#    plt.plot(y_test[0].ravel())
#    plt.plot(pred[0, :].ravel())
    
    
#    col_name = 'discriminative_col'
#    original = sku_prep[238:288].reset_index(drop=True)
#    predicted = pd.DataFrame(sc.autoenc.predict(original.values.reshape(-1, 50, 9)).reshape(50,9), 
#                             columns=original.columns)
#    plt.plot(original[col_name])
#    plt.plot(predicted[col_name].ravel())

# =============================================================================
    
    sku_files = os.listdir(default_section['sku_path'])
    chosen_sku = sku_files[5]
    dfs = []
    clustered = {0:[], 1:[], 2:[], 3:[], 4:[]}
    sequences = []
    for f in sku_files:
        sku = pd.read_csv(os.path.join(default_section['sku_path'], f),
                    encoding='cp1252',
                    sep=';')
        preprocessor = SKU_Preprocessor(**preprocessing_section)
        sku_prep = preprocessor._remove_columns(sku)
        sku_prep = preprocessor._sanitize_dataset(sku_prep)
        sku_prep = preprocessor.fit_transform(sku_prep, 'test_sku')
        
        # choose sequence for forecasting
        input_sequence = sku_prep[262:312]
        label_sequence = sku_prep[312:324][forecasting_section['forecast_column']]
        
        # assign series to cluster
        input_sequence_filtered = sc.filter_dataset(input_sequence).values
        _, sequence_class = sc.cluster(input_sequence_filtered.reshape(-1, *input_sequence_filtered.shape))
        sequence_class = sequence_class[0]
        print(f'file: {f}, class of sequence: {sequence_class}')
        clustered[sequence_class].append(input_sequence)
        sequences.append(input_sequence)
        
        
    for s in clustered[1]:
        plt.figure()
        plt.plot(s[forecasting_section['forecast_column']], 'r--', alpha=0.5, label='original')
        sf = sc.filter_dataset(s)
        result = sc.autoenc.predict(sf.values.reshape(-1, *sf.shape))
        plt.plot(sf, alpha=0.5, label='predicted')
        plt.legend()
        
    plt.figure(figsize=(10, 7))
    for i, c in zip([0, 1, 2, 3], ['r', 'g', 'b', 'c']):
        for s in clustered[i]:
            sf = sc.filter_dataset(s).values
            plt.plot(sc.predict(sf.reshape(-1, *sf.shape)).ravel(), alpha=0.3, c=c)
            
    plt.figure(figsize=(10, 7))
    for i, c in zip([0, 1, 2, 3], ['r', 'g', 'b', 'c']):
        plt.figure()
        for s in clustered[i]:
            plt.plot(s.values, alpha=0.3, c=c)
            
# =============================================================================
#%% 
#    for f in sku_files:
#        sku = pd.read_csv(os.path.join(default_section['sku_path'], f),
#                    encoding='cp1252',
#                    sep=';')
#        preprocessor = SKU_Preprocessor(**preprocessing_section)
#        sku_prep = preprocessor._remove_columns(sku)
#        sku_prep = preprocessor._sanitize_dataset(sku_prep)
#        sku_prep = preprocessor.fit_transform(sku_prep, 'test_sku')
#        plt.figure(figsize=(10, 7))
#        for c in sku_prep.columns:
#            plt.plot(sku_prep[c])