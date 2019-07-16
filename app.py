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


def generate_report():
    pass

if __name__ == '__main__':
    config = configparser.ConfigParser()
    try:
        config.read('./test_config.cnf')
    except:
        print('No config file!')
        sys.exit(-1)

    #configuration sections
    default_section = config['DEFAULT']
    preprocessing_section = config['PREPROCESSING']
    clustering_section = config['CLUSTERING']
    forecasting_section = config['FORECASTING']

# =============================================================================
#     Training Section
# =============================================================================
#%% BEGIN OF TRAINING SECTION
    if default_section['train_models'] == 'True': 
        print('RUNNING TRAINING SECTION...')
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
        y = y.reshape(*y.shape, 1)
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
    else:
        print('OMITTING TRAINING SECTION...')
#%%
# =============================================================================
#     Evaluation Section - complete workflow
# =============================================================================
    # 1. Preparing dataframes 
    sp = SKU_Preprocessor(**{**default_section, **preprocessing_section})
#    sp.run()
    # 2. Preparing clusters, fitting sample
    sc = SKU_Clusterer(**clustering_section)
    if not sc.load_models():
        print('\nSKU_Clusterer model not trained, Aborting...\n')
        sys.exit(-1)    
        
    full_data = sc._load_datasets()
    np.random.shuffle(full_data)
    full_data = full_data[:1000]
    sample = full_data[-4:-3]
    _, clusters_tmp = sc.cluster(full_data, plot_clusters=True)
    extended_data, clusters = sc.cluster(full_data, sample, plot_clusters=True)
        
    
    plt.figure()
    plt.plot(sample.ravel(),'--')
    plt.plot(full_data[-4: -3].ravel())
    
    for i in range(len(set(clusters))):
        plt.figure(figsize=(10, 7))
        cluster = extended_data[clusters == i]
        print('cluster_len:', len(cluster))
        for s in cluster:
            plt.plot(s.ravel(), alpha=0.3)
        if clusters[0] == i:
            plt.plot(sample.ravel(), 'r--')
        
        

##%% BEGIN OF EVALUATION SECTION
#    # SKU file preparation
#    statistics = {'file':[], 'mean_squared_error':[], 'mean_absolute_error':[], 'median_absolute_error':[]}
#
#    sc = SKU_Clusterer(**clustering_section)
#    if not sc.load_models():
#        print('\nSKU_Clusterer model not trained, Aborting...\n')
#        sys.exit(-1)
#
#
#    input_begin = 262
#    input_length = int(forecasting_section['n_steps'])
#    output_length = int(forecasting_section['forecast_horizon'])
#    output_begin = input_begin + input_length
#    
#    input_range = range(input_begin, input_begin + input_length)
#    output_range = range(output_begin, output_begin + output_length)
#    prediction_range = range(input_begin + output_length, input_begin + input_length + output_length)
#
#    for f in os.listdir(default_section['sku_path']):
#        chosen_sku = f
#        sku = pd.read_csv(os.path.join(default_section['sku_path'], chosen_sku),
#                    encoding=default_section['encoding'],
#                    sep=';')
#        preprocessor = SKU_Preprocessor(**preprocessing_section)
#        sku_prep = preprocessor._remove_columns(sku)
#        sku_prep = preprocessor._sanitize_dataset(sku_prep)
#        sku_prep = preprocessor.fit_transform(sku_prep, 'test_sku')
#        n_features = len(sku_prep.columns)
#        
#
#        # choose sequence for forecasting
#        input_sequence = sku_prep[input_begin : input_begin + input_length]
#        label_sequence = sku_prep[output_begin : output_begin + output_length][forecasting_section['forecast_column']]
##        input_sequence = sku_prep[262:312]
##        label_sequence = sku_prep[312:324][forecasting_section['forecast_column']]
#        
#        
#        input_sequences_filtered = sc.filter_dataset(sku_prep).values
#        _, sequence_class = sc.cluster(input_sequence_filtered.reshape(-1, *input_sequence_filtered.shape))
#        sequence_class = sequence_class[0]
#        print(f'class of sequence: {sequence_class}')
#        
#        # get apropriate forecaster for cluster class and make prediction
#        forecaster = forecasters[sequence_class]
#        forecast = forecaster.predict(input_sequence.values.reshape(-1, *input_sequence.shape))
#        
#        # inverse transform output to the original space
#        original = sku[input_begin : input_begin + input_length].reset_index(drop=True)
#        label = sku[output_begin : output_begin + output_length][forecasting_section['forecast_column']].reset_index(drop=True)
##        original = sku[262:312].reset_index(drop=True)
##        label = sku[312:324][forecasting_section['forecast_column']].reset_index(drop=True)
#        rescaled = preprocessor.inverse_transform(input_sequence, 'test_sku')
#            
##        seq_with_forecast = np.append(input_sequence[forecasting_section['forecast_column']], forecast)
#        seq_with_forecast = forecast.ravel()
#        seq_with_forecast_repeated = np.array(np.repeat([seq_with_forecast], n_features, axis=0)).T
#        _df = pd.DataFrame(seq_with_forecast_repeated, columns=sku.columns)
#        rescaled = preprocessor.inverse_transform(_df, 'test_sku')[forecasting_section['forecast_column']]
#
#        # forecast visualisation
#        const = original[forecasting_section['forecast_column']][0] - rescaled.ravel()
#        plt.figure()
#        rescaled += 0#const
#        plt.plot(list(prediction_range), rescaled.ravel(), label='predicted_output')
#        plt.plot(list(input_range), original[forecasting_section['forecast_column']])
#        plt.plot(list(output_range), label, label='original_output')
#        plt.grid(linestyle='--')
#        plt.ylabel('wartosc')
#        plt.xlabel('nr tygodnia')
#        plt.legend()
#        plt.title(chosen_sku)
#        plt.savefig(f'./screens/{chosen_sku}.png')
#
#        # error measurement
#        mae = metrics.mean_absolute_error(label, rescaled[-12:])
#        mse = metrics.mean_squared_error(label, rescaled[-12:])
#        mmae = metrics.median_absolute_error(label, rescaled[-12:])
#        
#        # saving error measures
#        statistics['file'].append(chosen_sku)
#        statistics['mean_squared_error'].append(mse)
#        statistics['mean_absolute_error'].append(mae)
#        statistics['median_absolute_error'].append(mmae)
#
##   Writing results to excel
#    writer = pd.ExcelWriter('./results.xlsx')
#    statistics_df = pd.DataFrame.from_dict(statistics)
#    statistics_df.to_excel(writer, index=False, sheet_name='results')
#    writer.close()
#
#
## =============================================================================
##%% Clusterer test 
## =============================================================================
#
#plt.figure()
#sample = input_sequence_filtered.reshape(-1, 50, 6)
#out = sc.autoenc.predict(sample)
#plt.plot(out[0, :, 0].ravel())
#plt.plot(sample[0, :, 0].ravel())
#
#
#
#
#
#
#
#
##%%
#plt.figure()
#def generate_color(min = 75, max = 200):
#    for i in range(10):
#        r = str(hex(np.random.randint(min, max))[2:])
#        g = str(hex(np.random.randint(min, max))[2:])
#        b = str(hex(np.random.randint(min, max))[2:])
#        r, g, b = [_c if len(_c) == 2 else f'0{_c}' for _c in [r,g,b] ]
#        return f'#{r}{g}{b}'
#
#print(r, g, b, c)
#_x = np.array([1,2,2,3,4]) + np.random.randint(2, 10)
#plt.plot(_x, c=c)
##%%
#
#for i in range(3):
#    plt.figure()
#    embedded = sc.embed(X_filtered)
#    plt.scatter(embedded[:, 0], embedded[:, 1])