#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete app for SKU preprocessing, clustering and forecasting
@author: t.blawucki@smartgeometries.pl
@company: SmartGeometries SP. Z.O.O
"""
import os
os.environ['VERBOSITY_LEVEL'] = '0'

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from preprocessor import SKU_Preprocessor
from clusterer import SKU_Clusterer
from forecaster import SKU_Forecaster
import sklearn.metrics as metrics
from utils import print, cleanup, ConfigSanitizer
from io import BytesIO

def load_datasets(sku_path, encoding):
    for file in os.listdir(sku_path):
            datasets = {file: pd.read_csv(os.path.join(sku_path, file),
                                               encoding=encoding,
                                               sep=';')
                        for file in os.listdir(sku_path)}
    return datasets

def load_forecasters(params):
    forecasters = {}
    for i in range(100):
        f = SKU_Forecaster(**params)
        model_exists = f.load_model(model_name=i)
        if not model_exists:
            return forecasters
        forecasters[i] = f
    return forecasters

if __name__ == '__main__':
    config = ConfigSanitizer('./test_config.cnf')
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
        sp = SKU_Preprocessor(**preprocessing_section)
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
#        cluster, cluster_idxs = sc.cluster(X_filtered)
        
        classes = set(sc.clusters_indices)
        forecasters = {}
        
        for cl in classes:
            mask = sc.clusters_indices == cl
            X_cl = X[mask]
            y_cl = y[mask]
            forecaster = sf.train(X_cl, y_cl, model_name=str(cl))
#            forecasters[cl] = forecaster
        
#        cleanup()
    else:
        print('OMITTING TRAINING SECTION...')
#%%
# =============================================================================
#     Evaluation Section - complete workflow
# =============================================================================
    # 1. Preparing dataframes 
    sp = SKU_Preprocessor(**{**default_section, **preprocessing_section})
    sp.run()
    # 2. Preparing clusters, fitting sample
    sc = SKU_Clusterer(**clustering_section)
    if not sc.load_models():
        print('\nSKU_Clusterer model not trained, Aborting...\n', verbosity=2)
        sys.exit(-1)    
    sc.load_models()
    
    full_data = sc._load_datasets()
    np.random.shuffle(full_data)
    full_data = full_data[:2000]
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
            plt.plot(sample.ravel(), 'r--',linewidth=2)
        plt.title(f'CLUSTER {i+1} | N_ELEMENTS: {cluster.shape[0]}')
        
    forecasters = load_forecasters(params=forecasting_section)
    sc.load_configuration()
    
    sku_dict = load_datasets(default_section['sku_output_path'], default_section['encoding'])
    n_windows = range(12)
    sku_partial_forecasts_dict = {}
    sku_image_data_dict = {}
    n_features = 14
    for key, sku in sku_dict.items():
        partial_results = []
        output_data = sku[default_section['forecast_column']][312:324].values.reshape(1, -1) 
        rescaled_true = sp.inverse_transform_seq(output_data.ravel(), key)
        for window_offset in n_windows:
            input_data = sku[262 + window_offset : 312 + window_offset].values
            print(input_data.shape)
#            input_data = input_data.reshape(*input_data.shape)
            input_data_filtered = sc.filter_dataset(pd.DataFrame(input_data, columns=sku.columns)).values
            input_data_cluster, _, _ = sc.predict_class(input_data_filtered.reshape(int(default_section['n_steps']),-1), plot_cluster=False)
            forecaster = forecasters[input_data_cluster]
            pred = forecaster.predict(input_data.reshape(-1, *input_data.shape))[:, -12:, :].reshape(1, -1)
            rescaled_pred = sp.inverse_transform_seq(pred.ravel(), key)
            rescaled_pred_padded = np.r_[[np.nan for _ in range(window_offset)], rescaled_pred]
            partial_results.append(rescaled_pred_padded.ravel())
            if window_offset == 0:
                figure = plt.figure(figsize=(8, 6))
                plt.plot(list(range(312,324)), rescaled_true.ravel(), 'o-', label='oryginał')
                plt.plot(list(range(312,324)), rescaled_pred.ravel(),'o-', label='prognoza')
                plt.grid()
                plt.xlabel('tydzien')
                plt.ylabel(default_section['forecast_column'], )
                plt.title(f'Prognoza dla `{key}`', loc='right')
                plt.legend()
                imgdata = BytesIO()
                figure.savefig(imgdata, format='png')
                sku_image_data_dict[key] = imgdata
        partial_results.append(rescaled_true.ravel())
        sku_partial_forecasts_dict[key] = partial_results
    
    
#%% ReportGenerator evaluation
from utils import generate_report
generate_report(sku_partial_forecasts_dict, sku_image_data_dict)

            

