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


if __name__ == '__main__':
    config = configparser.ConfigParser()
    try:
        config.read('./config.cnf')
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
    statistics = {'file':[], 'mean_squared_error':[], 'mean_absolute_error':[], 'median_absolute_error':[]}

    sc = SKU_Clusterer(**clustering_section)
    if not sc.load_models():
        print('\nSKU_Clusterer model not trained, Aborting...\n')
        sys.exit(-1)


    input_begin = 262
    input_length = 50
    output_length = 12
    output_begin = input_begin + input_length
    
    input_range = range(input_begin, input_begin + input_length)
    output_range = range(output_begin, output_begin + output_length)


    for f in os.listdir(default_section['sku_path']):
        chosen_sku = f
        sku = pd.read_csv(os.path.join(default_section['sku_path'], chosen_sku),
                    encoding=default_section['encoding'],
                    sep=';')
        preprocessor = SKU_Preprocessor(**preprocessing_section)
        sku_prep = preprocessor._remove_columns(sku)
        sku_prep = preprocessor._sanitize_dataset(sku_prep)
        sku_prep = preprocessor.fit_transform(sku_prep, 'test_sku')
        n_features = len(sku_prep.columns)
        

        # choose sequence for forecasting
        input_sequence = sku_prep[input_begin : input_begin + input_length]
        label_sequence = sku_prep[output_begin : output_begin + output_length][forecasting_section['forecast_column']]
#        input_sequence = sku_prep[262:312]
#        label_sequence = sku_prep[312:324][forecasting_section['forecast_column']]
        
        input_sequence_filtered = sc.filter_dataset(input_sequence).values
        _, sequence_class = sc.cluster(input_sequence_filtered.reshape(-1, *input_sequence_filtered.shape))
        sequence_class = sequence_class[0]
        print(f'class of sequence: {sequence_class}')
        
        # get apropriate forecaster for cluster class and make prediction
        forecaster = forecasters[sequence_class]
        forecast = forecaster.predict(input_sequence.values.reshape(-1, *input_sequence.shape))
        
        # inverse transform output to the original space
        original = sku[input_begin : input_begin + input_length].reset_index(drop=True)
        label = sku[output_begin : output_begin + output_length][forecasting_section['forecast_column']].reset_index(drop=True)
#        original = sku[262:312].reset_index(drop=True)
#        label = sku[312:324][forecasting_section['forecast_column']].reset_index(drop=True)
        rescaled = preprocessor.inverse_transform(input_sequence, 'test_sku')
            
        seq_with_forecast = np.append(input_sequence[forecasting_section['forecast_column']], forecast)
        seq_with_forecast_repeated = np.array(np.repeat([seq_with_forecast], n_features, axis=0)).T
        _df = pd.DataFrame(seq_with_forecast_repeated, columns=sku.columns)
        rescaled = preprocessor.inverse_transform(_df, 'test_sku')[forecasting_section['forecast_column']]

        # forecast visualisation
        const = original[forecasting_section['forecast_column']][0] - rescaled.ravel()[0]
        plt.figure()
        rescaled += 0#const
        plt.plot(list(output_range), rescaled.ravel()[-12:], label='predicted_output')
        plt.plot(list(input_range), original[forecasting_section['forecast_column']])
        plt.plot(list(output_range), label, label='original_output')
        plt.grid(linestyle='--')
        plt.ylabel('wartosc')
        plt.xlabel('nr tygodnia')
        plt.legend()
        plt.title(chosen_sku)
        plt.savefig(f'./screens/{chosen_sku}.png')

        # error measurement
        mae = metrics.mean_absolute_error(label, rescaled[-12:])
        mse = metrics.mean_squared_error(label, rescaled[-12:])
        mmae = metrics.median_absolute_error(label, rescaled[-12:])
        
        # saving error measures
        statistics['file'].append(chosen_sku)
        statistics['mean_squared_error'].append(mse)
        statistics['mean_absolute_error'].append(mae)
        statistics['median_absolute_error'].append(mmae)

#   Writing results to excel
    writer = pd.ExcelWriter('./results.xlsx')
    statistics_df = pd.DataFrame.from_dict(statistics)
    statistics_df.to_excel(writer, index=False, sheet_name='results')
    writer.close()


