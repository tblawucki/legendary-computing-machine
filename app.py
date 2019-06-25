#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete app for SKU preprocessing, clustering and forecasting
@author: t.blawucki@smartgeometries.pl
@company: SmartGeometries SP. Z.O.O
"""
import configparser
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# =============================================================================
# 1. Preprocess raw SKU files
#   a) Scale to -1 <=> 1
#   b) Standarize datasets
#   c) Remove trend from data
#   d) fill NaN values
#   e) remove text/date columns
#   f) save as new files
# =============================================================================


class TrendRemover:
    def __init__(self):
        self.fst_elements = {}
    
    def fit_transform(self, df, columns = []):
        if len(columns) == 0:
            self.columns = df.columns
        else:
            self.columns = columns
        for c in self.columns:
            self.fst_elements[c] = df[c][:1]
            df[c][1:] = np.diff(df[c])
        return df
        
    def inverse_transform(self, df):
        return np.cumsum(df)
        
    
class SKU_Preprocessor:
    def __init__(self, sku_path, sku_output_path='', remove_trend=False, 
                 standarize=False, scale=False, crop_idx=None,  drop_cols=[], *args, **kwargs):
        '''
        Constructor of SKU Preprocessor class
        parameters:
            sku_path -> path to folder with sku files
            sku_output_path -> target directory where class can save transformed dataset. 
                If None files will overwrite originals.
            remove_trend -> flag indicating trend removal option. Default = False
            standarize -> flag indicating standarization process. Default = False
            scale -> flag indicating scaling process. Default = False
        '''
        self.sku_path = sku_path
        self.sku_output_path = sku_output_path if sku_output_path != '' else sku_path
        self.remove_trend = True if remove_trend == 'True' else False
        self.standarize = True if standarize == 'True' else False
        self.scale = True if scale == 'True' else False
        self.transform_ops = {}
        self.dataframes = {}
        self.crop_idx = int(crop_idx)
        self.drop_cols = drop_cols.split(';')
        print(self.sku_path, 
              self.sku_output_path, 
              self.remove_trend, 
              self.standarize, 
              self.scale,
              self.crop_idx,
              self.drop_cols)
        
        
    def fit_transform(self, df, df_key):
        pass
    
    def fit(self, df, df_key):
        self.transform_ops[df_key] = {}
        self.transform_ops[df_key]['scaler'] = scaler = MinMaxScaler()
        self.transform_ops[df_key]['standarizer'] = standarizer = StandardScaler()
        self.transform_ops[df_key]['trend_remover'] = trend_remover = TrendRemover()

    def transform(self, df, df_key):
        '''
        Main method applying cleaning, removing trend, normalizing 
        and scaling dataset through all columns in target dataset
        '''
        pass

    def inverse_transform(self, df):
        pass

    def _remove_columns(self, df):
        '''
        Method for string and datetime type columns removal.
        '''
        df.drop(columns=self.drop_cols, axis=1, inplace = True)
        return df
    
    def _remove_trend(self, df, operator, columns=None,):
        '''
        Method for removing trend from time series.
        '''
        df = operator.fit_transform(df, columns)
        return df
        
    def _save_dataframe(self, df, filename):
        if not os.path.exists(self.sku_output_path):
            os.mkdir(self.sku_output_path)
        df.to_csv(os.join(self.sku_output_path, filename), 
                  index=False,
                  encoding='cp1252',
                  sep=';')

    def _sanitize_dataset(self, df):
        return df.fillna(value=0)
    
    def run(self):
        '''
        Method executing whole preprocessing step.        
        '''
        files = os.listdir(self.sku_path)
        for file in files:
            self.transform_ops[file] = {}
            self.transform_ops[file]['scaler'] = scaler = MinMaxScaler()
            self.transform_ops[file]['standarizer'] = standarizer = StandardScaler()
            self.transform_ops[file]['trend_remover'] = trend_remover = TrendRemover()
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding='cp1252',
                                               sep=';')[:self.crop_idx]
            df = df.fillna(value=0)
            df_copy = df.copy()
            df = self._remove_columns(df)
            df_cols = df.columns
            df = df.astype(np.float64)
            if self.remove_trend:
                df = trend_remover.fit_transform(df, df.columns)
            if self.standarize:
                df = standarizer.fit_transform(df)
            if self.scale:
                df = scaler.fit_transform(df)
        df = pd.DataFrame(df, columns=df_cols)
        self.dataframes[file] = df
        return self.dataframes 
# =============================================================================
# 2. Datasets Clustering
# =============================================================================

# =============================================================================
# 3. Create LSTM/GRU based models for prediction on clustered time series
# =============================================================================

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
        config.read('./config.cnf')
    except:
        print('No config file!')
    
    default_section = config['DEFAULT']
    preprocessing_section = config['PREPROCESSING']
    clustering_section = config['CLUSTERING']
    forecasting_section = config['FORECASTING']
       
    
    sp = SKU_Preprocessor(**{**default_section, **preprocessing_section})
    dfs = sp.run()
    df = dfs['dane_parametry BLGWK-08-00-1500-KR-S355.csv']
    plt.plot(df, alpha=0.6)
    
    df_cp = df.copy()
    tr = TrendRemover()
#    plt.figure()
#    plt.plot(df, alpha=0.4)
    df = tr.fit_transform(df)
    plt.figure()
    plt.plot(df, alpha=0.4)
    df = tr.inverse_transform(df)
    
    plt.figure(figsize=(10, 10))
    plt.plot(df, alpha=0.4)
    plt.plot(df_cp, alpha=0.4)