#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
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

    def fit(self, df, columns):
        pass

    def transform(self, df, columns):
        pass

    def fit_transform(self, df, columns = []):
        if len(columns) == 0:
            self.columns = df.columns
        else:
            self.columns = columns
        for c in self.columns:
            self.fst_elements[c] = df[c][:1]
            df[c][1:] = np.diff(df[c])
#            df[c][0] = 0
        return df.values

    def inverse_transform(self, df):
        n_cols = df.shape[1]
        try:
            for c in range(n_cols):
                df[:, c] = np.cumsum(df[:, c])
            return df
        except TypeError:
            return np.cumsum(df).values
        except:
            print('Invalid input data!')


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
        self.crop_idx = int(crop_idx) if crop_idx != '' else None
        self.drop_cols = drop_cols.split(';') if drop_cols else []
        self.encoding = kwargs.get('encoding', 'utf8')
        self.forecast_col = kwargs['forecast_column']
        self.forecast_col_idx = -1
        self.n_features = 1

    def fit_transform(self, df, df_key):
        self.fit(df, df_key)
        df = self.transform(df, df_key)
        return df

    def fit(self, df, df_key):
        self.transform_ops[df_key] = {}
        self.transform_ops[df_key]['scaler'] = MinMaxScaler()
        self.transform_ops[df_key]['standarizer'] = StandardScaler()
        self.transform_ops[df_key]['trend_remover'] = TrendRemover()

    def transform(self, df, df_key):
        '''
        Main method applying cleaning, removing trend, normalizing
        and scaling dataset through all columns in target dataset
        '''
        cols = df.columns
        self.forecast_col_idx = np.argmax(df.columns == self.forecast_col)
        self.n_features = len(cols)
        trend_remover = self.transform_ops[df_key]['trend_remover']
        standarizer = self.transform_ops[df_key]['standarizer']
        scaler = self.transform_ops[df_key]['scaler']
        if self.remove_trend:
            df = trend_remover.fit_transform(df, df.columns)
        if self.scale:
            df = scaler.fit_transform(df)
        if self.standarize:
            df = standarizer.fit_transform(df)
        return pd.DataFrame(df, columns = cols)

    def inverse_transform(self, df, df_key):
        cols = df.columns
        trend_remover = self.transform_ops[df_key]['trend_remover']
        standarizer = self.transform_ops[df_key]['standarizer']
        scaler = self.transform_ops[df_key]['scaler']
        if self.standarize:
            df = standarizer.inverse_transform(df)
        if self.scale:
            df = scaler.inverse_transform(df)
        if self.remove_trend:
            df = trend_remover.inverse_transform(df)
        return pd.DataFrame(df, columns = cols)

    def inverse_transform_seq(self, seq, key):
        seq = seq.reshape(*seq.shape, -1)
        if seq.shape[-1] != self.n_features and seq.shape[-1] == 1:
            seq = np.repeat(seq, self.n_features).reshape(-1, self.n_features)
        trend_remover = self.transform_ops[key]['trend_remover']
        standarizer = self.transform_ops[key]['standarizer']
        scaler = self.transform_ops[key]['scaler']
        if self.standarize:
            seq = standarizer.inverse_transform(seq)
        if self.scale:
            seq = scaler.inverse_transform(seq)
        if self.remove_trend:
            seq = trend_remover.inverse_transform(seq)
        return seq[:, self.forecast_col_idx]

    def _remove_columns(self, df):
        '''
        Method for string and datetime type columns removal.
        '''
        for c in self.drop_cols:
            try:
                df.drop(columns=c, axis=1, inplace = True)
            except:
                print(f'omiting invalid column: {c}')
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
        df.to_csv(os.path.join(self.sku_output_path, filename),
                  index=False,
                  encoding=self.encoding,
                  sep=';')

    def _sanitize_dataset(self, df):
        if self.crop_idx:
            df = df[:self.crop_idx]
        df = df.fillna(value=0)
        df = df.astype(np.float64)
        return df

    def run(self):
        '''
        Method executing whole preprocessing step.
        '''
        files = os.listdir(self.sku_path)
        for file in files:
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding=self.encoding,
                                               sep=';')
            df = self._remove_columns(df)
            df = self._sanitize_dataset(df)
            df = self.fit_transform(df, file)
#            df = self.inverse_transform(df, file)
            self._save_dataframe(df, file)
            self.dataframes[file] = df
        return self.dataframes
    
    
if __name__ == '__main__':
    from utils import ConfigSanitizer
    import matplotlib.pyplot as plt
    
    config = ConfigSanitizer('./config.cnf')
    
    #configuration sections
    preprocessing_section = config['PREPROCESSING']
    
    
    sp = SKU_Preprocessor(**preprocessing_section)
    sp.run()
    
    sku_key =  list(sp.dataframes.keys())[2]
    df = sp.dataframes[sku_key]
    rescaled_df = sp.inverse_transform(df, sku_key)
    
    true_df = pd.read_csv(os.path.join(preprocessing_section["sku_path"],sku_key), sep=';', encoding=preprocessing_section['encoding'])
    true_df = true_df['zmniejszenie_stanu']
    plt.plot(df.values.ravel(), 'x', alpha=0.3)
    plt.plot(true_df.values.ravel(), '--', alpha=0.3)
    plt.plot(rescaled_df.values.ravel(), 'o', alpha=0.3)
    
    
    output_seq = df['zmniejszenie_stanu'][:12].values
    predicted_seq = output_seq + 0.001
    
    plt.figure()
    plt.plot(output_seq)
    plt.plot(predicted_seq)
    plt.figure()
    plt.plot(true_df[:12])
    
    
    plt.figure()
    rescaled_true = sp.inverse_transform_seq(output_seq, sku_key)
    rescaled_pred = sp.inverse_transform_seq(predicted_seq, sku_key)
    plt.figure()
    plt.plot(rescaled_true + 20, alpha=0.5)
    plt.plot(rescaled_pred + 40, alpha=0.5)
    plt.plot(true_df[:12])

#    plt.plot(true_df['zmniejszenie_stanu'][:12])
    