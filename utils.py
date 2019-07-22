#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import configparser
import os
import numpy as np 
import pandas as pd

def print(*args, verbosity=0, **kwargs):
    from builtins import print as pn
    VERBOSITY_LEVEL = int(os.environ.get('VERBOSITY_LEVEL', '0'))
    verb_levels = {0:'INFO', 1:'WARNING', 2:'ERROR', 3:'DEBUG'} 
    if verbosity not in verb_levels.keys():
        raise KeyError(f'Invalid verbosity level, values: {verb_levels}')
    if verbosity >= VERBOSITY_LEVEL:
        return pn(f'{verb_levels[verbosity]}:', *args, **kwargs)

def cleanup(path='./', substring  = '_.csv'):
    csv_files = [f for f in os.listdir('./') if substring in f]
    print('Files to remove: ', csv_files, verbosity=1)
    if len(csv_files) > 0:
        proceed = input('proceed? y/[n]\t') or 'n'
    else:
        print('nothing to do...')
        return
    if proceed == 'y':
        for f in csv_files:
            os.remove(os.path.join(path, f))
            
def generate_color(min = 75, max = 200):
    for i in range(10):
        r,g,b = np.random.uniform(low=min, high=max, size=3).astype(np.int32)
        r = str(hex(r)[2:])
        g = str(hex(g)[2:])
        b = str(hex(b)[2:])
        r, g, b = [_c if len(_c) == 2 else f'0{_c}' for _c in [r,g,b] ]
        return f'#{r}{g}{b}'

def mean_absolute_precentage_error(y_true, y_pred):
    if type(y_true) is not np.array:
        y_true = np.array(y_true, dtype=np.float64)
    if type(y_pred) is not np.array:
        y_pred = np.array(y_pred, dtype=np.float64)
        
    y_true_zeros_mask = y_true == 0
    y_pred_zeros_mask = y_pred == 0
    y_true[y_true_zeros_mask] = 0.000001
    y_pred[y_pred_zeros_mask] = 0.000001
    print(y_true)
    print(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def weighted_mean_absolute_precentage_error(y_true, y_pred):
    if type(y_true) is not np.array:
        y_true = np.array(y_true, dtype=np.float64)
    if type(y_pred) is not np.array:
        y_pred = np.array(y_pred, dtype=np.float64)
        
    y_true_zeros_mask = y_true == 0
    y_pred_zeros_mask = y_pred == 0
    y_true[y_true_zeros_mask] = 0.000001
    y_pred[y_pred_zeros_mask] = 0.000001
    return (np.sum(np.abs(y_true-y_pred)/y_true * 100 * y_true))/np.sum(y_true)

def generate_report(sku_forecasts, sku_images):
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('./reports/raport_12_tygodniowy.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    summary_dict = {'SKU':[],
                    'Średnia rzeczywista \n(12 tygodni)':[], 
                    'Średnia z prognozy \n(12 tygodni)':[],
                    'Odchylenie [%] \n(12 tygodni)':[], 
                    'Średnia rzeczywista \n(4 tygodnie)':[], 
                    'Średnia z prognozy \n(4 tygodnie)':[],
                    'Odchylenie [%] \n(4 tygodnie)':[]}

    pd.DataFrame([]).to_excel(writer, sheet_name='1.Podsumowanie')
    for sku_key, sku_predictions in sku_forecasts.items():
        df = pd.DataFrame.from_dict(sku_predictions)
        imgdata = sku_images[sku_key]
        sheet_name = sku_key[15:-4][:31]
        
        df_true_values = df.values[-1][:12]
        df_predictions = df.values[:-1][:, :df_true_values.shape[0]]
        
        df_devs_values = ((df_predictions - df_true_values)/df_true_values) * 100
        df_devs_values = df_devs_values.round(4)
        df_devs = pd.DataFrame(df_devs_values)
        df_devs_cols = [(x, y) for x in ['Odchylenie prognoz od wartości rzeczywistych [%]'] for y in df_devs.columns]
        df_devs_cols = pd.MultiIndex.from_tuples(df_devs_cols)
        df_devs = pd.DataFrame(df_devs_values, columns=df_devs_cols)
        
        df = df.round(5)
        df_cols = [(x, y) for x in [f'Prognoza krocząca na 12 tygodni dla {sku_key}'] for y in df.columns]
        df.columns = pd.MultiIndex.from_tuples(df_cols)
        df.to_excel(writer, sheet_name=sheet_name, index=True, index_label='Tydzień')
        df_devs.to_excel(writer, sheet_name=sheet_name, index=True, index_label='Tydzień', startrow=18)
        
        worksheet = writer.sheets[sheet_name]
        worksheet.insert_image('A35', 'tmp.png', {'image_data':imgdata})

        mean_true = df.iloc[0].mean()
        mean_pred = df.iloc[-1].mean()
        if mean_true == 0:
            mean_true = 0.00001
        pred_deviation = ((mean_true - mean_pred) / mean_true) * 100
        
        mean_true_4_weeks = df.iloc[0][:4].mean()
        mean_pred_4_weeks = df.iloc[-1][:4].mean()
        if mean_true_4_weeks == 0:
            mean_true_4_weeks = 0.00001
        pred_deviation_4_weeks = ((mean_true_4_weeks - mean_pred_4_weeks) / mean_true_4_weeks) * 100
        
        summary_dict['SKU'].append(sku_key)
        summary_dict['Średnia rzeczywista \n(12 tygodni)'].append(mean_true)
        summary_dict['Średnia z prognozy \n(12 tygodni)'].append(mean_pred)
        summary_dict['Odchylenie [%] \n(12 tygodni)'].append(pred_deviation)
        summary_dict['Średnia rzeczywista \n(4 tygodnie)'].append(mean_true_4_weeks)
        summary_dict['Średnia z prognozy \n(4 tygodnie)'].append(mean_pred_4_weeks)
        summary_dict['Odchylenie [%] \n(4 tygodnie)'].append(pred_deviation_4_weeks)
        
        
    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.set_index('SKU', inplace=True, drop=True)
    summary_df = summary_df.round(2)
    summary_cols = [(x, y) for x in ['Zestawienie wyników prognoz'] for y in summary_df.columns]
#    summary_cols = summary_cols[:int(len(summary_cols)/2)]
    summary_df.columns = pd.MultiIndex.from_tuples(summary_cols)
    summary_df.to_excel(writer, sheet_name='1.Podsumowanie', index=True)
#    text_fmt = writer.book.add_format({'align': 'left', 'num_format': '$#,##0',
#                                 'bold': True, 'bottom':6})
    worksheet = writer.sheets['1.Podsumowanie']
    worksheet.set_column('A:A', 50)
    worksheet.set_column('B:G', 25)
    worksheet.set_row(1, 40)
    writer.save()

class ConfigurationError(Exception):
    pass

class ActionNotAllowedError(Exception):
    pass


class ConfigSanitizer():
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        config_files = self.config.read(config_path)
        if config_files == []: 
            raise ConfigurationError('Configuration file not found!')
        self._sanitize_sections()
    
    def _validate_dependencies(self, section_key):
        if section_key == 'FORECASTING':
            section = self[section_key]
            layer_sizes = []
            n_neurons = [[int(el) for el in l.split('|')] for l in [p for p in section['n_neurons'].split(';')]]
            n_layers = [int(p) for p in section['n_layers'].split(';')]
            print(n_neurons, verbosity=3)
            print(n_layers, verbosity=3)
            for layer in n_neurons:
                layer_sizes.append(len(layer))
            if len(n_layers) > 1 and layer_sizes[0] > 1:
                raise ConfigurationError('Invalid `n_layers` and `n_neurons` parameters dependency')
            if len(set(layer_sizes)) > 1:
                raise ConfigurationError('Inconsistent `n_neurons` parameter values')
            if n_layers[0] != len(n_neurons[0]) and len(n_neurons[0]) > 1:
                raise ConfigurationError('Parameters `n_neurons` and `n_layers` did not match')

            
    def _validate_types(self, line_key, line):
        '''
        Chacking types of elements in options
        available types: str, int, float
        '''
        element_types = []
        for element in line:
            try:
                int(element)
                element_types.append(int)
            except:
                element_types.append(str)
            if len(set(element_types)) > 1:
                raise ValueError(f'Inconsistent value types in line: `{line_key} = {";".join(line)}`\n')
        return line
    
    def scanning_required(self, section_key):
        scanning_req = False
        section = self.config[section_key]
        for key, value in section.items():
            param = value.split(';')
            if len(param) > 1:
                scanning_req = True
                print(f'Scanning required for param `{key} = {value}`', verbosity=0)
        return scanning_req

    def _sanitize_lines(self, section):
            for line_key, line in section.items():
                line = line.strip()
                line = set([p for p in line.split(';') if p != ''])
                line = self._validate_types(line_key, line)
                section[line_key] = ';'.join(line)
    
    def _sanitize_sections(self):
        for section_key, section in self.config.items():
            self._sanitize_lines(section)
            self._validate_dependencies(section_key)
            
    def get_sections(self):
        return {key:dict(self.config[key]) for key in self.config.keys()}
    
    def __getitem__(self, key):
        return dict(self.config[key])
    
    def __setitem__(self, key, value):
        raise ActionNotAllowedError('Config object is ReadOnly')
        
    def __delitem__(self, key):
        raise ActionNotAllowedError('Config object is ReadOnly')


if __name__ == '__main__':
    
#    generate_report()
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    # Create a Pandas dataframe from some data.
    df = pd.DataFrame({'Data': [10, 20, 30, 20, 15, 30, 45]})
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('./reports/pandas_image.xlsx', engine='xlsxwriter')
    
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    
    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Insert an image.
    imgdata = BytesIO()
    fig = plt.figure()
    plt.plot([1,2,3,4,4])
    fig.savefig(imgdata, format='png')
    worksheet.insert_image('D3', 'tmp.png', {'image_data':imgdata})
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    cols_tuples = [(x, y) for x in ['A'] for y in ['X','Y','Z','I']]
    cols = pd.MultiIndex.from_tuples(cols_tuples)
    pd.DataFrame(np.random.randn(10, 4), columns=cols)

    df_1 = np.array([1,2,3,4,5,6,6])
    df_2 = np.array([[np.nan, np.nan, np.nan, 1,2,3,4], [np.nan, np.nan, np.nan, np.nan ,2,3,4]])
    
    ((df_2 - df_1)/df_1) * 100
    