#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import configparser
import os

def print(*args, verbosity=0, **kwargs):
    print.VERBOSITY_LEVEL = int(os.environ.get('VERBOSITY_LEVEL', '0'))
    from builtins import print as pn
    verb_levels = {0:'INFO', 1:'WARNING', 2:'ERROR', 3:'DEBUG'} 
    if verbosity not in verb_levels.keys():
        raise KeyError(f'Invalid verbosity level, values: {verb_levels}')
    if verbosity >= print.VERBOSITY_LEVEL:
         return pn(f'{verb_levels[verbosity]}:', *args, **kwargs)
    return None

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
            if layer_sizes[0] != len(n_neurons[0]):
                print(layer_sizes[0], n_neurons[0], verbosity=3)
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
    VERBOSITY_LEVEL = 1

    config = configparser.ConfigParser()
    
    config.read('./test_config.cnf')
    
    config_sanitizer = ConfigSanitizer(config_path = './test_config.cnf')
    
    
    sections = config_sanitizer.get_sections()
    
    chosen_section = config_sanitizer['FORECASTING']
    section = 'FORECASTING'
    config_sanitizer._validate_dependencies(section)
    
    print(f'section : {section} | scanning : {config_sanitizer.scanning_required(section)}', verbosity=2)
    

    print('abc', verbosity=3)
    