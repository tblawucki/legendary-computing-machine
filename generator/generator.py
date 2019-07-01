#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# =============================================================================
# Założenia:
#   przygotować zbiór 20 plików po 326 punktów podobnych do SKU
#   każdy dataset jest wielowymiarowy (około 14 wymiarów)
#   
# =============================================================================
# =============================================================================
# Dataset generation code
# =============================================================================
def sine_function(X):
    return np.sin(X + np.random.random()) + 0.2 * np.cos(0.3 * X + np.random.random()) + np.sin(5 * X) + np.random.random()

def sine_function_without_random(X):
    return np.sin(X) + 0.2 * np.cos(0.3 * X) + np.sin(5 * X)

def simple_sine_function(X):
    return np.sin(X + np.random.random()) * np.random.uniform(0.5, 1)

def simple_sine_function_without_random(X):
    return np.sin(X)

def line_function(X):
    return 0*X + np.random.random()

def line_function_without_random(X):
    return 0*X + 0.45

def random_noise(X):
    return np.random.uniform(size=X.ravel().shape[0])

def generate_dataset(time, functions, offset=10):
    sequences = []
    columns = []
    cl = np.random.choice(functions)
    cl_name = cl.__name__
    time = time + np.random.random()*10
    for f in functions:
        columns.append(f.__name__)
        sequences.append(f(time))
    columns.append('discriminative_col')
    sequences.append(cl(time))
    return np.array(sequences).T, columns, cl_name

if __name__ == '__main__':
    functions = [sine_function,
                 sine_function_without_random,
                 simple_sine_function,
                 simple_sine_function_without_random,
                 line_function,
                 line_function_without_random,
                 random_noise]
    t = np.linspace(0, 12, num=326)
    for f in os.listdir('./datasets/'):
        os.remove(os.path.join('./datasets/', f))
    for i in range(20):
        ds, cols, cl = generate_dataset(t, functions)
        df = pd.DataFrame(ds, columns=cols)
        df.to_csv(f'./datasets/dts_{i}_class_{cl}.csv', sep=';', index=True, index_label='trend_col', encoding='cp1252')