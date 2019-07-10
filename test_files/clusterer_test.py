#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

import configparser
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessor import SKU_Preprocessor
from clusterer import SKU_Clusterer
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, SpectralClustering, DBSCAN
import tensorflow.keras as K


def sine_function(X):
    return np.sin(X * np.random.uniform(0.5, 1.5)) + 0.2 * np.cos(0.3 * X * np.random.uniform(0.5, 1.5) + np.random.random()) + np.sin(5 * X * np.random.uniform(0.5, 1.5)) + np.random.random()

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
    classes = []
    label_sequences = []
    for i in range(n_sequences):
#        time = time + np.random.uniform()*10
        function = np.random.choice(functions)
        sequence = function(time)
        sequences.append(sequence[:-offset])
        label_sequences.append(sequence[-offset:])
        classes.append(function.__name__)
    return np.array(sequences).reshape(n_sequences, -1, 1), \
           np.array(label_sequences).reshape(n_sequences, -1), \
           np.array(classes)

if __name__ == '__main__':
    K.backend.clear_session()
    config = configparser.ConfigParser()
    try:
        config.read('./config.cnf')
    except:
        print('No config file!')
        sys.exit(-1)

    #configuration sections
    default_section = config['DEFAULT']
    clustering_section = config['CLUSTERING']   
    
    n_steps = int(clustering_section['n_steps'])
    n_sequences = int(clustering_section['n_sequences'])
    functions = [simple_sine_function, sine_function, line_function]
    function_names = [f.__name__ for f in functions]
    time = np.linspace(0, 10, n_steps + 10)
    X, y, cls = generate_dataset(time, functions, n_sequences)
    
    sc = SKU_Clusterer(**clustering_section)
    sc.train(X)
    
    #%% sklearn models
    for i in range(10):
        pred = sc.autoenc.predict(X[i].reshape(-1, *X[i].shape))
        plt.figure()
        plt.plot(X[i].ravel())
        plt.plot(pred.ravel())
        
    plt.figure()
    colors = ['red','green','blue', 'orange']
    for fn, c in zip(function_names, colors):
        mask = cls == fn
        filtered = X[mask]
        for i in range(10):
            encoded = sc.predict(filtered[i].reshape(-1, *filtered[i].shape))
            plt.plot(encoded.ravel(), alpha=0.2, c=c)
    
    plt.figure()
    mask = cls == 'line_function'
    filtered = X[mask]
    plt.plot(filtered.reshape(-1, n_steps).T, alpha=0.3)
    
    flattened = sc.predict(X)
    embedded = TSNE(n_components=2, perplexity=25).fit_transform(flattened)    
    plt.figure()
    for fn, c in zip(function_names, colors):
        mask = cls == fn
        filtered = embedded[mask]
        plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=fn)
    plt.legend()
    
    plt.figure()
    ms = MeanShift(n_jobs=-1)
    clusters = ms.fit_predict(embedded)
    unique_clusters = set(clusters)
    for clas, c in zip(unique_clusters, colors):
        mask = clusters == clas
        filtered = embedded[mask]
        plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=clas)
    plt.legend()
#    
    plt.figure()
    spc = SpectralClustering(n_clusters=3)
    clusters = spc.fit_predict(embedded)
    unique_clusters = set(clusters)
    for clas, c in zip(unique_clusters, colors):
        mask = clusters == clas
        filtered = embedded[mask]
        plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=clas)
    plt.legend()
    
    plt.figure()
    dbs = DBSCAN(n_jobs=-1, eps=3)
    clusters = dbs.fit_predict(embedded)
    unique_clusters = set(clusters)
#    for clas, c in zip(unique_clusters, colors):
    for clas in unique_clusters:
        mask = clusters == clas
        filtered = embedded[mask]
        plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=clas)
    plt.legend()

            
#            encoded = sc.predict(filtered[i].reshape(-1, *filtered[i].shape))
#            plt.plot(encoded.ravel(), alpha=0.2, c=c)