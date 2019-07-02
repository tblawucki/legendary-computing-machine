#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tslearn.clustering import TimeSeriesKMeans
from autoencoder import create_autoencoder_models
from pickle import dump, load
import matplotlib.pyplot as plt
# =============================================================================
# 2. Datasets Clustering
# =============================================================================

class SKU_Clusterer:
    def __init__(self, n_clusters, sku_path, rnn_epochs, n_steps, 
                 encoder_output_units, decoder_output_units, k_means_n_iterations,
                 cold_start, *args, **kwargs):
        self.clusters = {}
        self.n_clusters = int(n_clusters)
        self.sku_path = sku_path
        self.n_epochs = int(rnn_epochs)
        self.n_steps = int(n_steps)
        self.encoder_output_units = int(encoder_output_units)
        self.decoder_output_units = int(decoder_output_units)
        self.kmeans_iterations = int(k_means_n_iterations)
        self.cold_start = True if cold_start == 'True' else False
        self.discriminative_cols = kwargs.get('discriminative_columns', None)
        if self.discriminative_cols: self.discriminative_cols = self.discriminative_cols.strip().split(';')
        self.autoencoder_path = './models/autoencoder.pkl'
        self.encoder_path = './models/encoder.pkl'
        self.decoder_path = './models/decoder.pkl'
        self.classifier_path = './models/kmeans.pkl'
        self.k_means_metric = kwargs.get('k_means_metric', 'euclidean')
        if self.k_means_metric not in ['dtw', 'euclidean', 'softdtw']:
            print('invalid k_means metric, seting to `euclidean`')
            self.k_means_metric = 'euclidean'
            
        
        
    def filter_dataset(self, df):
        chosen_cols = []
        for c in self.discriminative_cols:
                if c not in df.columns:
                    print(f'invalid column name: `{c}`, omitting...')
                else:
                    chosen_cols.append(c)
        self.discriminative_cols = chosen_cols
        if self.discriminative_cols != []:
            print(f'RUNNING FILTERING on columns:{", ".join(self.discriminative_cols)}')
            df = df.filter(items = self.discriminative_cols)
        else:
            print('No discriminative columns passed, running algoritm on all columns')
        return df
        
    def _load_datasets(self):
        datasets = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding='cp1252',
                                               sep=';')
            df = self.filter_dataset(df)
            n_splits = df.shape[0] // self.n_steps
            trim = df.shape[0] % self.n_steps
            df = df[trim:]
            for split_idx in range(n_splits):
                chunk = df[split_idx * self.n_steps : (split_idx + 1) * self.n_steps]
                datasets.append(chunk.values)
        return np.array(datasets, dtype=np.float64)
        
    def load_models(self, cold_start=False):
        models_exists = os.path.isfile(self.autoencoder_path) \
                        and os.path.isfile(self.encoder_path) \
                        and os.path.isfile(self.decoder_path)
        k_means_exists = os.path.isfile(self.classifier_path)
        if not (models_exists and k_means_exists):
            print('NO MODELS FOUND, COLD START REQUIRED...')
        if not cold_start and models_exists:
            print('MODELS EXISTS, LOADING...')
            self.autoenc = load_model(self.autoencoder_path)
            self.enc = load_model(self.encoder_path)
            self.dec = load_model(self.decoder_path)
        if not cold_start and k_means_exists:
            print('K_MEANS MODEL EXISTS, LOADING...')
            with open(self.classifier_path, 'rb') as model_file:
                self.classifier = load(model_file)
        return models_exists and k_means_exists and not cold_start

    def train(self):
        dataset = self._load_datasets()
        n_features = dataset.shape[2]
        print(f'LOAD MODEL RETURNS: {self.load_models(self.cold_start)}')
        if not self.load_models(self.cold_start):
            self.autoenc, self.enc, self.dec = create_autoencoder_models(dataset=dataset,
                                                        n_features=n_features,
                                                        n_steps=self.n_steps,
                                                        epochs=self.n_epochs,
                                                        enc_units=self.encoder_output_units,
                                                        dec_units=self.decoder_output_units)
            hist = self.autoenc.history.history
            loss = hist['loss']
            val_loss = hist['val_loss']
            plt.figure(figsize=(10, 7))
            plt.plot(loss, label='training_loss')
            plt.plot(val_loss, label='validation_loss')
            plt.legend()
            plt.title('Autoencoder loss')
            plt.savefig('./loss/autoencoder_loss.png')
            
            classifier_inputs = self.enc.predict(dataset)
            self.classifier = TimeSeriesKMeans(n_clusters=self.n_clusters, 
                                           metric=self.k_means_metric, 
                                           n_init=self.kmeans_iterations,
                                           verbose=True,
                                           max_iter=1000)
            self.classifier.fit(classifier_inputs)
            with open(self.classifier_path, 'wb') as model_file:
                dump(self.classifier, model_file)

    def predict(self, sample):
        result = self.enc.predict(sample)
        return result
    
    def compress_dataset(self, dataset):
        print(dataset.shape)
        return self.enc.predict(dataset)
    
    def cluster(self, dataset):
        compressed_dataset = self.compress_dataset(dataset)
        print(compressed_dataset.shape)
        return dataset, self.classifier.predict(compressed_dataset)