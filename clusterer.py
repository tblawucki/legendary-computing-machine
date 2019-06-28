#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tslearn.clustering import TimeSeriesKMeans
from autoencoder import create_autoencoder_models
from pickle import dump, load
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
        
    def _load_datasets(self):
        datasets = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding='cp1252',
                                               sep=';')
            n_splits = df.shape[0] // self.n_steps
            trim = df.shape[0] % self.n_steps
            df = df[trim:]
            for split_idx in range(n_splits):
                chunk = df[split_idx * self.n_steps : (split_idx + 1) * self.n_steps]
                datasets.append(chunk.values)
        return np.array(datasets, dtype=np.float64)
                
    def train(self):
        dataset = self._load_datasets()
        n_features = dataset.shape[2]
        models_exists = False
        
        autoencoder_path = './models/autoencoder.pkl'
        encoder_path = './models/encoder.pkl'
        decoder_path = './models/decoder.pkl'
        
        models_exists = os.path.isfile(autoencoder_path) \
                        and os.path.isfile(encoder_path) \
                        and os.path.isfile(decoder_path)
        if not models_exists:
            print('NO MODELS FOUND, COLD START...')
        if not self.cold_start and models_exists:
            print('MODELS EXISTS, LOADING...')
            self.autoenc = load_model('./models/autoencoder.pkl')
            self.enc = load_model('./models/encoder.pkl')
            self.dec = load_model('./models/decoder.pkl')
        else:
            self.autoenc, self.enc, self.dec = create_autoencoder_models(dataset=dataset,
                                                        n_features=n_features,
                                                        n_steps=self.n_steps,
                                                        epochs=self.n_epochs,
                                                        enc_units=self.encoder_output_units,
                                                        dec_units=self.decoder_output_units)
        classifier_inputs = self.enc.predict(dataset)
        classifier_path = './models/kmeans.pkl'
        if not self.cold_start and os.path.isfile(classifier_path):
            print('K_MEANS MODEL EXISTS, LOADING...')
            with open(classifier_path, 'rb') as model_file:
                self.classifier = load(model_file)
        else:
            print('K_MEANS MODEL NOT FOUND, COLD START...')
            self.classifier = TimeSeriesKMeans(n_clusters=self.n_clusters, 
                                           metric="dtw", 
                                           n_init=self.kmeans_iterations,
                                           verbose=True,
                                           max_iter=1000)
            self.classifier.fit(classifier_inputs)
            with open(classifier_path, 'wb') as model_file:
                dump(self.classifier, model_file)
        
    def predict(self, sample):
        result = self.enc.predict(sample)
        return result
    
    def compress_dataset(self, dataset):
        return self.enc.predict(dataset)
    
    def cluster(self, dataset):
        compressed_dataset = self.compress_dataset(dataset)
        return dataset, self.classifier.predict(compressed_dataset)