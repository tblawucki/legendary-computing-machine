#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from tslearn.clustering import TimeSeriesKMeans
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, SpectralClustering, DBSCAN
from autoencoder import create_autoencoder_models
from pickle import dump, load
import matplotlib.pyplot as plt
import talos
import sys
from utils import print
# =============================================================================
# 2. Datasets Clustering
# =============================================================================

def generate_color(min = 75, max = 200):
    for i in range(10):
        r = str(hex(np.random.randint(min, max))[2:])
        g = str(hex(np.random.randint(min, max))[2:])
        b = str(hex(np.random.randint(min, max))[2:])
        r, g, b = [_c if len(_c) == 2 else f'0{_c}' for _c in [r,g,b] ]
        return f'#{r}{g}{b}'

class SKU_Clusterer:
    def __init__(self, *args, **kwargs):
#clustering params
        self.clusters = {}
        self.n_clusters = int(kwargs['n_clusters'])
        self.use_kmeans = self.n_clusters > 0
        self.kmeans_iterations = int(kwargs['k_means_n_iterations'])
        self.k_means_metric = kwargs.get('k_means_metric', 'euclidean')
        if self.k_means_metric not in ['dtw', 'euclidean', 'softdtw']:
            print('invalid k_means metric, seting to `euclidean`', verbosity=1)
            self.k_means_metric = 'euclidean'
#RNN params
        self.n_epochs = [int(p) for p in (kwargs['rnn_epochs'].split(';'))]
        self.n_steps = [int(p) for p in (kwargs['n_steps']).split(';')][0]
        self.encoder_output_units = [int(p) for p in kwargs['encoder_output_units'].split(';')]
        self.decoder_output_units = [int(p) for p in kwargs['decoder_output_units'].split(';')]
        self.batch_size = [int(p) for p in kwargs['batch_size'].split(';')]
        self.early_stopping = [kwargs['early_stopping'].split(';')]
        self.discriminative_cols = kwargs.get('discriminative_columns', None)
        if self.discriminative_cols: self.discriminative_cols = self.discriminative_cols.strip().split(';')
#paths
        self.sku_path = kwargs['sku_path']
        self.autoencoder_path = './models/autoencoder.pkl'
        self.encoder_path = './models/encoder.pkl'
        self.decoder_path = './models/decoder.pkl'
        self.classifier_path = './models/classifier.pkl'
        self.embedder_path = './models/embedder.pkl'
#other params
        self.full_dataset = kwargs.get('full_dataset', False)
        self.cold_start = True if kwargs['cold_start'] == 'True' else False
        self.encoding = kwargs.get('encoding', 'utf8')
        self._load_datasets = self._load_datasets_full if self.full_dataset == 'True' else self._load_datasets_partial

    def filter_dataset(self, df):
        chosen_cols = []
        for c in self.discriminative_cols:
                if c not in df.columns:
                    print(f'invalid column name: `{c}`, omitting...', verbosity=1)
                else:
                    chosen_cols.append(c)
        self.discriminative_cols = chosen_cols
        if self.discriminative_cols != []:
            print(f'RUNNING FILTERING on columns:{", ".join(self.discriminative_cols)}')
            df = df.filter(items = self.discriminative_cols)
        else:
            print('No discriminative columns passed, running algoritm on all columns')
        return df
        
    def _load_datasets_partial(self):
        datasets = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding=self.encoding,
                                               sep=';')
            df = self.filter_dataset(df)
            n_splits = df.shape[0] // self.n_steps
            trim = df.shape[0] % self.n_steps
            df = df[trim:]
            for split_idx in range(n_splits):
                chunk = df[split_idx * self.n_steps : (split_idx + 1) * self.n_steps]
                datasets.append(chunk.values)
        return np.array(datasets, dtype=np.float64)
    
    def _load_datasets_full(self):
        datasets = []
        for file in os.listdir(self.sku_path):
            df = pd.read_csv(os.path.join(self.sku_path, file),
                                               encoding=self.encoding,
                                               sep=';')
            df = self.filter_dataset(df)
            for offset in range(df.shape[0] - self.n_steps):
                chunk = df[offset : offset + self.n_steps]
                datasets.append(chunk.values)
        return np.array(datasets, dtype=np.float64)
        
    def load_models(self, cold_start=False):
        models_exists = os.path.isfile(self.autoencoder_path) \
                        and os.path.isfile(self.encoder_path) \
                        and os.path.isfile(self.decoder_path)
        classifier_exists = os.path.isfile(self.classifier_path)
        embedder_exists = os.path.isfile(self.embedder_path)
        if not (models_exists and classifier_exists):
            print('NO MODELS FOUND, COLD START REQUIRED...', verbosity=1)
        if not cold_start and models_exists:
            print('AUTOENCODER MODELS EXISTS, LOADING...')
            self.autoenc = load_model(self.autoencoder_path)
            self.enc = load_model(self.encoder_path)
            self.dec = load_model(self.decoder_path)
        if not cold_start and classifier_exists:
            print('CLASSIFIER MODEL EXISTS, LOADING...')
            with open(self.classifier_path, 'rb') as model_file:
                self.classifier = load(model_file)
        if not cold_start and embedder_exists:
            with open(self.embedder_path, 'rb') as model_file:
                self.embedder = load(model_file)
        return models_exists and classifier_exists and embedder_exists and not cold_start

    def train(self, dataset=None):
        if dataset is None:
            dataset = self._load_datasets()
        n_features = dataset.shape[-1]
        if not self.load_models(self.cold_start):
            #Talos scan
            params = {
                        'n_steps':[self.n_steps],
                        'n_features':[n_features],
                        'epochs':self.n_epochs,
                        'enc_units':self.encoder_output_units,
                        'dec_units':self.decoder_output_units,
                        'batch_size':self.batch_size,
                        'early_stopping':self.early_stopping,
                        'scan':[True]
                    }
            results = talos.Scan(dataset, np.zeros_like(dataset), params=params, model=create_autoencoder_models)
            best_params = results.data.sort_values(by=['val_loss'], ascending=True).iloc[0].to_dict()
            best_params['scan'] = False
            print('\n', '='*30,
                  '\nBEST AUTOENCODER HYPERPARAMETERS:\n', 
                  '\n'.join([f'{key} = {value}' for key,value in best_params.items()]),
                  '\n',
                  '='*30)
            self.autoenc, self.enc, self.dec = create_autoencoder_models(dataset, np.zeros_like(dataset), params=best_params)
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
            self.embedder = TSNE(n_components=2, perplexity=40, random_state=42)
            embedded = self.embedder.fit_transform(classifier_inputs)
            if self.use_kmeans:
                plt.figure()
                self.classifier = TimeSeriesKMeans(n_clusters=self.n_clusters, 
                                               metric=self.k_means_metric, 
                                               n_init=self.kmeans_iterations,
                                               verbose=True,
                                               max_iter=1000)
                self.classifier.fit(embedded)
                self.classifier.transform = self.classifier.predict #hotfix
            else:
                print('CLUSTER COUNT NOT SPECIFIED, UNSUPERVISED CLUSTERING...', verbosity=1)
#                self.classifier = MeanShift(n_jobs=-1)
                self.classifier = DBSCAN(eps=3, n_jobs=-1)
                self.classifier.transform = self.classifier.fit_predict
            with open(self.classifier_path, 'wb') as model_file:
                dump(self.classifier, model_file)
            with open(self.embedder_path, 'wb') as model_file:
                dump(self.embedder, model_file)

# =============================================================================
# Cluster visualisation
# =============================================================================
            clusters = self.classifier.transform(embedded)
            unique_clusters = set(clusters)
            plt.figure()
            for clas in unique_clusters:
        #    for clas in unique_clusters:
                c = generate_color()
                mask = clusters == clas
                filtered = embedded[mask]
                plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=clas)
            plt.legend()
            plt.savefig('./clusters/clusters.png')


    def embed(self, dataset):
        flattened = self.enc.predict(dataset)
        embedded = self.embedder.fit_transform(flattened)
        return embedded
        
    def predict(self, sample):
        result = self.enc.predict(sample)
        return result
    
    def compress_dataset(self, dataset):
        return self.enc.predict(dataset)
    
    def cluster(self, dataset, sample=None, plot_clusters=False):
        if sample is not None:
            dataset = np.vstack([sample, dataset])
        compressed_dataset = self.compress_dataset(dataset)
        embedded_dataset = self.embedder.fit_transform(compressed_dataset)
        classes = self.classifier.fit_predict(embedded_dataset)
        
        if plot_clusters:
            plt.figure()
            unique_clusters = set(classes)
            for clas in unique_clusters:
                c = generate_color()
                mask = classes == clas
                filtered = embedded_dataset[mask]
                plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=clas)
            if sample is not None:
                plt.scatter(embedded_dataset[0, 0], embedded_dataset[0, 1], c='red', marker='x')
            plt.legend()
            
        return dataset, classes
    
    
if __name__ == '__main__':
    import configparser
    
    config = configparser.ConfigParser()
    try:
        config.read('./test_config.cnf')
    except:
        print('No config file!', verbosity=2)
        sys.exit(-1)

    #configuration sections
    clustering_section = config['CLUSTERING']
    
    cs = SKU_Clusterer(**clustering_section)
    cs.train()
    