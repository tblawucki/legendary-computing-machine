#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tslearn.clustering import TimeSeriesKMeans
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, SpectralClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from autoencoder import create_autoencoder_models
from pickle import dump, load
import matplotlib.pyplot as plt
import talos
import sys
from utils import print, generate_color
# =============================================================================
# 2. Datasets Clustering
# =============================================================================

class SKU_Clusterer:
    def __init__(self, *args, **kwargs):
#clustering params
        self.classifier=None
        self.clusters_indices = {}
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
        self.kmeans_path = './models/kmeans_model.pkl'
        self.embedder_path = './models/embedder.pkl'
        self.config_path = './models/clusterer_config.pkl'
#other params
        self.full_dataset = kwargs.get('full_dataset', False)
        self.cold_start = True if kwargs['cold_start'] == 'True' else False
        self.encoding = kwargs.get('encoding', 'utf8')
        self._load_datasets = self._load_datasets_full if self.full_dataset == 'True' else self._load_datasets_partial

        if not self.cold_start:
            self.load_configuration()

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
        
    def load_configuration(self):
        if not os.path.exists(self.config_path):
            print('Config file not found...', verbosity=1)
            return
        config = open(self.config_path, "rb")
        self.clusters_indices = load(config)
        self.n_clusters = load(config)
        self.use_kmeans = load(config)
        self.train_dataset = load(config)
        config.close()
        
    def save_configuration(self):
        config = open(self.config_path, "wb")
        dump(self.clusters_indices, config)
        dump(self.n_clusters, config)
        dump(self.use_kmeans, config)
        dump(self.train_dataset, config)
        config.close()


    def load_models(self, cold_start=False):
        models_exists = os.path.isfile(self.autoencoder_path) \
                        and os.path.isfile(self.encoder_path) \
                        and os.path.isfile(self.decoder_path)
        classifier_exists = os.path.isfile(self.classifier_path)
        kmeans_exists = os.path.isfile(self.kmeans_path)
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
        if not cold_start and kmeans_exists:
            print('K_MEANS MODEL EXISTS, LOADING...')
            with open(self.kmeans_path, 'rb') as model_file:
                self.k_means_classifier = load(model_file)
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
            self.train_dataset = dataset
            classifier_inputs = self.enc.predict(dataset)
            self.embedder = TSNE(n_components=2, perplexity=40, random_state=42)
            embedded = self.embedder.fit_transform(classifier_inputs)
            
            if not self.use_kmeans:
                print('CLUSTER COUNT NOT SPECIFIED, CALCULATING CLUSTER NUMBER...', verbosity=1)
                self.u_classifier = DBSCAN(eps=3, n_jobs=-1)
                classes = self.u_classifier.fit_predict(embedded)
                self.n_clusters = len(set(classes)) 
                self.use_kmeans = True
            self.k_means_classifier = TimeSeriesKMeans(n_clusters=self.n_clusters, 
                                           metric=self.k_means_metric, 
                                           n_init=self.kmeans_iterations,
                                           verbose=True,
                                           max_iter=1000)
            self.k_means_classifier.fit(embedded)
            self.k_means_classifier.transform = self.k_means_classifier.predict #hotfix
            self.clusters_indices = self.k_means_classifier.fit_predict(embedded)
            
            self.classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            self.classifier.fit(embedded, self.clusters_indices)
            
            with open(self.classifier_path, 'wb') as model_file:
                dump(self.classifier, model_file)
            with open(self.embedder_path, 'wb') as model_file:
                dump(self.embedder, model_file)
            with open(self.kmeans_path, 'wb') as model_file:
                dump(self.k_means_classifier, model_file)
            
            self.save_configuration()

# =============================================================================
# Cluster visualisation
# =============================================================================
            clusters = self.k_means_classifier.transform(embedded)
            unique_clusters = set(clusters)
            plt.figure()
            for clas in unique_clusters:
        #    for clas in unique_clusters:
                c = generate_color()
                mask = clusters == clas
                filtered = embedded[mask]
                plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=f'cluster {clas + 1}')
            plt.legend()
            plt.savefig('./clusters/clusters.png')


    def embed(self, dataset):
        flattened = self.enc.predict(dataset)
        embedded = self.embedder.fit_transform(flattened)
        return embedded
        
    def predict(self, sample):
        result = self.enc.predict(sample)
        return result
    
    def predict_class(self, sample, plot_cluster=False):
        extended_dataset = np.vstack(( self.train_dataset, sample.reshape(-1, *sample.shape) ))
        embedded_space = self.embed(extended_dataset)
        sample_coords = embedded_space[-1]
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(embedded_space[:-1])
        distances, indices = nbrs.kneighbors(sample_coords.reshape(1, -1))    
        n_classes, classes_counts = np.unique(self.clusters_indices[indices], return_counts = True)
        cls = n_classes[np.argmax(np.unique(classes_counts))]
        print(distances)
        print(indices)
        print(self.clusters_indices[indices])
        print(cls)
        if plot_cluster:
            plt.figure()
            plt.scatter(embedded_space[:,0], embedded_space[:,1])
            plt.scatter(sample_coords[0], sample_coords[1], marker='x', c='red')
        return cls, distances, indices
    
    def compress_dataset(self, dataset):
        return self.enc.predict(dataset)
    
    def cluster(self, dataset, sample=None, plot_clusters=False):
        if sample is not None:
            dataset = np.vstack((sample, dataset))
        compressed_dataset = self.compress_dataset(dataset)
        embedded_dataset = self.embedder.fit_transform(compressed_dataset)
        classes = self.k_means_classifier.fit_predict(embedded_dataset)
        
        if plot_clusters:
            plt.figure()
            unique_clusters = set(classes)
            for clas in unique_clusters:
                c = generate_color()
                mask = classes == clas
                filtered = embedded_dataset[mask]
                plt.scatter(filtered[:, 0], filtered[:, 1], c=c, label=f'cluster {clas + 1}')
            if sample is not None:
                plt.scatter(embedded_dataset[0, 0], embedded_dataset[0, 1], c='red', marker='x')
            plt.legend()
            
        return dataset, classes
    
    
if __name__ == '__main__':
    from utils import ConfigSanitizer
    
    config = ConfigSanitizer('./test_config.cnf')
    clustering_section = config['CLUSTERING']
    
    cs = SKU_Clusterer(**clustering_section)
    cs.train()
    
    
    for i in range(200, 250):
        sample =cs.train_dataset[i]
        distances, indices = cs.predict_class(sample, plot_cluster=True)
        neighbours = cs.train_dataset[indices].reshape(-1, 50)
        
        plt.figure()
        plt.plot(neighbours.T)
        plt.plot(sample, '--', linewidth=3)

    
#    
#    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
#    
#    A = np.array([(0,0), (0,1), (1,1), (2,1) ,(20,30), (22, 31), (19,32)])
#    Y = np.array([[5],[5],[5],[0],[1],[1],[1]])
#    sample = np.array([(15,15)])

#    knn.fit(A, Y)
#    pred = knn.predict(np.array([(-1, -1)]))
#    print(pred)
#    
#    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(A)
#    distances, indices = nbrs.kneighbors(sample)    
