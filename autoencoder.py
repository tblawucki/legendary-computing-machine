#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Tensorflow 
# from tensorflow import keras
# from tensorflow.keras import Sequential, Model
# from tensorflow.keras.layers import Input, LSTM, GRU, RepeatVector, TimeDistributed, Dense
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import plot_model
# import talos
# =============================================================================

# =============================================================================
# Keras 
import tensorflow.keras as keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, GRU, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
# =============================================================================
from utils import print

def create_autoencoder_models(X_train, y_train, X_val=None, y_val=None, params=None):
    keras.backend.clear_session()   
    # =============================================================================
    #  Model creation
    # =============================================================================
    # ENCODER MODEL
    n_steps = int(params['n_steps'])
    n_features = int(params['n_features'])
    enc_units = int(params['enc_units'])
    dec_units = int(params['dec_units'])
    epochs = int(params['epochs'])
    batch_size = int(params['batch_size'])
    early_stopping = params['early_stopping'] == 'True'
    dataset = X_train
    scan = params['scan']
    
    inputs = Input(shape=(n_steps, n_features))
    encoded_l1 = LSTM(n_steps, return_sequences=True)(inputs, training=True)
    encoded_l2 = LSTM(int((enc_units + n_steps)/2), return_sequences=True)(encoded_l1, training=True)
    encoded = LSTM(int(enc_units), return_sequences=False)(encoded_l2, training=True)
#    encoded = Reshape(target_shape=(enc_units, n_features))(encoded)
    encoder_model = Model(inputs, encoded)
    encoder_model.summary()

    #DECODER MODEL
    dec_inputs = Input(shape=(enc_units, ))
    repeated = RepeatVector(n_steps)(dec_inputs)
    decoded = LSTM(dec_units, return_sequences=True)(repeated, training=True)
    decoded_output = TimeDistributed(Dense(n_features))(decoded)
    decoder_model = Model(dec_inputs, decoded_output)
    decoder_model.summary()

    #AUTOENCODER MODEL    
    autoencoder_input = Input(shape=(n_steps, n_features))
    enc_output = encoder_model(autoencoder_input)
    dec_output = decoder_model(enc_output)
    autoencoder = Model(autoencoder_input, dec_output, name='Autoencoder')

    # =============================================================================
    # Training 
    # =============================================================================

    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto', restore_best_weights=True)
    callbacks = [] if not early_stopping else [es]
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
    hist = autoencoder.fit(dataset, dataset, epochs=epochs, batch_size = batch_size, shuffle=True, validation_split=0.1, callbacks=callbacks)
    if scan:
        return hist, autoencoder#, encoder_model, decoder_model 
    else:
    # =============================================================================
    # Save and return models    
    # =============================================================================
        autoencoder.save('./models/autoencoder.pkl')
        encoder_model.save('./models/encoder.pkl')
        decoder_model.save('./models/decoder.pkl')
    
        plot_model(autoencoder, to_file='./architecture/autoencoder_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')
        plot_model(encoder_model, to_file='./architecture/encoder_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')
        plot_model(decoder_model, to_file='./architecture/decoder_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')
        return autoencoder, encoder_model, decoder_model

#def create_autoencoder_models(dataset, n_steps, n_features, epochs=300, enc_units=100, dec_units=300, color_idx=0, batch_size=32):
#    keras.backend.clear_session()   
#    # =============================================================================
#    #  Model creation
#    # =============================================================================
#    # ENCODER MODEL
#    inputs = Input(shape=(n_steps, n_features))
#    encoded_l1 = LSTM(n_steps, return_sequences=True)(inputs, training=True)
#    encoded_l2 = LSTM(int((enc_units + n_steps)/2), return_sequences=True)(encoded_l1, training=True)
#    encoded = LSTM(int(enc_units), return_sequences=False)(encoded_l2, training=True)
##    encoded = Reshape(target_shape=(enc_units, n_features))(encoded)
#    encoder_model = Model(inputs, encoded)
#    encoder_model.summary()
#
#    #DECODER MODEL
#    dec_inputs = Input(shape=(enc_units, ))
#    repeated = RepeatVector(n_steps)(dec_inputs)
#    decoded = LSTM(dec_units, return_sequences=True)(repeated, training=True)
#    decoded_output = TimeDistributed(Dense(n_features))(decoded)
#    decoder_model = Model(dec_inputs, decoded_output)
#    decoder_model.summary()
#
#    #AUTOENCODER MODEL    
#    autoencoder_input = Input(shape=(n_steps, n_features))
#    enc_output = encoder_model(autoencoder_input)
#    dec_output = decoder_model(enc_output)
#    autoencoder = Model(autoencoder_input, dec_output, name='Autoencoder')
#
#    # =============================================================================
#    # Training 
#    # =============================================================================
#
#    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto', restore_best_weights=True)
#
#    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
#    autoencoder.fit(dataset, dataset, epochs=epochs, batch_size = batch_size, shuffle=True, validation_split=0.1, callbacks=[])
#
#    # =============================================================================
#    # Save and return models    
#    # =============================================================================
#    autoencoder.save('./models/autoencoder.pkl')
#    encoder_model.save('./models/encoder.pkl')
#    decoder_model.save('./models/decoder.pkl')
#
#    plot_model(autoencoder, to_file='./architecture/autoencoder_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')
#    plot_model(encoder_model, to_file='./architecture/encoder_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')
#    plot_model(decoder_model, to_file='./architecture/decoder_arch.png', show_shapes=True, show_layer_names=True, rankdir='TB')
#    
#    return autoencoder, encoder_model, decoder_model 

if __name__ == '__main__':
    from clusterer import SKU_Clusterer
    import configparser
    import talos
    import sys
    
    config = configparser.ConfigParser()
    try:
        config.read('./test_config.cnf')
    except:
        print('No config file!', verbosity=2)
        sys.exit(-1)

    #configuration sections
    clustering_section = config['CLUSTERING']

    dsts = SKU_Clusterer(**clustering_section)._load_datasets()
    params = {
                'n_steps':[50],
                'n_features':[dsts.shape[-1]],
                'epochs':[10, 20],
                'enc_units':[50],
                'dec_units':[300],
                'batch_size':[32],
                'scan':[True],
                'early_stopping':[True]
            }
    results = talos.Scan(dsts, dsts, params=params, model=create_autoencoder_models, debug=True)
    best_params = results.data.sort_values(by=['val_loss'], ascending=True).iloc[0].to_dict()
    best_params['scan'] = False
    best_model = create_autoencoder_models(dsts, dsts, None, None, best_params)