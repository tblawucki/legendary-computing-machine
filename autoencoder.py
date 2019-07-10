#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, GRU, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


def create_autoencoder_models(dataset, n_steps, n_features, epochs=300, enc_units=100, dec_units=300, color_idx=0, batch_size=32):
    keras.backend.clear_session()   
    # =============================================================================
    #  Model creation
    # =============================================================================
    # ENCODER MODEL
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

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
    autoencoder.fit(dataset, dataset, epochs=epochs, batch_size = batch_size, shuffle=True, validation_split=0.1, callbacks=[es])

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
