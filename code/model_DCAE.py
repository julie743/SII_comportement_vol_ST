#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:39:39 2022

@author: julie
"""

import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import numpy as np

import tensorflow
tensorflow.__version__

from data_preparation import main_4D_tensor_DCAE
from set_path import DATA_PATH

# Loading the 4D tensor for the training and test set :
tensor4D_train,tensor4D_test = main_4D_tensor_DCAE(DATA_PATH+'/Train_clean_datasets',DATA_PATH+'/Test_clean_datasets')

# Input shape :
nb_TS, window_size, nb_features, nb_windows = tensor4D_train.shape


# encoder ---------------------------------------------------------------------
conv_encoder = km.Sequential(name='conv_encoder')
conv_encoder.add(kl.Conv2D(nb_windows, (10, 1), activation = 'relu', input_shape=(window_size, nb_features, nb_windows), padding = 'same'))
# rk : input shape is of format : (rows, cols, channels) 
conv_encoder.add(kl.Conv2D(64, (10, 1), activation = 'relu', padding = 'same'))
conv_encoder.add(kl.Conv2D(128, (10, 1), activation = 'relu', padding = 'same'))
conv_encoder.add(kl.Conv2D(128, (1, 3), activation = 'relu', padding = 'same'))
conv_encoder.add(kl.Dense(384, activation = 'relu'))
conv_encoder.add(kl.Dense(5, activation = 'relu'))
conv_encoder.summary()

# shape of the output layer of the encoder part 
output_layer_shape = conv_encoder.layers[-1].output_shape[1:]

# decoder --------------------------------------------------------------------
conv_decoder = km.Sequential(name='conv_decoder')
conv_decoder.add(kl.Dense(384, activation = 'relu', input_shape=output_layer_shape))
conv_decoder.add(kl.Conv2D(128, (1, 3), activation = 'relu', padding = 'same'))
conv_decoder.add(kl.Conv2D(128, (10, 1), activation = 'relu', padding = 'same'))
conv_decoder.add(kl.Conv2D(64, (10, 1), activation = 'relu', padding = 'same'))
conv_decoder.add(kl.Conv2D(nb_windows, (10, 1), activation = 'sigmoid', padding = 'same'))
conv_decoder.summary()

# final model -----------------------------------------------------------------
conv_autoencoder = km.Sequential(name="ConvAutoencoderModel")
conv_autoencoder.add(conv_encoder)
conv_autoencoder.add(conv_decoder)
conv_autoencoder.summary()

conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#conv_autoencoder.fit(tensor4D_train, tensor4D_train, epochs=10, batch_size=10, validation_data=(tensor4D_test, tensor4D_test))
# ici on a model.fit(x,y) avec x = y = x_train_conv
