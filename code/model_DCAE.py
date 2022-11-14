#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:39:39 2022

@author: julie
"""
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
tensorflow.__version__

from data_preparation import main_4D_tensor_DCAE
from set_path import DATA_PATH

# Loading the 4D tensor for the training and test set :
tensor4D_train,tensor4D_test = main_4D_tensor_DCAE(DATA_PATH+'/Train_clean_datasets',DATA_PATH+'/Test_clean_datasets')

# Input shape :
nb_TS, window_size, nb_features, nb_windows = tensor4D_train.shape

# First model : reduction according to time only ------------------------------
# SANS MAXPOOLING ET UPSAMPLING
def DCAE_architecture1 (window_size, nb_features, nb_windows) :
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
    return conv_encoder, conv_decoder, conv_autoencoder


# Second model : reduction according to all dimensions ------------------------
# AVEC MAXPOOLING ET UPSAMPLING 

def DCAE_architecture2 (window_size, nb_features, nb_windows) :
# encoder ---------------------------------------------------------------------
    conv_encoder2 = km.Sequential(name='conv_encoder')
    conv_encoder2.add(kl.Conv2D(nb_windows, (10, 1), activation = 'relu', input_shape=(window_size, nb_features, nb_windows), padding = 'same'))
    # rk : input shape is of format : (rows, cols, channels) 
    conv_encoder2.add(kl.Conv2D(64, (10, 1), activation = 'relu', padding = 'same'))
    conv_encoder2.add(kl.MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    conv_encoder2.add(kl.Conv2D(128, (10, 1), activation = 'relu', padding = 'same'))
    conv_encoder2.add(kl.Conv2D(128, (1, 3), activation = 'relu', padding = 'same'))
    conv_encoder2.add(kl.Dense(384, activation = 'relu'))
    conv_encoder2.add(kl.Dense(5, activation = 'relu'))
    conv_encoder2.summary()
    
    # shape of the output layer of the encoder part 
    output_layer_shape2 = conv_encoder2.layers[-1].output_shape[1:]
    
    # decoder --------------------------------------------------------------------
    conv_decoder2 = km.Sequential(name='conv_decoder')
    conv_decoder2.add(kl.Dense(384, activation = 'relu', input_shape=output_layer_shape2))
    conv_decoder2.add(kl.Conv2D(128, (1, 3), activation = 'relu', padding = 'same'))
    conv_decoder2.add(kl.UpSampling2D((2, 2)))
    conv_decoder2.add(kl.Conv2D(128, (10, 1), activation = 'relu', padding = 'same'))
    conv_decoder2.add(kl.Conv2D(64, (10, 1), activation = 'relu', padding = 'same'))
    conv_decoder2.add(kl.Conv2D(nb_windows, (10, 1), activation = 'sigmoid', padding = 'same'))
    conv_decoder2.summary()
    conv_decoder2.compile(optimizer='adam', loss='binary_crossentropy')
    
    # final model -----------------------------------------------------------------
    conv_autoencoder2 = km.Sequential(name="ConvAutoencoderModel")
    conv_autoencoder2.add(conv_encoder2)
    conv_autoencoder2.add(conv_decoder2)
    conv_autoencoder2.summary()
    conv_autoencoder2.compile(optimizer='adam', loss='binary_crossentropy')
    return conv_encoder2, conv_decoder2, conv_autoencoder2



# Fit and test ----------------------------------------------------------------

conv_encoder, conv_decoder, conv_autoencoder = DCAE_architecture1(window_size, nb_features, nb_windows)
history = conv_autoencoder.fit(tensor4D_train, tensor4D_train, epochs=12, batch_size=10, validation_data=(tensor4D_test, tensor4D_test))
score = conv_autoencoder.evaluate(tensor4D_train, tensor4D_train)
print('score :', score)
tensor4D_pred = conv_autoencoder.predict((tensor4D_test))
#encoded_test_data = conv_encoder.predict(tensor4D_test)


# Evaluation methods to compare several architectures
def plot_loss(history, arch=1):
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title('loss variations - training of the 1D-DCAE architecture ' + str(arch) +' (resampled data)')
    plt.legend()
    plt.show()
    
def evaluate_decoding(X_test, conv_encoder, conv_decoder) : 
    encoded_data = conv_encoder(X_test).numpy()
    decoded_data = conv_decoder(encoded_data).numpy()
    nb_TS_test, window_size, nb_features, nb_windows = X_test.shape
    
    data_test_2D = X_test.reshape(nb_TS_test*window_size*nb_windows,nb_features)
    decoded_data_2D = decoded_data.reshape(nb_TS_test*window_size*nb_windows,nb_features)
    
    plt.plot(data_test_2D[0], 'b')
    plt.plot(decoded_data_2D[0], 'r')
    plt.fill_between(np.arange(len(decoded_data_2D[0])), decoded_data_2D[0], data_test_2D[0], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

def print_stats(true_data, pred_data):
    nb_TS_test, window_size, nb_features, nb_windows = true_data.shape
    true_data_2D = true_data.reshape(nb_TS_test*window_size*nb_windows,nb_features)
    pred_data_2D = pred_data.reshape(nb_TS_test*window_size*nb_windows,nb_features)
    print("RMSE = {}".format(np.sqrt(mean_squared_error(true_data_2D, pred_data_2D))))
    print("MAE = {}".format((mean_absolute_error(true_data_2D, pred_data_2D))))
    print("R2 = {}".format(r2_score(true_data_2D, pred_data_2D)))

plot_loss(history, arch=2)
print_stats(true_data=tensor4D_test, pred_data=tensor4D_pred)
evaluate_decoding(tensor4D_test,conv_encoder, conv_decoder)










