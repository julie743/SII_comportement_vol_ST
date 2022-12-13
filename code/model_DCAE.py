#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:39:39 2022

@author: julie
"""
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
tensorflow.__version__

import os
from data_loading import mkdir
from data_preparation import main_4D_tensor_DCAE
from set_path import DATA_PATH

# First model : reduction according to time only ------------------------------
# DCAE WITHOUT MAXPOOLING ET UPSAMPLING
def DCAE_architecture1 (window_size, nb_features, nb_windows) :
    '''
    Description of the function : implement the second architecture of the 
    1D-DCAE which offers dimension reduction both on the time steps and the 
    features
    
    Inputs
    ----------
    window_size : (int) size of the sliding window
    nb_features : (int) number of features in the data
    nb_windows : (int) number of slidingwindows

    Outputs
    -------
    conv_encoder : convolutional encoder model (keras)
    conv_decoder : convolutional decoder model (keras)
    conv_autoencoder : convolutional autoencoder model (keras)
    '''
    
    conv_encoder = km.Sequential(name='conv_encoder')
    conv_encoder.add(kl.Conv2D(nb_windows, (10, 1), activation = 'relu', input_shape=(window_size, nb_features, nb_windows), padding = 'same'))
    # rk : input shape is of format : (rows, cols, channels) 
    conv_encoder.add(kl.Conv2D(64, (10, 1), activation = 'relu', padding = 'same'))
    conv_encoder.add(kl.Conv2D(128, (10, 1), activation = 'relu', padding = 'same'))
    conv_encoder.add(kl.Conv2D(128, (1, 3), activation = 'relu', padding = 'same'))
    conv_encoder.add(kl.Dense(384, activation = 'relu'))
    conv_encoder.add(kl.Dense(5, activation = 'relu')) #5
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
    '''
    Description of the function : implement the second architecture of the 
    1D-DCAE which offers dimension reduction both on the time steps and the 
    features
    
    Inputs
    ----------
    window_size : (int) size of the sliding window
    nb_features : (int) number of features in the data
    nb_windows : (int) number of slidingwindows

    Outputs
    -------
    conv_encoder : convolutional encoder model (keras)
    conv_decoder : convolutional decoder model (keras)
    conv_autoencoder : convolutional autoencoder model (keras)
    '''
    
    # encoder -----------------------------------------------------------------
    conv_encoder2 = km.Sequential(name='conv_encoder')
    conv_encoder2.add(kl.Conv2D(nb_windows, (10, 1), activation = 'relu', input_shape=(window_size, nb_features, nb_windows), padding = 'same'))
    # rk : input shape is of format : (rows, cols, channels) 
    conv_encoder2.add(kl.Conv2D(64, (10, 1), activation = 'relu', padding = 'same'))
    conv_encoder2.add(kl.MaxPooling2D(pool_size=(1, 2), padding = 'same'))
    conv_encoder2.add(kl.Conv2D(128, (10, 1), activation = 'relu', padding = 'same'))
    conv_encoder2.add(kl.Conv2D(128, (1, 3), activation = 'relu', padding = 'same'))
    conv_encoder2.add(kl.Dense(384, activation = 'relu'))
    conv_encoder2.add(kl.Dense(5, activation = 'relu')) #5
    conv_encoder2.summary()
    
    # shape of the output layer of the encoder part 
    output_layer_shape2 = conv_encoder2.layers[-1].output_shape[1:]
    
    # decoder --------------------------------------------------------------------
    conv_decoder2 = km.Sequential(name='conv_decoder')
    conv_decoder2.add(kl.Dense(384, activation = 'relu', input_shape=output_layer_shape2))
    conv_decoder2.add(kl.Conv2D(128, (1, 3), activation = 'relu', padding = 'same'))
    conv_decoder2.add(kl.UpSampling2D((1, 2)))
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


# Evaluation methods to compare several architectures
def plot_loss(history, arch=1):
    '''
    Description of the function : plot the loss of the training of the DCAE
    
    Inputs
    ------
    history : history obtained from the training of the DCAE
    arch : (int) chosen architecture 1 or 2. The default is 1.
    '''
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title('loss variations - training of the 1D-DCAE architecture ' + str(arch) )
    plt.legend()
    plt.xticks(np.arange(0, len(history.history["loss"]),2))
    plt.show()
    
# to delete : 
def evaluate_decoding(X_test, conv_encoder, conv_decoder) : 
    '''
    Description of the function : print error of the DCAE
    
    Inputs
    ----------
    X_test : np.darray of test data
    conv_encoder : convolutional encoder built with the DCAE
    conv_decoder : convolutional decoder built with the DCAE
    '''
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
    '''
    Description of the function : print metrics for the reconstruction of the 
    test set by the autoencoder
    
    Inputs
    ----------
    true_data : np.darray of test data
    pred_data : np.darray of predicted data by the DCAE
    '''
    nb_TS_test, window_size, nb_features, nb_windows = true_data.shape
    true_data_2D = true_data.reshape(nb_TS_test*window_size*nb_windows,nb_features)
    pred_data_2D = pred_data.reshape(nb_TS_test*window_size*nb_windows,nb_features)
    print("RMSE = {}".format(np.sqrt(mean_squared_error(true_data_2D, pred_data_2D))))
    print("MAE = {}".format((mean_absolute_error(true_data_2D, pred_data_2D))))
    print("R2 = {}".format(r2_score(true_data_2D, pred_data_2D)))

def save_weigths(conv_encoder, file_name) :
    '''
    Description of the function : save trained encoder model
    Inputs
    ------
    con_encoder : convolutional encoder built with the DCAE
    file_name : name to give to the file
    '''
    directory_weigths = os.path.join(DATA_PATH,'encoder_weights')
    mkdir(directory_weigths)
    conv_encoder.save(os.path.join(directory_weigths,file_name)) # save encoder's weigth
    
    
def load_encoder_data(path,X_train,X_test) :
    '''
    Description of the function : load the trained encoder model
    
    Inputs
    -------
    path : (string) path where to find the encoder model's weigths
    X_train : np.darray of train data
    X_test : np.darray of test data

    Outputs
    -------
    X_train_encode : np.darray of encoded train data
    X_test_encode : np.darray of encoded test data
    '''
    encoder = load_model(path)
    # encode the train data
    X_train_encode = encoder.predict(X_train)
    # encode the test data
    X_test_encode = encoder.predict(X_test)
    return X_train_encode, X_test_encode

#-------------------------------------------------------------------------------
def main_DCAE(arch=1,file_name='encoder_architecture1.h5',epochs=15,batch_size=10) :
    '''
    Description of the function : calls the previous functions in the correct 
    order to load the data, build the DCAE, train it, plot the results and save
    the trained model

    Inputs
    -------
    file_name : (string) path where to find the encoder model's weigths
    epochs : (int) The default is 15.
    batch_size : (int) The default is 10.
    '''
    # Loading the 4D tensor for the training and test set :
    tensor4D_train,tensor4D_test = main_4D_tensor_DCAE(DATA_PATH+'/Train_clean_datasets',DATA_PATH+'/Test_clean_datasets')
    
    # Input shape :
    nb_TS, window_size, nb_features, nb_windows = tensor4D_train.shape
    
    # Fit and test the model --------------------------------------------------
    if arch == 1 :
      DCAE_model = DCAE_architecture1
    else : 
      DCAE_model = DCAE_architecture2
    conv_encoder, conv_decoder, conv_autoencoder = DCAE_model(window_size, nb_features, nb_windows)
    history = conv_autoencoder.fit(tensor4D_train, tensor4D_train, epochs=epochs, batch_size=batch_size, validation_data=(tensor4D_test, tensor4D_test))
    tensor4D_pred = conv_autoencoder.predict(tensor4D_test)
    
    # Plot results ------------------------------------------------------------
    score = conv_autoencoder.evaluate(tensor4D_train, tensor4D_train)
    print('score :', score)
    plot_loss(history, arch=arch)
    print_stats(true_data=tensor4D_test, pred_data=tensor4D_pred)
    evaluate_decoding(tensor4D_test,conv_encoder, conv_decoder)
        
    # save model's weigths
    save_weigths(conv_encoder,file_name)


#main_DCAE(epochs=15,batch_size=10)


