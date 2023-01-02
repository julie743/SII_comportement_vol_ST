# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 23:08:28 2022

@author: Lila
"""

# Import libraries
#--------------------------------------
# computing and plotting libraires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# evaluation metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import functions for MiniRocket
from tsai.basics import *
import sktime
import sklearn
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
import torch

# import our own functions
import data_preparation as DP
import retrieve_labels as RL
from model_DCAE import load_encoder_data


def print_setup():
    return my_setup(sktime, sklearn)

def load_datasets(data_train_path,data_test_path,
                  isDCAE=False,weigths_path=None):
    
    """
    This function loads the datasets and corresponding labels.
    
    Inputs: 
    - data_train_path: location of the train set
    - data_test_path: location of the test set
    - IsDCAE (bool): True if data_list and data_list_test are from the DCAE
      ouput, False otherwise. 
    - weigths_path: localisation of the DCAE trained weigths if isDCAE=True.
      
    Ouputs: 
    - data_train, data_test: 
        * if IsDCAE=False: data_train and data_test are lists containing the 
        dataframes of train and test multivariate time series.
        * if IsDCAE=True: data_train and data_test are DCAE outputs for 
        of the train and test sets.
    """
    
    print("Start of data loading...")
    if isDCAE == False: #raw datasets
        data_train, data_test = DP.main_preprocessing_data(data_train_path,data_test_path)
    else: #dataset from DCAE outpout
        tensor4D_train,tensor4D_test = DP.main_4D_tensor_DCAE(data_train_path,data_test_path)
        #encoded dataset = DCAE outpout: 
        data_train,data_test = load_encoder_data(path=weigths_path,
                                                 X_train=tensor4D_train,
                                                 X_test=tensor4D_test)
    print("...done!")  
    return data_train, data_test
    

def formats_dataset(classification_task,data_train,data_test,isDCAE=False):
    """
    This function creates the dataset and the corresponding labels for MiniRocket.
    
    Inputs:
    - classification_task (str): indicates which classification task is performed.
      Can take the values:
      *'is_error':  creates labels for error detection 
      (binary classification:  1 if error, 0 otherwise)
      *'scenario_name': creates labels for scenario classifciation 
      (multiclass classification)
      *'error_name': creates labels for type of error classification
      (multiclass classification)
    - data_train,data_test: 
        * if IsDCAE=False: data_train and data_test are lists containing the 
        dataframes of train and test multivariate time series.
        * if IsDCAE=True: data_train and data_test are DCAE outputs for 
        of the train and test sets.
    - IsDCAE (bool): True if data_list and data_list_test are from the DCAE
      ouput, False otherwise. 
    
    Outputs:
    - X (np.array): explanatory variables (contatenation of train and test set)
    - y (np.array): corresponding labels according to classification_task
      (contatenation of train and test set)
    - splits (tuple): contains (indices of the train set, indices of the test set)
    """

    # Create train and test arrays
    # for MiniRocket, we need a np.ndarray or a torch.Tensor. We will use np.ndarray. 
    #----------------------------------------
    
    if isDCAE == False: #raw datasets
        # Transform data_train into an array of shape (n_observations, n_features, n_timesteps):
        X_train = np.array(data_train, dtype = np.float32) #type float32 otherwise pythorch error
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[2],X_train.shape[1])
      
        # Transform data_test into an array of shape (n_observations, n_features, n_timesteps):
        X_test = np.array(data_test, dtype = np.float32) #type float32 otherwise pythorch error
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[2],X_test.shape[1])
        
    else: #dataset from DCAE outpout
        # Put the sliding windows from DCAE output one after another to have an 
        # array of shape (n_observations, n_features, n_timesteps)
        nb_TS_train, window_size, nb_features, nb_windows = data_train.shape
        X_train = data_train.reshape((nb_TS_train,nb_features,window_size*nb_windows))
        X_train = X_train.astype(np.float32)#type float32 otherwise pythorch error
        
        nb_TS_test, window_size, nb_features, nb_windows = data_test.shape
        X_test = data_test.reshape((nb_TS_test,nb_features,window_size*nb_windows))
        X_test = X_test.astype(np.float32)#type float32 otherwise pythorch error
        
    
    # Create train and test labels
    #----------------------------------------
    y_train,y_test = RL.main_retrieve_labels()
    
    # Concatenate train and test sets (required configuration for MiniRocket)
    #-------------------------------------------
    # All time series (train and test)
    X = np.concatenate((X_train,X_test),axis = 0)
    
    # Train and tests idices 
    train_idx = np.arange(X_train.shape[0])
    test_idx = np.arange(X_train.shape[0],X.shape[0])
    splits = (list(train_idx), list(test_idx))
    
    # Create different labels according to the classification_task
    y = np.concatenate((y_train[classification_task].values, y_test[classification_task].values))
    
    print("\nConfiguration:")
    print(check_data (X, y, splits))
    
    return X,y,splits


def offline_MiniRocket_features(X,splits,save_path=None,save_name='MRF'):
  """
  Pytorch implementation of minirocket using tsai and fastai libraries.
  In the offline calculation, all features obtained with MiniRocket 
  are calculated in a first stage and ramain the same throughout training.
  In order to avoid leakage between the train and test sets, fit 
  MiniRocketFeatures using JUST the train samples.

  Inputs: 
  - X (np.array): explanatory variables (train and test set)
  - splits (tuple): contains (indices of the train set, indices of the test set)
  - save_path (str): path to save MiniRocket features. If None, no save is done.
  - save_name (str): name of the file containing the MiniRocket features.

  Outputs:
  - X_feat (np.array): features computed by MiniRocket. 
    size: [sample_size x n_features x 1]. 
    The last dimension (1) is added because tsai expects 3D input data.
  """

  print("\nStart of MiniRocket feature computation...")

  mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device()) 
  mrf.fit(X[splits[0]],chunksize=10) #chunksize=10 otherwise CUDA out of memory error
  #chunksize = number of data read at a time. Small chunisize => slow process 
  #Function used to split a large dataset into chunks, avoiding memory error: 
  X_feat = get_minirocket_features(X, mrf, chunksize=10, to_np=True)

  print("...done!")

  # Save the features for later use
  if save_path is not None:
    features_path = Path(save_path + '/' + save_name + '.pt')
    features_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), features_path)

  return X_feat,mrf


def offline_MiniRocket_training(X_feat,y,splits,n_epoch=10,lr=5e-4,dropout=.2,
                                save_path=None,save_name='MRL'):
  """
  This function puts the previously computed MiniRocket features into dataloader
  to create bacthes and then train a Pytorch model.
  A simple linear neural network model is used to classify the MiniRocket 
  features. 

  Inputs:
  - X_feat (np.array): MiniRocket features.
  - y (np.array): labels (train and test set).
  - n_epoch (int): number of epochs to train the model.
  - lr (int): learing rate. If None, the user will enter its own lr.
  - dropout (float): dropout used in the linear network.
  - save_path (str): path to save the learner. If None, no save is done.
  - save_name (str): name of the file containing the learner

  Ouputs: 
  - learn (fastai.learner): learner = trained model. 
  - lr (float): learning rate
  """

  # Using tsai/fastai, create DataLoaders for the features in X_feat:
  tfms = [None, TSClassification()] 
  batch_tfms = TSStandardize(by_sample=True) 
  # creates DataLoaders:
  dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms) 

  # model is a linear classifier Head:
  model = build_ts_model(MiniRocketHead, dls=dls, fc_dropout = dropout)

  # Drop into fastai and use it to find a good learning rate.
  learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())

  if lr is None:
    print("\nTo reduce the amount of guesswork on picking a good starting learning "+
    "\nrate when training the model, we choose a value that is approximately "+
    "\nin the middle of the sharpest downward slope on the figure below:\n") 
    lf = learn.lr_find()
    print(lf)
    plt.plot(lf)
    plt.show()
    lr = input( "\nChoose the learning rate (possible value = 5e-4): " )
    lr = float(lr)
  
  print("\nStart of MiniRocket training...")
  timer.start()
  learn.fit_one_cycle(n_epoch = n_epoch, lr_max = lr, reset_opt = True)
  timer.stop()
  print("\n...done!")

  # Save the learner for later use
  if save_path is not None:
    learner_path = Path(save_path + '/' + save_name + '.pkl')
    learner_path.parent.mkdir(parents=True, exist_ok=True)
    learn.export(learner_path)
    
  return learn,lr 


def offline_MiniRocket_prediction(X,y,splits,classification_task, mrf=None,learn=None,
                                  save_path_features=None,save_name_features="MRF",
                                  save_path_learner=None,save_name_learner="MRL",
                                  save_path_train_im=None,save_path_test_im=None):

  """
  This function makes the prediction using MiniRocket. 

  Inputs:
    - X (np.array): explanatory variables (train and test set)
    - y (np.array): labels (train and test set).
    - splits (tuple): contains (indices of the train set, indices of the test set)
    - mrf,learn: MiniRocket features and lerner previously computed.
      If None, use save_path_features and save_path_learner to load mrf and learn
    - save_path_features,save_path_learner (str): path where MiniRocket features
      and learner are saved. If None, use the MiniRocket features mrf and 
      learner learn previously computed in this notebook.
    - save_name_features,save_name_learner (str): name of the file containing the
    MiniRocket features and learner.
    - classification_task (str): indicates which classification task is performed.
      Can take the values:
      *'is_error':  creates labels for error detection 
      *'scenario_name': creates labels for scenario classifciation 
      *'error_name': creates labels for type of error classification
    - save_path_train_im, save_path_test_im (str): path to the folder where
      the image is saved. If None, no image is saved. 

  Outputs:
    - accuracy_score_train, accuracy_score_test (float): accuracy score on the 
    train and test sets.
    - f1_score_train, f1_score_test (float): macro f1 score on the train and 
    test sets.
  """
  
  if save_path_features is not None:
    # Recreate mrf (MiniRocketFeatures) if we want to load MiniRocket 
    # features previously calculated:
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device()) 

    features_path = Path(save_path_features + '/' + save_name_features + '.pt')
    mrf.load_state_dict(torch.load(features_path))

  if save_path_learner is not None:
    learner_path = Path(save_path_learner + '/' + save_name_learner + '.pkl')
    learn = load_learner(learner_path, cpu=False)

  print("Compute prediction...")
  # Create new features 
  new_feat_train = get_minirocket_features(X[splits[0]], mrf, chunksize=10, to_np=True) #X[splits[0]] = train set
  new_feat_test = get_minirocket_features(X[splits[1]], mrf, chunksize=10, to_np=True) #X[splits[1]] = test set

  # and pass the newly created features:
  probas_train, _, preds_train = learn.get_X_preds(new_feat_train)
  probas_test, _, preds_test = learn.get_X_preds(new_feat_test)
  print("...done!")

  if classification_task == 'is_error':
    preds_train = preds_train.astype(int)
    preds_test = preds_test.astype(int)

  # print results
  # --- train results
  accuracy_score_train = accuracy_score(y[splits[0]], preds_train)
  f1_score_train = f1_score(y[splits[0]], preds_train, average='macro')
  print("\nAccuracy on the train set:", accuracy_score_train)
  print("F1 score on the train set:", f1_score_train)
  f, ax = plt.subplots(1,figsize=(6, 6))
  cm_train = confusion_matrix(y[splits[0]], preds_train)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=np.unique(y))
  disp.plot(ax=ax)
  disp.ax_.set_xticklabels(np.unique(y), rotation = 90)
  ax.set_title("Confusion matrix for the train set:")
  if save_path_train_im != None:
      f.savefig(save_path_train_im + "/" + classification_task + "_CMtrain.png")
  plt.show()
  
  # --- test results
  accuracy_score_test = accuracy_score(y[splits[1]], preds_test)
  f1_score_test = f1_score(y[splits[1]], preds_test, average='macro')
  print("\nAccuracy on the test set:", accuracy_score_test)
  print("F1 score on the test set:", f1_score_test)
  f, ax = plt.subplots(1,figsize=(6, 6))
  cm_test = confusion_matrix(y[splits[1]], preds_test)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.unique(y))
  disp.plot(ax=ax)
  disp.ax_.set_xticklabels(np.unique(y), rotation = 90)
  ax.set_title("Confusion matrix for the test set:")
  if save_path_test_im != None:
      f.savefig(save_path_test_im + "/" + classification_task + "_CMtest.png")
  plt.show()

  return accuracy_score_train, accuracy_score_test, f1_score_train, f1_score_test