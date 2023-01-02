# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 22:51:13 2022

@author: Lila
"""

# import libraries
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy.stats import qmc

from tsai.basics import *
import sktime
import sklearn
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *


def compute_KcrossVal(X,y,classification_task,dropout,n_epochs,lr,k=10):
  """
  This functions uses a k-fold cross-validation to 
  calculate the accuracy of the model.

  Inputs: 
  -------
  - X, y (np.array): training set and corresponding labels
  - classification_task (str): indicates which classification task is performed.
  - dropout (float): dropout value in the linear head. 
  - n_epochs (int): number of epochs to train the models
  - lr (float): learning rate value.
  - k (int): number of folds in the k-fold cross validation

  Oupouts:
  -------
  - accuracy_kf (float): accuracy computed by k-fold cross validation
  """

  # Initialize lists with final results
  y_pred_total = []
  y_val_total = []

  # Split data into test and train
  kf = KFold(n_splits=k,shuffle=True,random_state=2022)

  # kf-fold cross-validation loop
  for train_index, test_index in kf.split(X):
    X_train_kf, X_val_kf = X[train_index], X[test_index]
    y_train_kf, y_val_kf = y[train_index], y[test_index]

    # ====================================
    # BEGIN data formatting for MiniRocket
    # ====================================
    # 1 -- Put data into correct format for MiniRocket
    X_kf = np.concatenate((X_train_kf,X_val_kf),axis = 0)
    y_kf = np.concatenate((y_train_kf,y_val_kf))
    # 2 -- Train and tests idices 
    train_kf_idx = np.arange(X_train_kf.shape[0])
    val_kf_idx = np.arange(X_train_kf.shape[0],X_kf.shape[0])
    splits_kf = (list(train_kf_idx), list(val_kf_idx))
    # ====================================
    # END data formatting for MiniRocket
    # ====================================

    # ====================================
    # BEGIN MiniRocket training
    # ====================================
    # 1 -- Compute MiniRocket features on X_kf and fit on X_train_kf only
    mrf = MiniRocketFeatures(X_kf.shape[1], X_kf.shape[2],random_state=42).to(default_device()) 
    mrf.fit(X_train_kf,chunksize=10) 
    # function used to split a large dataset into chunks, avoiding memory error
    X_feat_kf = get_minirocket_features(X_kf, mrf, chunksize=10, to_np=True)

    # 2 -- Create DataLoaders for the features in X_feat.
    tfms = [None, TSClassification()] 
    batch_tfms = TSStandardize(by_sample=True)
    # creates DataLoaders
    dls = get_ts_dls(X_feat_kf, y_kf, splits=splits_kf, tfms=tfms, batch_tfms=batch_tfms)

    # 3 -- Model is a linear classifier Head
    model = build_ts_model(MiniRocketHead, dls=dls, fc_dropout = dropout)

    # 4 -- Create learner and train it
    learn = Learner(dls, model, metrics=accuracy)
    with learn.no_bar(), learn.no_logging(): #silence outpouts
      learn.fit_one_cycle(n_epoch=n_epochs, lr_max=lr, reset_opt = True)
    # ====================================
    # END MiniRocket training 
    # ====================================

    # ====================================
    # BEGIN MiniRocket prediction
    # ====================================
    # Predict on X_val_kf
    # 1 -- Create new features on the val set
    new_feat_val = get_minirocket_features(X_val_kf, mrf, chunksize=10, to_np=True)
    # 2 -- Pass the newly created features
    probas_val, _, y_pred_val = learn.get_X_preds(new_feat_val)
    if classification_task == 'is_error':
      y_pred_val = y_pred_val.astype(int)
    # ====================================
    # END MiniRocket prediction
    # ====================================
  
    # Append y_pred and y_test values of this k-fold step to list with total values
    y_pred_total.append(y_pred_val)
    y_val_total.append(y_val_kf)

  # Flatten lists with test and predicted values
  y_pred_total = [item for sublist in y_pred_total for item in sublist]
  y_val_total = [item for sublist in y_val_total for item in sublist]

  accuracy_kf = accuracy_score(y_val_total, y_pred_total)
  return accuracy_kf



def createParamGrid (l_bounds,u_bounds,n_values,N=10000):
  """
  This function creates a grid of values for the parameters 
  in order to optimally cover the space between two given limits. 
  To do that, it creates N Latin Hypercubes and chooses 
  the one with the lowest discrepancy value. 

  Inputs:
  ------
  - l_bounds (list): list of low boundaries for each parameter
  - u_bounds (list): list of upper boundaries for each parameter.
  - n_values (int): number of values we want to try for each parameter.
  - N (int): number or Latin Hypercubes to create in order to find the
    one with the lowest discrepancy.

  Outputs:
  -------
  - best_LHS (np.array): array of parameters values. 
    * each column corresponds to a parameter
    * each row corresponds to values taken by the parameters. 
      So each row corresponds to a parameters setup. 
  """

  # Generate N Latin Hypercubes
  d_min = 1 # we want to find the LHS that minimizes the discrepancy

  for i in range(N):
    # Latin Hypercube generator
    sampler = qmc.LatinHypercube(d=len(l_bounds)) 
    LHS = sampler.random(n=n_values)
    # Compute discrepancy
    d = qmc.discrepancy(LHS)   

    # Find LHS with lowes discrepancy
    if d <= d_min: 
      d_min = d
      best_LHS = LHS

  # scale the LHS to actual bounds 
  best_LHS = qmc.scale(best_LHS, l_bounds, u_bounds) 
  return best_LHS

def GridSearchCV_MR (X,y,classification_task,l_bounds,u_bounds,n_values,k=10):

  """
  This function is the equavalent of GridSearchCV in sklearn.model_selection
  It performs an exhaustive search over specified parameter values for MiniRocket.
  
  In l_bounds and u_bounds, pass bounds in the following order:
  [dropout, number of epochs, learning rate]

  Inputs:
  ------
  - X, y (np.array): training set and corresponding labels
  - classification_task (str): indicates which classification task is performed.
  - l_bounds (list): list of low boundaries for each parameter
  - u_bounds (list): list of upper boundaries for each parameter.
  - n_values (int): number of values we want to try for each parameter.
  - k (int): number of folds in the k-fold cross validation.

  Outputs:
  -------
  - best_params (list): configuration of parameters giving the best accuracy
    according to k-fold cross validation. 
  - best_accuracy (float): best accuracy computed with k-fold cross validation.
  """

  # Create an optimised parameters grid
  paramGrid = createParamGrid(l_bounds,u_bounds,n_values)

  # Find parameters giving the best accuracy
  best_accuracy = 0
  best_params = 0
  print("Start of grid search CV...")
  for i in range(paramGrid.shape[0]): 
    if i%5==0: #print every 5 searchs
        print("search {}/{}".format(i,n_values))
    
    dropout, n_epochs, lr = paramGrid[i,:]
    # transform n_epochs into int by rounding to closest integer
    n_epochs = int(np.round(n_epochs)) 
    # transform lr of type np.float64 into float
    lr = float(lr) #otherwise Learner error

    accuracy_kf = compute_KcrossVal(X,y,classification_task,dropout,n_epochs,lr,k)
    
    if accuracy_kf >= best_accuracy:
      best_accuracy = accuracy_kf
      best_params = [dropout,n_epochs,lr]
      
  print("...done!")
  return best_params, best_accuracy