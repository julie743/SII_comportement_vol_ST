#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:31:39 2022

@author: julie
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

from set_path import PATH,DATA_PATH,DATA_PATH_chosen


def load_data(files,path) :
    '''
    load_data : load time series .pkl files one by one and record them in a list
    Inputs : 
        - files : list of files names
        - path : where to pick up the files
    Outputs : 
        - data_list : list of the time series dataframes. Each element of the list 
        is one dataframe of a time series. 
    '''
    
    data_list = []
    for f in files : 
        data_list.append(pd.read_pickle(os.path.join(path,f)))
    return data_list


def rescale(data_list,var_quanti,scaler=0) : 
    '''
    rescale : rescale the quantitative variables with the min-max normalization. 
    The scaler is fit on the training dataset only, to do so time series of the
    training set are concatenated one after the other in a unique dataframe. 
    If the scaler = 0, we define a new scaler that is fit and transformed on 
    the training set. If the scaler is /= 0 then we use the input scaler which 
    is the one which was fit on the training set, in order to transform the 
    test set.
    Inputs : 
        - data_list : list of the time series dataframes
        - var_quanti : list of quantitative variables
    Outputs : 
        - data_list : list of rescaled time series dataframes
    '''
    if scaler == 0 :
        scaler = MinMaxScaler()
        scaler.fit(pd.concat([df_1ST[var_quanti] for df_1ST in data_list]))
    for i in range(len(data_list)) : 
        data_list[i][var_quanti] = pd.DataFrame(scaler.transform(data_list[i][var_quanti]),columns=var_quanti)
    return data_list, scaler


def var_quali_to_dummies(data_list,var_quali,var_quanti) :
    '''
    var_quali_to_dummies : Transform qualitative variables to dummies
    Inputs : 
        - data_list : list of the time series dataframes
        - var_quali : list of qualitative variables
        - var_quanti : list of quantitative variables
    Outputs : 
        - data_list : list of rescaled time series dataframes
        - var_dummies : list of dummy variables' names
    '''
    
    list_len_ST = [len(data_list[i].index) for i in range(len(data_list))] # record the length of each time series
    big_df = pd.concat([data_list[i] for i in range(len(data_list))])
    for var in var_quali :
        big_df[var]=pd.Categorical(big_df[var],ordered=False)
    dummies = pd.get_dummies(big_df[var_quali],drop_first=True)
    big_df = pd.concat([big_df[var_quanti],dummies],axis=1)
    index_ST = np.r_[[0],np.cumsum(list_len_ST)] # index of the beginning of each time series in the big dataframe
    # separate each time series in an own dataframe :
    for i in range(len(data_list)) :
        data_list[i] = pd.DataFrame(big_df.iloc[index_ST[i]:index_ST[i+1]])
    var_dummies = list(dummies.columns)
    return data_list, var_dummies



def extend_TS(data_list,var_quanti,len_ST=0) :
    '''
    extend_TS : extend time series so that they are all at the same length. The 
    length of the longest time series is taken as a reference. The shorter time 
    series are extended until they reach the same length. The quantitative 
    variables are extended with 0, and the qualitative variables are extended with
    their last value. 
    Inputs : 
        - data_list : list of the time series dataframes
        - var_quanti : list of quantitative variables
    Outputs : 
        - data_list : list of rescaled time series dataframes
        - len_ST : legnth of all time series after extension
        - ref_time : vector containing the time reference of the longest time series
    '''

    # take a reference time series : the longest one
    ref_ST = np.argmax([len(data_list[i].index) for i in range(len(data_list))])
    ref_time = data_list[ref_ST]['sim/time/total_flight_time_sec']
    # if the length of the time series was not already given, choose the length
    # of the longest time series : 
    if len_ST == 0 : 
        len_ST = len(data_list[ref_ST].index)
    for i in range(len(data_list)) : 
        # extension = last line repeated until we reach the longest value of time series
        extension = pd.DataFrame(data = data_list[i].iloc[-1].values.reshape(1,-1), columns = data_list[i].columns)
        extension[var_quanti] = 0 # for quantitative variables we put extend with 0
        len_ext = len_ST-len(data_list[i].index) # length of the necessary extension
        extension = extension.loc[extension.index.repeat(len_ext)].reset_index(drop=True) # repeating the same row
        data_list[i] = pd.concat([data_list[i],extension],axis=0,ignore_index=True) # add the extension at the end of the dataframe
    return data_list, len_ST, ref_time



def resample(data_list,len_ST,ref_time,chosen_resolution=0) :
    '''
    resample : resample time series to reduce the time resolution up to 25ms 
    (instead of 10ms)
    Inputs : 
        - data_list : list of time series dataframes
        - len_ST : legnth of all time series after extension
        - ref_time : vector containing the time reference of the longest time series
        - chosen_resolution : int indicating the index to pick up, for example 
        chosen_resolution = 2 means that we pick up indices 0,2,4,... in the time 
        series, so their length is divided by 2. If chosen_resolution is not 
        already chosen then it is calculated in order to reduce the resolution to 
        25 ms. 
    Outputs : 
        - data_list : list of resampled time series dataframes 
        - chosen_resolution : int indicating the index to pick up for the resolution
    '''
    
    if chosen_resolution == 0 :
        # choose the resolution : 
        possible_resolutions = np.array([i for i in range(1,len_ST) if len_ST % i == 0])
        ind_025sec = np.argmax(ref_time>0.25) # resolution of 0.25 seconds
        chosen_resolution =  min(possible_resolutions, key=lambda x:abs(x-ind_025sec))
    
    # select lines in the dataframe : 0, chosen_resolution, chosen_resolution*2, ...
    lines = np.arange(0,len_ST,chosen_resolution)
    for i in range(len(data_list)) : 
        data_list[i] = data_list[i].loc[lines].reset_index(drop=True)
    return data_list,chosen_resolution
    


def window_stride(data_list) :
    '''
    window_stride : function to choose a window and a stride. The window will be 
    2*stride because windows should be overlapping. The stride and the window are 
    chosen so that only a niglibgeable portion of the information will be left out
    if it doesn't fit in the last window
    Inputs : 
        - data_list : list of time series dataframes
    Outputs : 
        - window : size of the chosen window
        - stride : chosen stride
    '''
    
    len_ST = len(data_list[0].index)
    stride_max = np.ceil(len_ST*0.05) # stride_max = 5% of total length
    possible_strides = np.arange(1,stride_max)[::-1] # possible stride in decerasing order 
    # compute the ratio of forgotten data when applying a certain stride against the size of the window :
    left_ratio = [(len_ST % i)/(2*i) for i in possible_strides] 
    ind = np.argmax(np.array(left_ratio)<0.02) # we allow ourselves to forget 2% of the length of the chosen window
    stride = int(possible_strides[ind]) # chosen stride
    window = int(stride*2) # chosen window 
    return window, stride


def tensor_allTS(data_list, size_window, stride) :
    '''
    tensor_allTS : Build a 4D array for the input of the DCAE. The 4D array is of 
    size (nb_TS, window_size, nb_features, nb_windows ). 
    Inputs : 
        - data_list : list of time series dataframes
        - window : size of the chosen window
        - stride : chosen stride
    Outputs : 
        - tensor4D : 4D array of the slid windows over each time series
    '''
    
    # parameters for the size of the 4D array : 
    nb_TS = len(data_list)
    nb_features = len(data_list[0].columns)
    len_ST = len(data_list[0].index)
    nb_window = int((len_ST-size_window)/stride)+1
    
    tensor4D = np.zeros((nb_TS,size_window,nb_features,nb_window))
    
    # fill the 4D array with the slid windows 
    for j in range(nb_TS) :
        start = 0
        for i in range(len_ST) : 
            if start+size_window <= len_ST :
                tensor4D[j,:,:,i] = data_list[j].iloc[start:start+size_window]
                start += stride
    return tensor4D


# -----------------------------------------------------------------------------
# MAIN FUNCTIONS : The following functions do the above preprocessing steps in
# the correct order.

def main_preprocessing_data(path_train,path_test) :
    '''
    main_preprocessing_data : preprocess the time series data : loading the data 
    (train and test), transorming the qualittaive variables to dummies, extending 
    time series to the same length, rescaling time series
    Inputs : 
        - path_train : where to find the training data
        - path_test :  where to find the test data
    Outputs : 
        - data_list : training set : list of time series dataframes 
        - data_list_test : test set : list of time series dataframes 
    '''
    
    # loading qualitative and quantitative variables
    var_quanti = pd.read_pickle(os.path.join(path_train,'VAR_QUANTI_TAB.pkl')).values.squeeze()
    var_quali = pd.read_pickle(os.path.join(path_train,'VAR_QUALI_TAB.pkl')).values.squeeze()

    # loading training data 
    files_train = pd.read_csv(os.path.join(path_train,'file_names.csv')).squeeze()
    files_train = [f[:-2]+'pkl' for f in files_train]
    data_list = load_data(files_train,path_train)
    nb_train = len(data_list) # number of observations in the training dataset

    # loading the test data
    files_test = pd.read_csv(os.path.join(path_test,'file_names.csv')).squeeze()
    files_test = [f[:-2]+'pkl' for f in files_test]
    
    # record training and test set in the same dataframe to preprocess them all at once :
    data_list.extend(load_data(files_test,path_test)) 

    # processing the data :
    # 1. transforming qualitative variables to dummies
    data_list, var_dummies = var_quali_to_dummies(data_list,var_quali,var_quanti)
    
    # 2. extending time series so that they have the same length
    data_list, len_ST, ref_time = extend_TS(data_list,var_quanti)
    
    # 3. separation training / test set
    data_list_test = data_list[nb_train:] #test set
    data_list = data_list[:nb_train] # train set
    
    # 4. rescaling the time series
    data_list, scaler = rescale(data_list,var_quanti) 
    data_list_test, _ = rescale(data_list_test,var_quanti,scaler) 

    return data_list,data_list_test


def main_4D_tensor_DCAE(path_train,path_test) :
    '''
    main_4D_tensor_DCAE : preprocess the time series data for DCAE and put them in 
    the shape of a 4D tensor
    Inputs : 
        - path_train : where to find the training data
        - path_test :  where to find the test data
    Outputs : 
        - tensor4D_train : training set 
        - tensor4D_test : test set 
    '''
    
    # loading qualitative and quantitative variables
    var_quanti = pd.read_pickle(os.path.join(path_train,'VAR_QUANTI_TAB.pkl')).values.squeeze()
    var_quali = pd.read_pickle(os.path.join(path_train,'VAR_QUALI_TAB.pkl')).values.squeeze()

    # loading training data 
    files_train = pd.read_csv(os.path.join(path_train,'file_names.csv')).squeeze()
    files_train = [f[:-2]+'pkl' for f in files_train]
    data_list = load_data(files_train,path_train)
    nb_train = len(data_list) # number of observations in the training dataset

    # loading the test data
    files_test = pd.read_csv(os.path.join(path_test,'file_names.csv')).squeeze()
    files_test = [f[:-2]+'pkl' for f in files_test]
    data_list.extend(load_data(files_test,path_test))

    # processing the data :
    # 1. transforming qualitative variables to dummies
    data_list, var_dummies = var_quali_to_dummies(data_list,var_quali,var_quanti)
    
    # 2. extending time series so that they have the same length
    data_list, len_ST, ref_time = extend_TS(data_list,var_quanti)
    
    # 3. separation training / test set
    data_list_test = data_list[nb_train:] #test set
    data_list = data_list[:nb_train] # train set
    
    # 3. rescaling the time series
    data_list, scaler = rescale(data_list,var_quanti)
    data_list_test, _ = rescale(data_list_test,var_quanti,scaler) 
    
    # 4. resampling the time series to reduce the time resolution
    data_list, chosen_resolution = resample(data_list,len_ST,ref_time)
    data_list_test, _ = resample(data_list_test,len_ST,ref_time,chosen_resolution)
    
    # 5. choosing the window and stride on the training data
    size_window, stride = window_stride(data_list) 

    # 3. building the two tensors 
    tensor4D_train = tensor_allTS(data_list, size_window, stride)
    tensor4D_test = tensor_allTS(data_list_test, size_window, stride)
    return tensor4D_train,tensor4D_test


#path_train = DATA_PATH+'/Train_clean_datasets'
#path_test = DATA_PATH+'/Test_clean_datasets'
#tensor4D_train,tensor4D_test = main_4D_tensor_DCAE(path_train,path_test)











