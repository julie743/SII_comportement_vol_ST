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

'''
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader 
'''
#import data_loading as DL
from set_path import PATH,DATA_PATH,DATA_PATH_chosen

def load_data(files,path) :
    data_list = []
    for f in files : 
        data_list.append(pd.read_pickle(os.path.join(path,f)))
    return data_list

def extend_TS(data_list,len_ST=0) :
    if len_ST == 0 :
        ref_ST = np.argmax([len(data_list[i].index) for i in range(len(data_list))])
        len_ST = len(data_list[ref_ST].index)
    for i in range(len(data_list)) : 
        # extension = last line repeated until we reach the longest value of time series
        extension = pd.DataFrame(data = data_list[i].iloc[-1].values.reshape(1,-1), columns = data_list[i].columns)
        len_ext = len_ST-len(data_list[i].index) # length of the necessary extension
        extension = extension.loc[extension.index.repeat(len_ext)].reset_index(drop=True) # repeating the same row
        data_list[i] = pd.concat([data_list[i],extension],ignore_index=True) 
    return data_list,len_ST

def rescale(data_list,var_quanti) : 
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([df_1ST[var_quanti] for df_1ST in data_list]))
    for i in range(len(data_list)) : 
        data_list[i][var_quanti] = pd.DataFrame(scaler.transform(data_list[i][var_quanti]),columns=var_quanti)
    return data_list

def resample(data_list,len_ST,chosen_resolution=0) :
    if chosen_resolution == 0 :
        # choose the resolution : 
        possible_resolutions = np.array([i for i in range(1,len_ST) if len_ST % i == 0])
        time_ref = 'sim/time/total_flight_time_sec'
        ind_5sec = np.argmax(data_list[0][time_ref]>0.25) # resolution of 0.25 seconds
        chosen_resolution =  min(possible_resolutions, key=lambda x:abs(x-ind_5sec))
    
    # select lines in the dataframe : 0, chosen_resolution, chosen_resolution*2, ...
    lines = np.arange(0,len_ST,chosen_resolution)
    for i in range(len(data_list)) : 
        data_list[i] = data_list[i].loc[lines].reset_index(drop=True)
    return data_list,chosen_resolution
    

# choose a window and a stride
def window_stride(data_list) :
    len_ST = len(data_list[0].index)
    stride_max = np.ceil(len_ST*0.05) # 5% of total length
    possible_strides = np.arange(1,stride_max)[::-1]
    left_ratio = [(len_ST % i)/(2*i) for i in possible_strides] # ratio of untouched data against the size of the window
    ind = np.argmax(np.array(left_ratio)<0.02) # we allow ourselves to forget 2% of the time series length
    stride = int(possible_strides[ind])
    window = int(stride*2)
    return window, stride


def tensor_allTS(data_list, size_window, stride) :
    nb_TS = len(data_list)
    nb_features = len(data_list[0].columns)
    len_ST = len(data_list[0].index)
    nb_window = int((len_ST-size_window)/stride)+1
    
    tensor4D = np.zeros((nb_TS,size_window,nb_features,nb_window))
    
    for j in range(nb_TS) :
        start = 0
        for i in range(len_ST) : 
            if start+size_window <= len_ST :
                tensor4D[j,:,:,i] = data_list[j].iloc[start:start+size_window]
                start += stride
    return tensor4D


# -----------------------------------------------------------------------------

def main_4D_tensor(path_train,path_test) :
    # loading the 4D tensor train----------------------------------------------
    files = pd.read_csv(os.path.join(path_train,'file_names.csv')).squeeze()
    files = [f[:-2]+'pkl' for f in files]
    var_quanti = pd.read_pickle(os.path.join(DATA_PATH_chosen,'VAR_QUANTI_TAB.pkl')).values.squeeze()
    
    data_list = load_data(files,path_train)
    data_list, len_ST = extend_TS(data_list)
    data_list, chosen_resolution = resample(data_list,len_ST)
    data_list = rescale(data_list,var_quanti)
    size_window, stride = window_stride(data_list)
    
    tensor4D_train = tensor_allTS(data_list, size_window, stride)
    
    # loading the 4D tensor test-----------------------------------------------
    files = pd.read_csv(os.path.join(path_test,'file_names.csv')).squeeze()
    files = [f[:-2]+'pkl' for f in files]
    
    data_list = load_data(files,path_test)
    data_list, len_ST = extend_TS(data_list,len_ST)
    data_list, chosen_resolution = resample(data_list,len_ST,chosen_resolution)
    data_list = rescale(data_list,var_quanti)
    
    tensor4D_test = tensor_allTS(data_list, size_window, stride)
    
    return tensor4D_train,tensor4D_test


tensor4D_train,tensor4D_test = main_4D_tensor(DATA_PATH+'/Train_clean_datasets',DATA_PATH+'/Test_clean_datasets')













