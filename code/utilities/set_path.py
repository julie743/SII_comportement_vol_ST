#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:14:10 2022

@author: julie
"""
'''
This code is for indicating the path where to load the data. In this way you 
only need to change the path here, and if you add 
"from set_path import PATH,DATA_PATH,DATA_PATH_TRAIN,DATA_PATH_TEST"
at the beginning of all your other files, they will be referenced with the same
path 
'''
# Folder containing the data : -----------------------------------------------
PATH = '/content/drive/My Drive/SII_comportement_vol_ST'

# choose wich dataset to work on : 
#DATA_PATH = PATH + "/data/Dataset_V2" # separation 50/50
DATA_PATH = PATH + "/data/Dataset_V2" # separation 80/20

# loading training or test data : 
DATA_PATH_TRAIN = DATA_PATH + "/Train_clean_datasets"
DATA_PATH_TEST = DATA_PATH + "/Test_clean_datasets"
