#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 23:46:48 2022

@author: julie
"""

# loading the libraries : ----------------------------------------------------
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os
import glob
from itertools import combinations
from scipy.stats import chi2_contingency
from dython.nominal import associations

from sklearn.preprocessing import scale
from pandas.plotting import scatter_matrix

import data_loading as DL


# Folder containing the data : -----------------------------------------------
PATH = '/home/julie/Documents/cours/5A/projet/SII_comportement_vol_ST'
DATA_PATH = PATH + "/data/Dataset_V1_HDF5/"
os.chdir(DATA_PATH+'/initial_datasets/')

# loading the list of variables and df : -------------------------------------
var_quali = pd.read_pickle('VAR_QUALI_TAB.pkl')
var_quanti = pd.read_pickle('VAR_QUANTI_TAB.pkl')
df = pd.read_pickle('13_ST.pkl')

files = pd.read_csv('file_names.csv').squeeze()
data_list = []
for f in files : 
    df_1ST,_ =  DL.load_1TS(f)
    data_list.append(df_1ST)

# All usefull functions : ----------------------------------------------------

# from the list of variables, remove the ones that are constant and identical
# over all time series
def rm_all_cst_var(df,var_quali,var_quanti) :
    cst_var = DL.get_all_cst_var(df)
    var_quali_clean =  [v for v in var_quali if not v in cst_var]
    var_quanti_clean = [v for v in var_quanti if not v in cst_var]
    return var_quali_clean,var_quanti_clean

# transform qualitative variables to categorical 
def quali_to_categorical(data_list,var_quali):
    for df_1ST in data_list : 
        for v in var_quali : 
            df_1ST[v] = pd.Categorical(df_1ST[v],ordered=False)
    return data_list

# compute correlation matrix for quantitative variables 
def corr_quant_allTS(data_list,var_quanti) :
    corr_list = []
    for df_1ST in data_list :
        corr_list.append(df_1ST[var_quanti].corr())
    return corr_list

# compute correlation matrix for qualitative variables 
def corr_quali_allTS(data_list,var_quali) :
    corr_list = []
    for df_1ST in data_list :
        categorical_correlation = associations(df_1ST[var_quali])
        corr_list.append(categorical_correlation['corr']) 
    return corr_list

# compute correlation matrix for quantitative and qualitative variables 
def corr_mixte_allTS(data_list) :
    corr_list = []
    for df_1ST in data_list :
        complete_correlation = associations(df_1ST)
        corr_list.append(complete_correlation['corr'])
    return corr_list
            
def rm_corr_var(corr_list,var) :
    # remove the highly correlated variables : 
    to_drop = []
    for i in range(len(corr_list)) :
        upper_tri = corr_list[i].where(np.triu(np.ones(corr_list[i].shape),k=1).astype(np.bool))
        to_drop.append([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)])
    
    # we only delete the columns that can be deleted in all the Time Series   
    intersection = set(to_drop[0])
    for i in range(1,len(corr_list)) :
        intersection = intersection & set(to_drop[i])
        
    # update list of final qualitative variables : 
    var = [v for v in var if not v in intersection]
    
    return var

def time_ref(var_quanti) : 
    if 'sim/time/total_running_time_sec' in var_quanti :
        var_quanti.remove('sim/time/total_running_time_sec')
    
    if not 'sim/time/total_flight_time_sec' in var_quanti :
        var_quanti.append('sim/time/total_flight_time_sec')
    return var_quanti

## Operations in the correct order - main function : -------------------------
# 1. If we decide to treat all qualitative and quantitative variables separatly
def rm_correlated_var(df,data_list,var_quali,var_quanti) :
    # 1. remove the variables that are constant and equal between all TS 
    var_quali,var_quanti = rm_all_cst_var(df,var_quali,var_quanti)
    
    # 2. transform qualitative variables to categorical 
    data_list = quali_to_categorical(data_list,var_quali)
    
    # 3. handle highly correlated quantitative variables 
    corr_list_quanti = corr_quant_allTS(data_list,var_quanti)
    var_quanti = rm_corr_var(corr_list_quanti,var_quanti)
    
    # 3. handle highly correlated qualitative variables 
    corr_list_quali = corr_quali_allTS(data_list,var_quali)
    var_quali = rm_corr_var(corr_list_quali,var_quali)
    
    # 4. Update dataframes 
    var_quanti = time_ref(var_quanti) #choose the correct time reference
    for i in range(len(data_list)) :
        data_list[i] = data_list[i][np.r_[var_quanti,var_quali]]
    df = df[np.r_[var_quanti,var_quali]]
    
    # 5. Record results 
    os.chdir(DATA_PATH+'/clean_datasets/')
    pd.DataFrame(var_quanti).to_pickle('VAR_QUANTI_TAB.pkl')
    pd.DataFrame(var_quali).to_pickle('VAR_QUALI_TAB.pkl')
    df.to_pickle("13_ST.pkl")
    files_pkl = [f[:-3]+'.pkl' for f in files]
    for i in range(len(data_list)) : 
        data_list[i].to_pickle(files_pkl[i])
    return data_list, df, var_quali, var_quanti

# 2. If we decide to treat qualitative and quantitative variables togeter
def rm_correlated_mixt_var(df,data_list,var_quali,var_quanti) :
    # 1. remove the variables that are constant and equal between all TS 
    var_quali,var_quanti = rm_all_cst_var(df,var_quali,var_quanti)
    
    # 2. transform qualitative variables to categorical 
    data_list = quali_to_categorical(data_list,var_quali)
    
    # 3. handle highly correlated quantitative variables 
    corr_list_quanti = corr_quant_allTS(data_list,var_quanti)
    var_quanti = rm_corr_var(corr_list_quanti,var_quanti)
    for i in range(len(data_list)) : # update the dataframe
        data_list[i] = data_list[i][np.r_[var_quanti,var_quali]]
    
    # 3. handle highly correlated mixte variables 
    corr_list_quanti = corr_mixte_allTS(data_list)
    var_total = rm_corr_var(corr_list_quanti,np.r_[var_quanti,var_quali])
    
    # 4. Update dataframes and variable list
    for i in range(len(data_list)) :
        data_list[i] = data_list[i][var_total]
    df = df[var_total]
    
    var_quali =  list(set(var_total) & set(var_quanti))
    var_quali =  list(set(var_total) & set(var_quali))
    
    # 5. Record results 
    os.chdir(DATA_PATH+'/cleaned_datasets/')
    pd.DataFrame(var_quanti).to_pickle('VAR_QUANTI_TAB.pkl')
    pd.DataFrame(var_quali).to_pickle('VAR_QUALI_TAB.pkl')
    df.to_pickle("13_ST.pkl")
    for i in range(len(data_list)) : 
        data_list[i].to_pickle(files[i])
    
    return data_list, df, var_quali, var_quanti

#-----------------------------------------------------------------------------
# tests : 
data_list, df, var_quali, var_quanti = rm_correlated_var(df,data_list,var_quali,var_quanti)
