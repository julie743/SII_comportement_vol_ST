#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 23:46:48 2022

@author: julie
"""

# loading the libraries : ----------------------------------------------------
import pandas as pd
import numpy as np
from dython.nominal import associations


from set_path import PATH,DATA_PATH
import data_loading as DL

# choose to work on training or test set : 
DATA_PATH_chosen = DATA_PATH + "/Train" 
#DATA_PATH_chosen = DATA_PATH + "/Test" 

# loading the list of variables and df : -------------------------------------
var_quali = pd.read_pickle(DATA_PATH+'/VAR_QUALI_TAB.pkl')
var_quanti = pd.read_pickle(DATA_PATH+'/VAR_QUANTI_TAB.pkl')

files = pd.read_csv(DATA_PATH_chosen+'/file_names.csv').squeeze()
data_list = []


for f in files : 
    df_1ST,_ =  DL.load_1TS(DATA_PATH_chosen+'/'+f)
    data_list.append(df_1ST)

# All usefull functions : ----------------------------------------------------


def get_all_cst_var(data_list,var):
    '''
    get_all_cst_var : function to get all the variables that are constant through 
    time and equal between all time series
    Input : 
      - data_list : list containing the time series (loaded just above)
      - var : list of ll variables
    Output : 
      - cst_var : list of the variables that are consta
    '''
    
    cst_var = []
    for v in var : 
        list_allTS = []
        for TS in data_list : 
            list_allTS.extend(TS[v].to_list())
        if len(np.unique(list_allTS))==1:
            cst_var.append(v)
    pd.DataFrame(cst_var).to_csv(DATA_PATH_chosen+"cst_variables.csv",index=False)
    return cst_var


def rm_all_cst_var(data_list,var_quali,var_quanti) :
    '''
    rm_all_cst_var : from the list of variables, remove the ones that are constant 
    and equal over all time series
    Input : 
      - data_list : list containing the time series (loaded just above)
      - var_quali : list of qualitative variables
      - var_quanti : list of quantitative variables
    Output : 
      - var_quali_clean : list of non constant qualitative variables
      - var_quanti_clean : list of non constant quantitative variables
    '''
    
    cst_var = get_all_cst_var(data_list,var=np.r_[var_quali,var_quanti])
    var_quali_clean =  [v for v in var_quali if not v in cst_var]
    var_quanti_clean = [v for v in var_quanti if not v in cst_var]
    return var_quali_clean,var_quanti_clean



def rm_warnings(var_quali,var_quanti) :
    '''
    rm_warnings : function to remove the warnings (aircraft's answer to failure)
    Input : 
      - var_quali : list of qualitative variables
      - var_quanti : list of quantitative variables
    Output : 
      - var_quali : list of qualitative variables without warnings
      - var_quanti : list of quantitative variables without warnings
    '''
    
    warnings = open(DATA_PATH+'/listeWarnings.txt').readlines()
    warnings = [w[:-2]for w in warnings] #get rid of the '\n'
    var_quali = var_quali.to_list()
    var_quanti = var_quanti.to_list()
    
    for w in warnings : 
        found = False
        for v in var_quali : 
            if v[:len(w)] == w : 
                var_quali.remove(v)
                found = True
        if not found : 
            for v in var_quanti : 
                if v[:len(w)] == w : 
                    var_quanti.remove(v)
      
    return var_quali,var_quanti



def quali_to_categorical(data_list,var_quali):
    '''
    quali_to_categorical : transform qualitative variables to categorical 
    Input : 
      - data_list : list containing the time series (loaded just above)
      - var_quali : list of qualitative variables
    Output : 
      - data_list : same as the input with correct categorical types
    '''
    
    for i in range(len(data_list)) : 
        for v in var_quali : 
            data_list[i][v] = pd.Categorical(data_list[i][v],ordered=False)
    return data_list



def corr_quant_allTS(data_list,var_quanti) :
    '''
    corr_quant_allTS : compute the correlation between quantitative variables and 
    from a group of highly correlated variables (correlation>0.95) delete all 
    variables except one
    Input : 
      - data_list : list containing the time series (loaded just above)
      - var_quanti : list of quantitative variables
    Output : 
      - var_quanti : list of uncorrelated quantitative variables
    '''
    
    var_quant_to_drop = set(var_quanti)
    for i in range(len(data_list)) :
        corr_mat = data_list[i][var_quanti].corr()
        upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape),k=1).astype(np.bool))
        var_quant_to_drop = var_quant_to_drop  & set([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)])
    var_quanti = [v for v in var_quanti if not v in var_quant_to_drop]
    return var_quanti



def corr_quali_allTS(data_list,var_quali) :
    '''
    corr_quali_allTS : compute the correlation between qualitative variables and 
    from a group of highly correlated variables (correlation>0.95) delete all 
    variables except one
    Input : 
      - data_list : list containing the time series (loaded just above)
      - var_quali : list of qualitative variables
    Output : 
      - var_quali : list of uncorrelated qualitative variables
    '''
    
    var_quali_to_drop = set(var_quali)
    for i in range(len(data_list)) :
        corr_mat = associations(data_list[i][var_quali])
        upper_tri = corr_mat['corr'].where(np.triu(np.ones(corr_mat['corr'].shape),k=1).astype(np.bool))
        var_quali_to_drop = var_quali_to_drop & set([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)])
    var_quali = [v for v in var_quali if not v in var_quali_to_drop]
    return var_quali      



def time_ref(var_quanti) : 
    '''
    time_ref : function to make sur we keep the time reference we want 
    (sim/time/total_flight_time_sec), because both time reference are completly 
    equivalent 
    Input : 
      - var_quanti : list of quantitative variables
    Output : 
      - var_quanti : list of quantitative variables with correct time reference
    '''
    
    if 'sim/time/total_running_time_sec' in var_quanti :
        var_quanti.remove('sim/time/total_running_time_sec')
    
    if not 'sim/time/total_flight_time_sec' in var_quanti :
        var_quanti.append('sim/time/total_flight_time_sec')
    return var_quanti



def rm_correlated_var(data_list,var_quali,var_quanti) :
    '''
    rm_correlated_var : main function which calls the previous subfunctions in the
    correct order to select 
    Input : 
      - data_list : list containing the time series (loaded just above)
      - var_quali : list of qualitative variables
      - var_quanti : list of quantitative variables
    Output : 
      - var_quali_clean : list of final qualitative variables
      - var_quanti_clean : list of final quantitative variables
    '''
    
    # 1. remove the variables that are warnings or that are constant and equal 
    # between all TS 
    var_quali,var_quanti = rm_warnings(var_quali,var_quanti)
    var_quali,var_quanti = rm_all_cst_var(data_list,var_quali,var_quanti)
    
    # 2. transform qualitative variables to categorical 
    data_list = quali_to_categorical(data_list,var_quali)
    
    # 3. handle highly correlated quantitative variables 
    var_quanti = corr_quant_allTS(data_list,var_quanti)
    var_quanti = time_ref(var_quanti) #choose the correct time reference
    pd.DataFrame(var_quanti).to_pickle(DATA_PATH_chosen+'_clean_datasets/'+'VAR_QUANTI_TAB.pkl')
    
    # 3. handle highly correlated qualitative variables 
    var_quali = corr_quali_allTS(data_list,var_quali)
    pd.DataFrame(var_quali).to_pickle(DATA_PATH_chosen+'_clean_datasets/'+'VAR_QUALI_TAB.pkl')
    
    # 4. Update dataframes 
    for i in range(len(data_list)) :
        data_list[i] = data_list[i][np.r_[var_quanti,var_quali]]
    
    # 5. Record results 
    files_pkl = [f[:-3]+'.pkl' for f in files]
    for i in range(len(data_list)) : 
        data_list[i].to_pickle(files_pkl[i])
    return var_quali, var_quanti


#-----------------------------------------------------------------------------
# tests : 
#data_list, var_quali, var_quanti = rm_correlated_var(data_list,var_quali,var_quanti)




