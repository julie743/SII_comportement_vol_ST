import numpy as np
import pandas as pd
import h5py
import os
from set_path import PATH,DATA_PATH,DATA_PATH_chosen

def load_1TS(file_name:str) : 
    '''
    load_1TS : loading one time series in format .h5 and put it to dataframe
    Input : 
      - file_name : path+name of the file to load
    Output : 
      - df : dataframe of one time series
      - var : list of variables
    '''
    
    f = h5py.File(file_name,'r')
    data = f['data']
      
    # loading column names : 
    var0 = list(map(str, data['block0_items']))
    var0 = list(map(lambda liste: liste[2:-1], var0))
    var1 = list(map(str, data['block1_items']))
    var1 = list(map(lambda liste: liste[2:-1], var1))
    var2 = list(map(str, data['block2_items']))
    var2 = list(map(lambda liste: liste[2:-1], var2))
    var3 = list(map(str, data['block3_items']))
    var3 = list(map(lambda liste: liste[2:-1], var3))
    var = np.r_[var0,var1,var2,var3]
      
    # loading dataframe content :
    df0 = pd.DataFrame(data['block0_values'],columns=var0)
    df1 = pd.DataFrame(data['block1_values'],columns=var1)
    df2 = pd.DataFrame(data['block2_values'],columns=var2)
    df3 = pd.DataFrame(data['block3_values'],columns=var3)
    df = pd.concat([df0,df1,df2,df3],axis=1)
    return df,var


def get_full_df(files):
    '''
    get_full_df : concatenate all time series in one dataframe
    Input : 
      - files : list of file's names
    Output : 
      - df : dataframe of all time series
    '''
    
    nb_scenario = len(files)
    _,var = load_1TS(files[0])
    df = pd.DataFrame(index=np.arange(0,nb_scenario),columns=var)
    for i in range(0,nb_scenario) :
        print(i)
        dfi,_ = load_1TS(files[i])
        for v in var : 
            df.loc[i,v] = dfi[v].to_numpy().flatten()
    return df


def df_to_dict(df) : 
    '''
    df_to_dict : convert dataframe to dictionnary
    Input : 
      - df : dataframe of all time series
    Output : 
      - dic : dictionnary of all time series
    '''
    
    dic = df.to_dict()
    var = list(df.columns)
    for v in var : 
        dic[v] = (list(dic[v].values()))
    return dic



def mkdir(directory) : 
    '''
    mkdir : check if a directory exists and create it if it does not
    Input : 
      - directory : name of the directory to create
    '''
    
    if not os.path.exists(directory):
        os.makedirs(directory)


#files = pd.read_csv('file_names.csv').squeeze()
#df = get_full_df(files)
#df.to_pickle("150_ST_test.pkl")

#df = pd.read_pickle(PATH+'initial_datasets/13_ST.pkl')
#cst_var = get_all_cst_var(df)
