import numpy as np
import pandas as pd
import h5py
import os
import glob
import pickle 

# Folder containing the data : -----------------------------------------------
PATH = '/home/julie/Documents/cours/5A/projet'
DATA_PATH = PATH + "/data/Dataset_V1_HDF5"
#os.chdir(DATA_PATH) # on se place dans le dossier contenant les données 

# loaing one time series : 
def load_1TS(file_name:str) : 
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

# concatenate all time series in one dataframe : 
def get_full_df(files,var):
    nb_scenario = len(files)
    df = pd.DataFrame(index=np.arange(0,nb_scenario),columns=var)
    for i in range(0,nb_scenario) :
        dfi,_ = load_1TS(files[i])
        for v in var : 
            df.loc[i,v] = dfi[v].to_numpy().flatten()
    return df

# convert dataframe to dict : 
def df_to_dict(df) : 
    dic = df.to_dict()
    var = list(df.columns)
    for v in var : 
        dic[v] = (list(dic[v].values()))
    return dic

# function to get all the constant variables
def get_all_cst_var(df):
    var = list(df.columns)
    cst_var = []
    for v in var : 
        liste_allTS = []
        for line in df[v] : 
            liste_allTS.extend(line.tolist())
        if len(np.unique(liste_allTS)) == 1 : 
            cst_var.append(v)
    return cst_var

# function to remove the constant variables individually from each dataset : 
def remove_cst_var(cst_var) : 
    os.chdir(DATA_PATH+'initial_datasets/')
    files = glob.glob('*.h5')
    for f in files : 
        os.chdir(DATA_PATH+'initial_datasets/')
        df_1ST,_ = load_1TS(f) 
        df_1ST.drop(cst_var,axis=1,inplace=True)
        os.chdir(DATA_PATH+'cleaned_datasets/')
        df_1ST.to_pickle(f[:-3]+'_without_cst_var.pkl') 

def mkdir(directory) : 
    if not os.path.exists(directory):
        os.makedirs(directory)


#df = pd.read_pickle(PATH+'initial_datasets/13_ST.pkl')
#cst_var = get_all_cst_var(df)
#remove_cst_var(cst_var) 