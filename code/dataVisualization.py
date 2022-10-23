import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = '/home/julie/Documents/cours/5A/projet'
PATH_param = ROOT + '/data/parametres_xplane-xlsx_2022-10-04_0755'
#os.chdir(PATH)
data = pd.read_csv(PATH_param+"/Xplane_data.csv")
data.drop("Unnamed: 0",axis=1,inplace=True)
list_var0 = data.columns
print(len(list_var0))

# loading two datasets : 
PATH_data_V1 = ROOT + '/data/dataset_V1'
data1 = pd.read_csv(PATH_data_V1+'/TimeSeries_Scenario_1_ScenarioInstance1_Takeoff.csv')
list_var1 = data1.columns
print(len(list_var1))

data2 = pd.read_csv(PATH_data_V1+'/TimeSeries_Scenario_2_ScenarioInstance2_Takeoff.csv')
list_var2 = data2.columns
print(len(list_var2))

# get empty columns to delete them : 
empty_cols1 = list_var1[data1.isnull().all()]
print(len(empty_cols1))

empty_cols2 = list_var2[data2.isnull().all()]
print(len(empty_cols2))

# we can see that 48 columns are empty in both datasets (the same ones)=> we can delete them : 
data1.drop(empty_cols1,inplace=True,axis=1)
list_var1 = data1.columns
data2.drop(empty_cols2,inplace=True,axis=1)
list_var2 = data2.columns

# renormalize the data : 
scaler1 = StandardScaler()
scaler2 = StandardScaler()

scaler1.fit(data1)
data1 = pd.DataFrame(scaler1.transform(data1),columns=list_var1)

scaler2.fit(data2)
data2 = pd.DataFrame(scaler2.transform(data2),columns=list_var2)



