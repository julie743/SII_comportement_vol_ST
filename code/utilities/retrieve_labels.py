# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:50:33 2022

@author: Lila
"""

import pandas as pd
import os
import re

from set_path import DATA_PATH,DATA_PATH_TRAIN,DATA_PATH_TEST 

def retrieve_scenario_id (path,files):
  '''
  This function retrieve the scenario id of each time series. 
  For example, if the first time serie name in files is: 
  "TimeSeries_Scenario_00063_ScenarioInstanceListTakeOffAndTurnV2_id_0003.h5"
  then, the scenario id will be 63. 

  Inputs:
  - files: file containing the list of files names 
  - data_path: where to pick up the file

  Ouputs:
  - scenario_id: list of scenario id. 
  '''
  os.chdir(path)
  file_names = pd.read_csv(files)

  # Extract the scenario id of each time series
  scenario_id = []
  for i in range(len(file_names)):
    row_i = file_names.iloc[i].values[0] #extract text in row i
    nb = [int(s) for s in re.findall(r'\d+', row_i)] #all numbers in row i
    scenario_id.append(int(nb[0])) #add the 1st number to the list
  
  return scenario_id



def retrieve_all_labels (path_labels,file_labels):
  '''
  This function retrieve the information of all labels (test and train).

  Inputs:
  - file_labels: file containing the labels' information.
  - path_labels: where to pick up the file.

  Outpus:
  - Labels: dataframe of labels information.
    * Index: 
      scenario_id: scenario id.
    * Columns:
      scenario_name_parti: name of the particular scenario,
      scenario_name: manoeuvre performed,
      error_name: error name (or None if no error) 
      error_time: error time in sec (or None if no error), 
      is_error: presence of an error (boolan).
  '''
  os.chdir(path_labels)
  with open(file_labels) as f:
      GroundTruth = f.readlines()

  #all information contained in GroundTruth
  Labels = {}
  for i in range(len(GroundTruth)):
    #label_i: all information contained row i of GroundTruth
    #(without spaces and commas)
    label_i = list(map(str.strip, GroundTruth[i].split(',')))
    #if the scenario has an error, error=1, otherwise error=0.
    error = 1*(label_i[3] != 'None') + 0
    Labels[int(label_i[0])] = label_i[1:] + [error]

  # Transform dict to dataframe
  dfLabels = pd.DataFrame(Labels).T
  dfLabels.rename(columns = {0:"scenario_name_parti",1:"scenario_name",
               2:"error_name",3:"error_time",4:"is_error"},inplace = True)
  dfLabels.index.names = ['scenario_id']
  
  # Put correct type
  dfLabels["is_error"] = dfLabels["is_error"].astype(int)
  dfLabels["is_error"] = pd.Categorical(dfLabels["is_error"],ordered = False)
  dfLabels["scenario_name"] = pd.Categorical(dfLabels["scenario_name"],ordered = False)
  # Remove word 'Test' from "scenario_name"
  dfLabels["scenario_name"] =  dfLabels["scenario_name"].apply(lambda x: re.sub("Test", '', x)) 
  
  
  return dfLabels


def main_retrieve_labels():
    '''
    This function calls the above function in the correct order.
    
    Outpus:
    - train_labels: labels of the train set
    - test_labels: labels of the test set
    '''
    
    scenario_id_train = retrieve_scenario_id(DATA_PATH_TRAIN,"file_names.csv")
    scenario_id_test = retrieve_scenario_id(DATA_PATH_TEST,"file_names.csv")
    
    all_labels = retrieve_all_labels(DATA_PATH,'GroundTruth.txt')
    
    train_labels = all_labels.loc[scenario_id_train]
    train_labels.reset_index(inplace=True)
    
    test_labels = all_labels.loc[scenario_id_test]
    test_labels.reset_index(inplace=True)

    return train_labels, test_labels
