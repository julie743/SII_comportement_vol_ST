# SII_comportement_vol_ST

nouveau lien meeting:
https://voip.siinergy.net/meet/florianmontels

Description of the code files: 

**In the folder code/utilities:** 
  1) Data loading: 
  - `data_loading.py`: load the .h5 files provided by SII 
  - `retrieve_labels.py`: identify each label corresponding to each time series
  - `set_path.py`: function to set path for loading the data (the other functions pick up their path from this file)

  2) First feature selection: 
  - `correlations.py`: remove constant variables and keep only one variable from each group of highly correlated variables

  3) Dimension reduction with DCAE (Deep Convolutional Autoencoder): 
  - `data_preparation.py`: data preparation : rescaling, converting qualitative variables to dummies, extend time series so that they all have the same length, resampling, slidding window, building a 4D matrix for the DCAE input
  - `model_DCAE.py`: 2 possible architectures of the DCAE, training and testing
  
  4) Classification of multivariate time series with MiniRocket:
  - `offline_MiniRocket_v2.py`: data loading and formatting, computation of MiniRocket features, training and prediction.
  - `grid_search.py`:  gird search method to tune the parameters of MiniRocket. 
  - `save_results.py`: writes MiniRocket results into an excel table.
 
**notebooks in the folder code:**
The codes were run on google collab in order to have access to a GPU to accelerate the training of the models. The following notebooks are using the functions implemented in the folder utilities:
  - `creates_variable_lists.ipynb`: determine the type of data : qualitative/ quantitative
  - `data_proprocessing.ipynb`: preprocess the datasets for the DCAE or MiniRocket.
  - `run_model_DCAE.ipynb`: run the DCAE to reduce the dataset
  - `run_model_MiniRocket.ipynb`: run the model MiniRocket and predicts the classification.
 
To re-use our codes, the first thing to do is to modify the path defined in the file `code/utilities/set_path.py`. Once this path is set, all other files pick up the path from there. 

**In the folder results:** Results of the training of the DCAE and miniRocket models
  - `results_DCAE`: plots for DCAE training.
  - `miniRocket_images`: confusion matrices for MiniRocket predictions. 
  
**In the folder sorting_variables:** List of the variables selected after the preprocessing of the data. A .txt file in this folder describes the content of the .pkl files.


