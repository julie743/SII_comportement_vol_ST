# SII_comportement_vol_ST

nouveau lien meeting:
https://voip.siinergy.net/meet/florianmontels

Description des fichiers de code : 

**In the folder code/utilities :** 
  1) Data loading : 
  - `data_loading.py` : load the .h5 files provided by SII 
  - `retrieve_labels.ypnb` : identify each label corresponding to each time series
  - `set_path.py` : function to set path for loading the data (the other functions pick up their path from this file)

  2) First feature selection : 
  - `data_preprocessing.ypnb` : determine the type of data : qualitative/ quantitative
  - `correlations.py` : remove constant variables and keep only one variable from each group of highly correlated variables

  3) Dimension reduction with DCAE (Deep Convolutional Autoencoder) : 
  - `data_preparation.py` : data preparation : rescaling, converting qualitative variables to dummies, extend time series so that they all have the same length, resampling, slidding window, building a 4D matrix for the DCAE input
  - `model_DCAE.py` : 2 possible architectures of the DCAE, training and testing
 
**The notebooks :**
The codes were run on google collab in order to have access to a GPU to accelerate the training of the models. The following notebooks are using the functions implemented in the folder utilities :
  - `run_model_DCAE.ypnb`
  - `creates_variable_lists.ipynb`
  - `miniRocket_models`
 
To re-use our codes, the first thing to do is to modify the path defined in the file `code/utilities/set_path.py`. Once this path is set, all other files pick up the path from there. 

**In the folder results :** Results of the training of the DCAE and miniRocket models
  - results_DCAE
  - miniRocket_images 
  
**In the folder sorting_variables :** List of the variables selected after the preprocessing of the data. A .txt file in this folder describes the content of the .pkl files.


