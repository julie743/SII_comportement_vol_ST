# SII_comportement_vol_ST

nouveau lien meeting:
https://voip.siinergy.net/meet/florianmontels

Description des fichiers de code : 

1) Chargement des données : 
- data_loading.py : load the .h5 files provided by SII 
- retrieve_labels.ypnb : identify each label corresponding to each time series
- set_path : function to set path for loading the data (the other functions pick up their path from this file)

2) First feature selection : 
- data_preprocessing.ypnb : dtermine the type of data : qualitative/ quantitative
- correlations.py : remove constant variables and keep only one variable from each group of highly correlated variables

3) Dimension reduction with DCAE (Deep Convolutional Autoencoder) : 
- data_preparation.py : data preparation : rescaling, converting qualitative variables to dummies, extend time series so that they all have the same length, resampling, slidding window, building a 4D matrix for the DCAE input
- model_DCAE : 2 possible architectures of the DCAE, training and testing

to delete : dataVisualization (premier code pour voir à quoi ressemblent les données)
