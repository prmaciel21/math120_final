# Math 120 Python Final: Facial Recognition with PCA

## Project Overview
This Python project investigates facial recognition by applying Principal Component Analysis (PCA) to reduce image dimensionality and extract meaningful facial features. A classification model is then trained on these reduced representations to distinguish individuals based on their facial images. The project includes dataset preprocessing, feature extraction, model training, and evaluation of recognition performance. 

## Dataset

Source https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?select=lfw_allnames.csv

The files used from this set were the following:

* ```lfw_allnames.csv```: Has 2 columns, 1 corresponding to the name of the person and the other corresponding to how many images of that person exist in the dataset
* ```lfw-deepfunneled/lfw-deepfunneled/```: This is a directory containing folders of people, where that folder contains all the images pertaining to them

The csv file was used to filter all the people with only 20-30 images. From there, I took all the folders in ```lfw-deepfunneled/``` pertaining to those people in order to perform pca on them.

## Project Structure
```
math120_final
  *__pychache__/           #directory and file created
    *data.cpython-312.pyc  #by colab
  *cleaned_data/           #images of people who have between 20-30
    *...names/             #in the dataset
  *README.md
  *data.py                 #used for preprocessing and cleaning the data in the dataset
  *final_notebook.ipynb    #final notebook
  *pca_model.py            #functions that create pca model
```

## Requirements

* Google Colab

## How to run

### Google Colab
1. Download ```final_notebook.ipynb```
2. Upload to Google Colab
3. Run the first cell to set up the environment
4. Dependency cells will import data.py and pca_model.py

## Learning Objectives Demonstrated

* Loaded data with Pandas
* Used Kaggle API to retrieve data
* Loaded images from the dataset
* Used Python libraries to perform data analysis
* Implemented PCA using ```sklearn.decomposition```
* Used SVC classifier from ```sklearn.svm``` to train a classifier on PCA reduced features

## Summary

This repository contains a facial recognition project that explores the use of Principal Component Analysis (PCA) and supervised classification. Facial image data is obtained using the Kaggle API and preprocessed for analysis in Python. PCA is used to generate eigenfaces and reduce the dimensionality of the data, and a Support Vector Classifier (SVC) is trained to predict identities based on the reduced features. The project demonstrates the full workflow from data acquisition and preprocessing to model training and evaluation.

## Author
Pedro Ramos-Maciel

Math 120 - Fall 2025
