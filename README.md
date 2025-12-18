# Math 120 Python Final: Facial Recognition with PCA

## Project Overview
This Python project investigates facial recognition by applying Principal Component Analysis (PCA) to reduce image dimensionality and extract meaningful facial features. A classification model is then trained on these reduced representations to distinguish individuals based on their facial images. The project includes dataset preprocessing, feature extraction, model training, and evaluation of recognition performance. 

## Dataset

Source https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?select=lfw_allnames.csv

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

## Author
Pedro Ramos-Maciel

Math 120 - Fall 2025
