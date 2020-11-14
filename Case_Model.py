
import pandas as pd 
import numpy as np 
import matplotlib
#from matplotlib import pyplot as plt
#import conducto as C
from tkinter import Tk
#import streamlit as st
from tkinter import filedialog
from Case_pipeline import get_cases 
import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn 

#Data importation test
case_df = get_cases()
print("Case DataFrame: ", case_df)
print("\nShape of case dataframe",case_df.shape)
print("\nColumns of case dataframe: ",case_df.columns)
print("\nUnique values for each variable:\n",case_df.nunique(axis=0))
print("\nCase DataFrame Description: \n", case_df.describe())



#Train, test, split



case_dataset = tf.data.Dataset.from_tensor_slices(case_df)
print(case_dataset)

model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()


