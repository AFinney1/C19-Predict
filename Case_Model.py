
import pandas as pd 
import numpy as np 
import matplotlib
#from matplotlib import pyplot as plt
#import conducto as C
from tkinter import Tk
import streamlit as st
from tkinter import filedialog
from Case_pipeline import get_cases 
import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf
import sklearn 

#Data importation test
case_df = get_cases()
print("Case DataFrame: ",
    case_df)
print("\nShape of case dataframe",case_df.shape)
print("\nColumns of case dataframe: ",case_df.columns)

print("\nUnique values for each variable:\n",case_df.nunique(axis=0))
print("\nCase DataFrame Description: \n", case_df.describe())



#Train, test, split



case_dataset = tf.data.Dataset.from_tensor_slices(c)
print(case_dataset)