
from pandas.io.stata import StataReader
import pandas as pd 
import numpy as np 
import matplotlib
#from matplotlib import pyplot as plt
#import conducto as C
from tkinter import Tk
#import streamlit as st
from tkinter import filedialog
from Case_pipeline import get_cases 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn 

#Data importation test
case_df = get_cases()

print("Case DataFrame: ", case_df)
print("\nShape of case dataframe",case_df.shape)
columns = case_df.columns
print("\nColumns of case dataframe: ", columns)
print("\nUnique values for each variable:\n",case_df.nunique(axis=0))
#print("\nCase DataFrame Description: \n", case_df.describe())
#case_df = case_df.T
fips_list = case_df['FIPS']
print(case_df)
case_df.drop(columns[:5],inplace=True, axis = 1)

#Train, test, split

#from sklearn.model_selection import train_test_split

#regions = case_df.groupby("Province_State")['Mississippi'].plot(kind = 'line')
#

# regions.apply(print)

region_col = list(case_df.columns.values)
#region_col_ax = region_col.remove("Province_State")
region_col_ax = region_col[11:]
#print(region_col_ax)

plot_cols = region_col_ax
region = case_df.loc[case_df['Province_State'] == 'Mississippi']
print(region)

county_name = "Hinds"
county_list = region['Admin2']
county = region.loc[region['Admin2'] == county_name]
print(county)
county.drop(columns[5:11], inplace = True, axis = 1)

date_index = range(len(county))

county = county.T
#county.reindex(date_index)
#county = county.reset_index(drop=True)
print(county)
#county.plot(y = region_col_ax)
#plt.plot(list(county))
county.plot(
    title = ("Daily cases in " + county_name + " county"),
    legend = [county_name],
    xlabel = "Date",
    ylabel = 'Confirmed Cases')
plt.legend([county_name])
plt.show()


'''
for states in regions:
    #print(region[0])
    state = states[0]
    if state == 'Mississippi':
        #region = pd.DataFrame(region)
        #region = region.apply(pd.DataFrame)
        #print(region)
        date_time = pd.to_datetime(region_col_ax)
        plot_features = states[:300]
        counts = plot_features.plot()
       # plot_features.index = date_time
      #  _ = plot_features.plot(subplots=True)
     #   plot_features = region[plot_cols][11:]
      #  plot_features.index = date_time[11:]
'''


#plot_features = region[region_col_ax]
#_ = plot_features.plot()


#case_dataset = tf.data.Dataset.from_tensor_slices(case_df)
#print(case_dataset)

model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))
# Add a Dense layer with 10 units.
model.add(layers.Dense(10))
model.summary()


