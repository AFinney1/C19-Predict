
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
print("Days only case dataframe: ", case_df)
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

county_name = "Rankin"
county_list = region['Admin2']
county = region.loc[region['Admin2'] == county_name]
print(county)
county.drop(columns[5:11], inplace = True, axis = 1)

date_index = range(len(county))

county = county.transpose()
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


'''Training, Validation, and Test Split'''
county_length = len(county)
training_df = county[0:int(county_length*0.7)]
val_df = county[int(county_length*0.7):int(county_length*0.9)]
test_df = county[int(county_length*0.9):]

num_feature_days = county.shape[0]
print("Number of Days:", str(num_feature_days))


'''Normalization'''
training_mean = training_df.mean()
training_std = training_df.std()
print("TYPES: \n",type(training_std))
def normalize(df):
    normed_df = (df - training_mean)/training_std
    return normed_df
training_df = normalize(training_df)
print(type(training_df))
val_df = normalize(val_df)
test_df = normalize(test_df)

'''Peek at the dataset's distribution of features'''
case_df.drop(columns[5:10], inplace = True, axis = 1)
print(case_df)
case_df_std = pd.DataFrame(normalize(training_std))
case_df_std = case_df_std.melt(var_name = 'Day', value_name = 'Normalized Cases')
plt.figure(figsize=(12,6))
ax = sns.violinplot(x = "Day", y = 'Normalized Cases', data = case_df_std)
#_ = ax.set_xticklabels(case_df.keys(), rotation = 90)
#plt.show()

#plot_features = region[region_col_ax]
#_ = plot_features.plot()


#case_dataset = tf.data.Dataset.from_tensor_slices(case_df)
#print(case_dataset)
'''Data Windowing, heavily referenced from https://www.tensorflow.org/tutorials/structured_data/time_series, maybe modularized later'''
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                training_df = training_df, val_df = val_df, test_df = test_df,
                label_columns = None):

    #raw data
        self.training_df = training_df
        self.val_df = val_df
        self.test_df = test_df 

        #Determine label column indices
        self.label_columns = label_columns 
        if label_columns is not None:
            self.label_columns_indices = {name:i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(label_columns)}

        #Set up window parameters.
        self.input_width = input_width
        self.label_width = label_width 
        self.shift = shift 

        self.total_window_size = input_width + shift 

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window
        )[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f"Total window size:{self.total_window_size}",
            f"Input indices: {self.input_indices}",
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
label_columns =[])



model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))
# Add a Dense layer with 10 units.
model.add(layers.Dense(10))
model.summary()


