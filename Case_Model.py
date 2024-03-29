
#from os import name
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
columns = columns[5:11]
county.drop(columns, inplace = True, axis = 1)

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
        print("COLUMN INDICES ATTRIBUTE: ")
        print(self.column_indices)
        #Set up window parameters.
        self.input_width = input_width
        self.label_width = label_width 
        self.shift = shift 

        self.total_window_size = input_width + shift 

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

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



    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.training_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


    '''Split Window function '''
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
       # print(self.column_indices[name])
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:, self.column_indices[name]] for name in self.label_columns],
                axis = -1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels



    def plot(self, model=None, plot_col=columns, max_subplots = 3):
        inputs, labels = self.example
        plt.figure(figsize = (12,8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker=',', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else: 
                label_col_index = plot_col_index 

            if label_col_index is None:
                continue 

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n==0:
                plt.legend()

    plt.xlabel("Date")

'''Create two windows'''
w1 = WindowGenerator(input_width=1, label_width=1, shift=1,
label_columns = columns)

w2 = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=columns)


test_window = tf.stack([np.array(training_df[:w2.total_window_size]),
                        np.array(training_df[10:10+w2.total_window_size]),
                        np.array(training_df[20:20+w2.total_window_size])
                        ])

#example_inputs, example_labels = w2.split_window(test_window)
w2.plot()

print("All shapes are: (batch, time, features)") 
print(f'Window shape: {test_window.shape}')
#print(f'Inputs shape: {example_inputs.shape}')
#print(f'Inputs shape: {example_labels.shape}')





"Modeling"
model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))
# Add a Dense layer with 10 units.
model.add(layers.Dense(10))
model.summary()







