
#from os import name

from datetime import datetime, timezone
#import streamlit as st
#from matplotlib import pyplot as plt
#import conducto as C
from tkinter import Tk, filedialog

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from pandas.io.stata import StataReader
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.losses import MeanAbsoluteError

from Case_pipeline import get_cases


# Data importation test
case_df = get_cases()

print("Case DataFrame: ", case_df)
print("\nShape of case dataframe", case_df.shape)
columns = case_df.columns
print("\nColumns of case dataframe: ", columns)
print("\nUnique values for each variable:\n", case_df.nunique(axis=0))
#print("\nCase DataFrame Description: \n", case_df.describe())
#case_df = case_df.T
fips_list = case_df['FIPS']
print(case_df)
case_df.drop(columns[:5], inplace=True, axis=1)
print("Days only case dataframe: ", case_df)
#Train, test, split

#from sklearn.time_series_model_selection import train_test_split

#regions = case_df.groupby("Province_State")['Mississippi'].plot(kind = 'line') 
#

# regions.apply(print)

region_col = list(case_df.columns.values)
#region_col_ax = region_col.remove("Province_State")
region_col_ax = region_col[11:]
# print(region_col_ax)

plot_cols = region_col_ax
region = case_df.loc[case_df['Province_State'] == 'Mississippi']
print(region)

county_name = "Rankin"
county_list = region['Admin2']
county = region.loc[region['Admin2'] == county_name]
print(county)
columns = columns[5:11]

county.drop(columns, inplace=True, axis=1)

date_index = range(len(county))

county = county.transpose()

# county.reindex(date_index)
#county = county.reset_index(drop=True)
print(county)
#county.plot(y = region_col_ax)
# plt.plot(list(county))
def county_df():
    return county

county.plot(
    title=("Daily cases in " + county_name + " county"),
    legend=[county_name],
    xlabel="Date",
    ylabel='Confirmed Cases')
plt.legend([county_name])
#plt.show()


'''Training, Validation, and Test Split'''
county_length = len(county)
training_df = county[0:int(county_length*0.6)]
val_df = county[int(county_length*0.4):int(county_length*0.6)]
test_df = county[int(county_length*0.6):]

num_feature_days = county.shape[0]
print("Number of Days:", str(num_feature_days))


'''Normalization'''
training_mean = training_df.mean()
training_std = training_df.std()
print("TYPES: \n", type(training_std))


def normalize(df):
    normed_df = (df - training_mean)/training_std
    return normed_df


training_df = normalize(training_df)
print(type(training_df))
val_df = normalize(val_df)
test_df = normalize(test_df)

'''Denormalization'''


def denormalize(df):
    denormalized_df = training_std.values/(df.values - training_mean.values)
    return denormalized_df


'''Peek at the dataset's distribution of features'''
case_df.drop(columns[5:10], inplace=True, axis=1)
print(case_df)
'''
case_df_std = pd.DataFrame(normalize(training_std))
case_df_std = case_df_std.melt(var_name = 'Day', value_name = 'Normalized Cases')
plt.figure(figsize=(12,6))
ax = sns.violinplot(x = "Day", y = 'Normalized Cases', data = case_df_std)
#_ = ax.set_xticklabels(case_df.keys(), rotation = 90)
#plt.show()

#plot_features = region[region_col_ax]
#_ = plot_features.plot()
'''

#case_dataset = tf.data.Dataset.from_tensor_slices(case_df)
# print(case_dataset)
'''Data Windowing, heavily referenced from https://www.tensorflow.org/tutorials/structured_data/time_series, maybe modularized later'''


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 training_df=training_df, val_df=val_df, test_df=test_df,
                 label_columns=None):

        # raw data
        self.training_df = training_df
        self.val_df = val_df
        self.test_df = test_df

        # Determine label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(label_columns)}
        print("COLUMN INDICES ATTRIBUTE: ")
        print(self.column_indices)
        # Set up window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

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
       # ds = ds.map(self.split_window)

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


'''Create two windows'''
w1 = WindowGenerator(input_width=1, label_width=1, shift=1,
                     label_columns=columns)

w2 = WindowGenerator(input_width=1, label_width=1,
                     shift=1, label_columns=columns)


test_window = tf.stack([np.array(training_df[:w2.total_window_size]),
                        np.array(training_df[10:10+w2.total_window_size]),
                        np.array(training_df[20:20+w2.total_window_size])
                        ])

#example_inputs, example_labels = w2.split_window(test_window)


print("All shapes are: (batch, time, features)")
print(f'Window shape: {test_window.shape}')
#print(f'Inputs shape: {example_inputs.shape}')
#print(f'Inputs shape: {example_labels.shape}')


"time_series_modeling"
time_series_model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
time_series_model.add(layers.Embedding(input_dim=1000, output_dim=64), )
# Add a LSTM layer with 128 internal units.
time_series_model.add(layers.LSTM(128))
# Add a Dense layer with 1 unit.
time_series_model.add(layers.Dense(1))
time_series_model.summary()

y_pred = time_series_model.predict(test_df)

with tf.GradientTape() as tape:
    loss = tf.keras.backend.mean(tf.keras.backend.mean(
        tf.keras.losses.mse(y_true=test_df, y_pred=y_pred)))

time_series_model.compile(loss=tf.losses.MeanSquaredError(),
                          optimizer=tf.optimizers.Adam(),
                          metrics=[tf.metrics.MeanAbsoluteError()])

x = training_df.to_numpy()
x = x.reshape(1, -1)
y = val_df.to_numpy()
y = y.reshape(1, -1)

print(x.shape)
print(y.shape)


"""Model training"""
time_series_model.fit(x,
                      y,
                      batch_size=32,
                      epochs=1)


"""Model saving"""


def save_function():
    present = datetime.now()
    date = datetime.now(tz=timezone.utc).strftime('%a %b %d %H:%M:%S %Z %Y')
    model_filepath = "~/saved_models/"+str(date)
    time_series_model.save(model_filepath)
    return date, model_filepath


saved_model = save_function()

"""Model Prediction and Plotting"""
y_pred = pd.DataFrame(time_series_model.predict(test_df))
# y_pred.reshape(-1,1)
y_pred2 = pd.DataFrame(denormalize(y_pred))
#y_pred2 = y_pred2.T

#y_pred2 = pd.DataFrame(time_series_model.predict(test_df))
y_pred2.plot(
    title=("Projected confirmed cases in " + county_name + " county"),
    legend=[county_name],
    xlabel="Date",
    ylabel='Projected Confirmed Cases')
plt.legend([county_name])
plt.savefig(saved_model[1])
plt.show()
