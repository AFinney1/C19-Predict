
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
columns = case_df.columns
print("Days only case dataframe: ", case_df)
print(case_df)



def main():
    print("Case DataFrame: ", case_df)
    print("\nShape of case dataframe", case_df.shape)
    print("\nColumns of case dataframe: ", columns)
    print("\nUnique values for each variable:\n", case_df.nunique(axis=0))
    #print("\nCase DataFrame Description: \n", case_df.describe())
    #case_df = case_df.T
    fips_list = case_df['FIPS']


def preprocessing(case_df):
    print("Preprocessing...")
    case_df.drop(columns[:5], inplace=True, axis=1)
    region_col = list(case_df.columns.values)
    region_col_ax = region_col[11:]
    plot_cols = region_col_ax
    region = case_df.loc[case_df['Province_State'] == 'Mississippi']
    print(region)
    county_name = "Rankin"
    county_list = region['Admin2']
    county = region.loc[region['Admin2'] == county_name]
    columns = columns[5:11]
    county.drop(columns, inplace=True, axis=1)
    date_index = range(len(county))
    county = county.transpose()
    def plot_county_cases():
        county.plot(
            title=("Daily cases in " + county_name + " county"),
            legend=[county_name],
            xlabel="Date",
            ylabel='Confirmed Cases')
        plt.legend([county_name])
    county_plot = plot_county_cases
    return region_col, region_col_ax, region, county
#plt.show()


'''Training, Validation, and Test Split'''
county_length = len(county)
training_df = county[0:int(county_length*0.6)]
val_df = county[int(county_length*0.4):int(county_length*0.6)]
test_df = county[int(county_length*0.6):]
num_feature_days = county.shape[0]
print("Number of Days:", str(num_feature_days))



training_mean = training_df.mean()
training_std = training_df.std()
print("TYPES: \n", type(training_std))

def normalize(df):
    normed_df = (df - training_mean)/training_std
    return normed_df


'''Normalization'''
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


"time_series_modeling"
time_series_model = tf.keras.Sequential()
time_series_model.add(layers.Embedding(input_dim=1000, output_dim=64), )
time_series_model.add(layers.LSTM(128))
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


if __name__ == "__main__":
    main()
