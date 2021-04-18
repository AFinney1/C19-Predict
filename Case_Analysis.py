
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


def preprocessing(case_df):
    print("Preprocessing...")
    columns = case_df.columns
    case_df.drop(columns[:5], inplace=True, axis=1)
    region_col = list(case_df.columns.values)
    region_col_ax = region_col[11:]
    plot_cols = region_col_ax
    state_name = input("Enter state name")
    region = case_df.loc[case_df['Province_State'] == state_name]
    print(region)
    county_name = input("Enter county name")
    county_list = region['Admin2']
    county = region.loc[region['Admin2'] == county_name]
    columns = columns[5:11]
    county.drop(columns, inplace=True, axis=1)
    date_index = range(len(county))
    county = county.transpose()
    return region_col, region_col_ax, region, county, county_name

def plot_county_cases(county, county_name):
    case_plot = county.plot(
        title=("Daily cases in " + county_name + " county"),
        legend=[county_name],
        xlabel="Date",
        ylabel='Confirmed Cases')
    plt.legend([county_name])
    return(case_plot)
#plt.show()


'''Training, Validation, and Test Split'''
def train_test_val_split(preprocessed_data, county):
    county_length = len(county)
    training_df = county[0:int(county_length*0.6)]
    val_df = county[int(county_length*0.4):int(county_length*0.6)]
    test_df = county[int(county_length*0.6):]
    num_feature_days = county.shape[0]
    print("Number of Days:", str(num_feature_days))
    training_mean = training_df.mean()
    training_std = training_df.std()
    print("TYPES: \n", type(training_std))
    return(training_df, val_df, test_df, training_std)

def normalize(df, training_mean, training_std):
    normed_df = (df - training_mean)/training_std
    return normed_df


'''Denormalization'''
def denormalize(df, training_std, training_mean):
    denormalized_df = training_std.values/(df.values - training_mean.values)
    return denormalized_df


'''Peek at the dataset's distribution of features'''
#case_df.drop(columns[5:10], inplace=True, axis=1)
#print(case_df)

def build_time_series_model(test_df, training_df, val_df):
    time_series_model = tf.keras.Sequential()
    time_series_model.add(layers.Embedding(input_dim=1000, output_dim=64), )
    time_series_model.add(layers.LSTM(128))
    time_series_model.add(layers.Dense(1))
    time_series_model.summary()
    y_pred = time_series_model.predict(test_df)
    with tf.GradientTape() as tape:
        loss = tf.keras.backend.mean(tf.keras.backend.mean(
            tf.keras.losses.mse(y_true=test_df, y_pred=y_pred)))
    '''Model compilation'''
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
    return time_series_model

def model_save_function(time_series_model):
    present = datetime.now()
    date = datetime.now(tz=timezone.utc).strftime('%a %b %d %H:%M:%S %Z %Y')
    model_filepath = "~/saved_models/"+str(date)
    time_series_model.save(model_filepath)
    return date, model_filepath



"""Model Prediction and Plotting"""

# y_pred.reshape(-1,1)
def test_predictions(time_series_model, test_df):
    model_output = pd.DataFrame(time_series_model.predict(test_df))
    denorm_predictions = pd.DataFrame(denormalize(model_output))
    return(denorm_predictions)
#y_pred2 = y_pred2.T

#y_pred2 = pd.DataFrame(time_series_model.predict(test_df))
def plot_case_predictions(predictions, county_name, saved_model):
    predictions.plot(
        title=("Projected confirmed cases in " + county_name + " county"),
        legend=[county_name],
        xlabel="Date",
        ylabel='Projected Confirmed Cases')
    plt.legend([county_name])
    plt.savefig(saved_model[1])
    plt.show()

def main():
    case_df = get_cases()
    preprocessed_data = preprocessing(case_df)
    training_df, val_df, test_df = train_test_val_split(preprocessed_data=preprocessed_data)
    training_df = normalize(training_df)
    val_df = normalize(val_df)
    test_df = normalize(test_df)
    model = build_time_series_model(test_df, training_df, val_df)
    saved_model = model_save_function(model)
    test_predictions(model, test_df)
    



    '''Normalization'''
 

if __name__ == "__main__":
    main()
