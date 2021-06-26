
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
import streamlit as st
from Case_pipeline import get_cases

# Data importation test


def preprocessing(case_df):
    print("Preprocessing...")
    columns = case_df.columns
    #datecolumns = case_df.columns()
    case_df.drop(columns[:5], inplace=True, axis=1)
    region_col = list(case_df.columns.values)
    region_col_ax = region_col[11:]
    plot_cols = region_col_ax
    st.text("Case database last updated: " + str(case_df.columns[-1]))
    state_name = st.text_input("Enter state name ",) #'Mississippi') #or 'Mississippi'
    region = case_df.loc[case_df['Province_State'] == state_name]
    print(region)
    st.write(region)
    county_list = region['Admin2']
    default_county = county_list.iloc[0]
    county_name = st.text_input("Enter county name ", )#"Hinds")# or default_county   
    county = region.loc[region['Admin2'] == county_name]
    columns = columns[5:11]
    county.drop(columns, inplace=True, axis=1)
    date_index = range(len(county))
    county = county.transpose()
    lastdate = (str(region.columns[-1]))
    return region_col, region_col_ax, region, county, county_name, lastdate

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
def train_test_val_split(preprocessed_data):
    county = preprocessed_data[3]
    county_length = len(county)
    training_df = county[0:int(county_length*0.6)] # training set for model parameter optimization
    val_df = county[int(county_length*0.4):int(county_length*0.6)] #validation set used to find optimal model hyperparameters
    test_df = county[int(county_length*0.6):] #test set used to determine model performance in general
    num_feature_days = county.shape[0]
    print("Number of Days:", str(num_feature_days))
    training_mean = training_df.mean()
    training_std = training_df.std()
    print("TYPES: \n", type(training_std))
    return(training_df, val_df, test_df, training_mean, training_std)

def normalize(df, training_mean, training_std):
    normed_df = (df - training_mean)/training_std
    return normed_df


'''Denormalization'''
def denormalize(df, training_mean, training_std ):
    #denormalized_df = training_std.values/(df.values - training_mean.values)
    denormalized_df = training_std.values*df.values + training_mean.values
    return denormalized_df


'''Peek at the dataset's distribution of features'''
#case_df.drop(columns[5:10], inplace=True, axis=1)
#print(case_df)

def build_time_series_model(test_df, training_df, val_df):
    time_series_model = tf.keras.Sequential()
    time_series_model.add(layers.Embedding(input_dim=1000, output_dim=64), )
    time_series_model.add(layers.LSTM(1280))
    time_series_model.add(layers.Dropout(0.2))
    time_series_model.add(layers.LSTM(1280))
    time_series_model.add(layers.Dropout(0.2))
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
    model_filepath = 'saved_models'#"/saved_models/"+str(date)
    time_series_model.save(model_filepath)
    return date, model_filepath



"""Model Prediction and Plotting"""

# y_pred.reshape(-1,1)
def test_predictions(time_series_model, test_df, training_mean, training_std):
    model_output = pd.DataFrame(time_series_model.predict(test_df))
    denorm_predictions = pd.DataFrame(denormalize(model_output, training_mean, training_std))
    return(denorm_predictions)
#y_pred2 = y_pred2.T

#y_pred2 = pd.DataFrame(time_series_model.predict(test_df))
def plot_case_predictions(predictions, county_name, saved_model, lastdate):
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
    import streamlit as st
    from datetime import datetime
    import seaborn as sns
    sns.set_theme()
    #dates.DayLocator(bymonthday = range(1,182), interval = len(predictions))
    print("THIS IS MY LASTDATE VARIABLE: ", lastdate,  type(lastdate))
    datearray = pd.date_range(start = lastdate, end = "03-21-21", freq = 'D').strftime("%m-%d-%Y")
    #tickvalues = list(range(predictions))
    print("THIS IS THE DATEARRAY ",datearray)

   # plt.axis(xmin = 0, xmax = 10)
    predictions = predictions.T
    predictions.columns=datearray.tolist()
    predictions = predictions.T
    #predictions.reset_index()
  #  predictions.set_index(datearray)

    plt.plot(predictions)
    plt.title("Projected COVID-19 cases in " + county_name + " county")
    plt.xlabel("Date")
    plt.ylabel("Projected COVID-19 cases in " + county_name + " county")
    plt.xticks(np.arange(0, 150, 10.0))
    #plt.ylim([0, int(predictions.max())])
    plt.legend([county_name])
    #plt.locator_params(axis = 'x', nbins=len(predictions)/10)
   # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
    plt.gcf().autofmt_xdate()
    #plt.show()
    #plt.autofmt_xdate()
   # plt.savefig(saved_model[1])
    #plt.savefig("Predicted Cases")
    #fig = plt.figure()

    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot()
  
    
   

def main():
    case_df = get_cases()
    preprocessed_data = preprocessing(case_df)
    county_name = preprocessed_data[-2]
    lastdate = preprocessed_data[-1]
    training_df, val_df, test_df, training_mean, training_std = train_test_val_split(preprocessed_data=preprocessed_data)
    training_df = normalize(training_df, training_mean, training_std)
    val_df = normalize(val_df, training_mean, training_std)
    test_df = normalize(test_df, training_mean, training_std)
    model = build_time_series_model(test_df, training_df, val_df)
    saved_model = model_save_function(model)
    model_test= test_predictions(model, test_df, training_mean, training_std)
    print(model_test)
    plot_case_predictions(model_test, county_name, saved_model, lastdate)
 

if __name__ == "__main__":
    main()
