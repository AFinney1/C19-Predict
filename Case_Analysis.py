
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
    try:
        state_name = st.text_input("Enter state name ",) #'Mississippi') #or 'Mississippi'
    except:
        st.error("Please enter the proper name of the state without whitespace(e.g. 'Texas', not 'texas ')")
    region = case_df.loc[case_df['Province_State'] == state_name]
    #print(region)
    st.write(region)
    county_list = region['Admin2']
    #default_county = county_list.iloc[0]
    try:
        county_name = st.text_input("Enter county name ", )#"Hinds")# or default_county
    except:
        st.error("Please enter the proper name of the county without whitespace(e.g. 'Austin', not 'austin '")   
    county = region.loc[region['Admin2'] == county_name]
    columns = columns[5:11]
    county.drop(columns, inplace=True, axis=1)
    date_index = range(len(county))
    county = county.transpose()
    lastdate = (str(region.columns[-1]))
    col_length = len(region.columns)
    initial_startdate = (str(region.columns[6]))
    test_startdate = (str(region.columns[int(col_length*.803)]))
    val_startdate = (str(region.columns[int(col_length*.4)]))
    print(test_startdate)
    return region_col, region_col_ax, region, county, initial_startdate, val_startdate, test_startdate, county_name, lastdate

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
#Need to add test to check if training and validation sets have the same size due to ValueError
def train_test_val_split(preprocessed_data):
    county = preprocessed_data[3]
    county_length = len(county)
    training_df = county[0:int(county_length*0.4)] # training set for model parameter optimization
    try:
        val_df = county[int(county_length*0.4):int(county_length*0.8)] #validation set used to find optimal model hyperparameters
    except:
        val_df = county[int(county_length*0.4):int(county_length*0.798)] #second splice needed in case previous split doesn't work based on odd vs even days.
    test_df = county[int(county_length*0.8):] #test set used to determine model performance in general
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
    time_series_model.add(layers.LSTM(128, return_sequences=True))
    time_series_model.add(layers.Dropout(0.2))
    time_series_model.add(layers.LSTM(128, return_sequences=True))
    time_series_model.add(layers.Dropout(0.2))
    time_series_model.add(layers.Dense(1))
    time_series_model.summary()

    def df_num_reshape(df):
        num = df.to_numpy()
        num = num.reshape(1,-1)
        print(str(num.shape))# f"{df=}")
        return(num)

    y_pred = time_series_model.predict(test_df)

    '''Loss function'''
    with tf.GradientTape() as tape:
        loss = tf.keras.backend.mean(tf.keras.backend.mean(
            tf.keras.losses.mse(y_true=test_df, y_pred=y_pred)))
            
    '''Model compilation'''
    time_series_model.compile(loss=tf.losses.MeanSquaredError(),
                            optimizer=tf.optimizers.Adam(),
                            metrics=[tf.metrics.MeanAbsoluteError()])

    '''conversion of dataframes to numpy arrays and appropriate reshaping'''


    x = df_num_reshape(training_df)
    y = df_num_reshape(val_df)
    #df_num_reshape(test_df)


    """Model training"""
    time_series_model.fit(x,
                        y,
                        batch_size=5,
                        epochs=8)
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
   # tdf = test_df.to_numpy()
    #tf = tdf.reshape(-1,1)
    #print(tf, tf.shape)
    model_output = time_series_model.predict(test_df)
    model_output = pd.DataFrame(model_output.reshape(test_df.shape))
    print(test_df.shape)
    #df_temp = pd.DataFrame(pd.np.empty((test_df.shape)))
    denorm_predictions = pd.DataFrame(denormalize(model_output, training_mean, training_std))
    print("PREDICTIONS:")
    print(denorm_predictions)
    return(denorm_predictions)
#y_pred2 = y_pred2.T

#y_pred2 = pd.DataFrame(time_series_model.predict(test_df))
def plot_cases(cases, county_name, startdate, lastdate, title = "Projected COVID-19 cases", history = True):
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
    import streamlit as st
    from datetime import datetime
    import seaborn as sns
    sns.set_theme()
    #dates.DayLocator(bymonthday = range(1,182), interval = len(predictions))
    print("THESE ARE MY DATE VARIABLES: ", startdate, type(startdate), lastdate,  type(lastdate))
    datearray = pd.date_range(start = startdate, end = lastdate, freq = 'D').strftime("%m-%d-%Y")
    #tickvalues = list(range(predictions))
    print("THIS IS THE DATEARRAY ",datearray)

   # plt.axis(xmin = 0, xmax = 10)
    cases = cases.T
    print(cases.shape)
    casedate_difference = (cases.shape)[1] - len(datearray.tolist()) 
    cases.drop(labels = cases.columns[:casedate_difference], inplace = True, axis = 1)
    cases.columns = datearray.tolist()
    cases = cases.T
    #predictions.reset_index()
  #  predictions.set_index(datearray)

    
    if history == False:
        plt.title("Predicted COVID-19 cases in " + county_name + " county")
        plt.plot(cases, color = 'r', linestyle = '--')
    elif history == True:
        plt.title("Past Covid-19 cases in " + county_name + " county")
        plt.plot(cases, color = 'blue')
        #cases = denormalize(cases)
    plt.xlabel("Date")
    plt.ylabel("COVID-19 cases")
    plt.xticks(np.arange(0, len(datearray), 15.0))
    
    #plt.ticklabel_format(useOffset=False, style='plain')
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
    plot_cases(model_test, county_name, saved_model, lastdate)
 

if __name__ == "__main__":
    main()
