import streamlit as st
from Case_Analysis import *
import tkinter
import matplotlib
matplotlib.use("Tkagg")

st.write("""
# COVID-19 Confirmed Case Projection App
""")

case_df = get_cases()
st.write(case_df)
preprocessed_data = preprocessing(case_df)
county_name = preprocessed_data[-1]
training_df, val_df, test_df, training_mean, training_std = train_test_val_split(preprocessed_data=preprocessed_data)
training_df = normalize(training_df, training_mean, training_std)
val_df = normalize(val_df, training_mean, training_std)
test_df = normalize(test_df, training_mean, training_std)
model = build_time_series_model(test_df, training_df, val_df)
saved_model = model_save_function(model)
model_test= test_predictions(model, test_df, training_mean, training_std)
predicted_cases = plot_case_predictions(model_test, county_name, saved_model)
st.write(predicted_cases)

st.write(predicted_cases)


