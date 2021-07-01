# C19-Predict

This is a side-project I work on in my spare time to help me learn more about data science and general software development: a COVID-19 case prediction system written primarily in Python.

Utilizing a simple Long Short-Term Memory (LSTM) neural network architecture, the script trains models on confirmed COVID-19 cases and trys to predict the daily cases for some time into the future.

This project can be run by typing in the terminal:

`streamlit run Case_Dash_st.py`

 with the aid of libraries including TensorFlow, Pandas, numpy, and others (see the requirements.txt or use the Dockerfile).

The neural network architecture and some of the feature engineering code are referenced and inspired from:   https://www.tensorflow.org/tutorials/structured_data/time_series. 

COVID-19 case data are sourced from: https://coronavirus-resources.esri.com/datasets/628578697fb24d8ea4c32fa0c5ae1843_0/data?where=(Confirmed%20%3E%200)

The following gif depicts the project in-action at a glance:

![alt-text](C19-predict_demo3.gif)

This project is still a work in progress.

Future changes to the project will include:

- higher quality predictions and more sophisticated modeling
- statistical comparisons of this project to mainstream case projections
- improved user interface
- add cloud hosting for training new models from devices without GPUs
- improved exception handling
