# C19-Predict

This project is a rudimentary attempt to build a COVID-19 case prediction system primarily using TensorFlow and Keras.
Utilizing a simple LSTM neural network architecture, the script trains models on confirmed COVID-19 cases.

The neural network architecture and some of the feature engineering code are referenced and inspired from:   https://www.tensorflow.org/tutorials/structured_data/time_series 

COVID-19 case data are sourced from: https://coronavirus-resources.esri.com/datasets/628578697fb24d8ea4c32fa0c5ae1843_0/data?where=(Confirmed%20%3E%200)

This project is still a very early work in progress.

Future changes to the project will include:

- statistical comparisons of this project to mainstream case projections
- automated dataset updates to compare past predictions with recent data 
- improved user interface (first CLI, then a GUI will be developed)
- add cloud hosting for training new models from devices without GPUs
