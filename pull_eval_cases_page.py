
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
import streamlit as st
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from pandas.io.stata import StataReader
from Case_pipeline import get_cases
from PIL import Image
import plotly.express as px


device = "cpu" #if torch.cuda.is_available() else "cpu"
print(f"{device} " " is available")


def preprocessing(case_df):
    print("Preprocessing...")
    columns = case_df.columns
    #datecolumns = case_df.columns()
    case_df.drop(columns[:5], inplace=True, axis=1)
    region_col = list(case_df.columns.values)
    region_col_ax = region_col[11:]
    plot_cols = region_col_ax
    st.text("Case data last updated: " + str(case_df.columns[-1]))
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
    return region_col, region_col_ax, region, county, initial_startdate, val_startdate, test_startdate, county_name, state_name, lastdate

def plot_county_cases(county, county_name):
    case_plot = county.plot(
        title=("Daily cases in " + county_name + " county"),
        legend=[county_name],
        xlabel="Date",
        ylabel='Confirmed Cases')
    plt.legend([county_name])
    return(case_plot)
#plt.show()



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



def denormalize(df, training_mean, training_std ):
    #denormalized_df = training_std.values/(df.values - training_mean.values)
    denormalized_df = training_std.values*df.values + training_mean.values
    return denormalized_df


#case_df.drop(columns[5:10], inplace=True, axis=1)
#print(case_df)
def torch_data_loader(x_train, x_val, x_test):

    train_features = torch.Tensor(x_train.values)
    val_features = torch.Tensor(x_val.values)
    test_features = torch.Tensor(x_test.values)
    print(train_features.shape)
    train_targets = torch.Tensor(x_train.values)
    val_targets = torch.Tensor(x_val.values)
    test_targets = torch.Tensor(x_test.values)
    batch_size = 1

    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_loader)
    print(train_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loader_one = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_loader, val_loader, test_loader, test_loader_one

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros

        #x = torch.FloatTensor(x)
        h0 = torch.zeros(self.layer_dim, len(x), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, len(x), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
  
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

    



def get_model(model, model_params):
    models = {
        "lstm": LSTMModel
    }
    return models.get(model.lower())(**model_params)
    
class Optimization:
    def __init__(self, model, loss_fn, optimizer, state_county):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.model_p = "torch_models/"+state_county+"/"+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))##f'torch_models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

    
    def train_step(self, x):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        #print(x)
        x = torch.stack(x).to(device)
        yhat = self.model(x).to(device)

        # Computes loss
        y = x
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = self.model_p

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            #print(train_loader) 
            print("Training...")
            for x_batch, y_batch in train_loader:
                x_batch = torch.tensor(x_batch[0])
                #print(type(x_batch))# y_batch)
                #x_batch = torch.tensor(x_batch)
                x_batch = x_batch.view([batch_size, n_features]).to(device)
                y_batch = y_batch.to(device)
               #print(x_batch, y_batch)
                loss = self.train_step((x_batch, y_batch))
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
        torch.save(self.model.state_dict(), model_path)
        
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values 

    def train_if_empty(self, model_dir, train_loader, val_loader, batch_size = 1,  n_epochs = 10):
        "if directory is empty, train model, else load one"
        print(model_dir)
        if not os.listdir(model_dir):
            self.train(train_loader, val_loader, batch_size = batch_size, n_epochs=n_epochs)
           # print(self.model_p)
           # self.save_model(self.model.state_dict())#, "torch_models/"+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            plt.plot(self.train_losses, label="Training loss")
            plt.plot(self.val_losses, label="Validation loss")
            print(self.train_losses, self.val_losses)
            plt.legend()
            plt.title("Losses")
            plt.savefig(model_dir+"/Losses.png")
            fig = plt
            st.pyplot(fig)
        else:
            
            #print("DOOT")
            print(os.listdir(model_dir))
            #model = torch.load()
            self.model.load_state_dict(torch.load(model_dir+"/"+os.listdir(model_dir)[-1]))
            fig = Image.open(model_dir+"/"+"Losses.png")
            st.image(fig)
   


        
        # save the figure. if losses are zero after loading model, then load the figure for the losses
        # next is to plot the cases 

        
        
# y_pred.reshape(-1,1)


#y_pred2 = pd.DataFrame(time_series_model.predict(test_df))




def app():
    st.title("Covid-19 Cases")
    case_df = get_cases()
    preprocessed_data = preprocessing(case_df)
    county_name = preprocessed_data[-3]
    state_name = preprocessed_data[-2]
    state_county = state_name+"/"+county_name
    state_county_dir = "torch_models/"+state_name+"/"+county_name
    if state_county not in os.listdir("torch_models"):
        try:   
            os.makedirs(state_county_dir)
        except:
            pass

    
    

    #print(model_test)
    #plot_cases(model_test, county_name, saved_model, lastdate)
 

if __name__ == "__main__":
    app()
