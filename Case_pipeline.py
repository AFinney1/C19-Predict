import pandas as pd
import numpy as np
#import conducto as C
#import streamlit as st
import tkinter
from tkinter import filedialog
import os
import requests
import re


def get_cases():
    '''
    window = tkinter.Tk()
    window.title("Case data explorer")
    window.withdraw()
    '''
    cwd = os.getcwd()
    csv_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    req = requests.get(csv_url, allow_redirects=True)
    open('Local_case_data/time_series_covid19_confirmed_US.csv', 'wb').write(req.content)
    #f = filedialog.askopenfilename(initialdir="/Local_case_data/",initialfile="time_series_covid19_confirmed_US.csv", title="Select a File",filetypes=(("CSV files", "*.csv*"),("all files", "*.*")), )
    cwd = os.getcwd()
    f = cwd+"/Local_case_data/time_series_covid19_confirmed_US.csv"
    #label_file_explorer.configure(text="File Opened: "+f)
    #f.close()
    case_df = pd.read_csv(str(f))
    #window.mainloop
    #window.destroy()
    return case_df


test = get_cases()