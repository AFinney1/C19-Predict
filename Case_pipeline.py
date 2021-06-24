import pandas as pd
import numpy as np
#import conducto as C
#import streamlit as st
import tkinter
from tkinter import filedialog
import os

#Need to add code to automatically update covid case data
def get_cases():
    window = tkinter.Tk()
    window.title("Case data explorer")
    window.withdraw()
    cwd = os.getcwd()
    #f = cwd+"/Local_case_data/time_series_covid19_confirmed_US.csv"
    f = filedialog.askopenfilename(initialdir="/Local_case_data/",initialfile="time_series_covid19_confirmed_US.csv", title="Select a File",filetypes=(("CSV files", "*.csv*"),("all files", "*.*")), )
    #label_file_explorer.configure(text="File Opened: "+f)
    #f.close()
    case_df = pd.read_csv(str(f))
    #window.mainloop
    #window.destroy()
    return case_df


test = get_cases()
#need to write actual tests