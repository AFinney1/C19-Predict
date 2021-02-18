import pandas as pd
import numpy as np
#import conducto as C
#import streamlit as st
import tkinter
from tkinter import filedialog


def get_cases():
    window = tkinter.Tk()
    window.title("Case data explorer")
    
    f = filedialog.askopenfilename(initialdir="C19-Predict/Local_case_data/time_series_covid19_confirmed_US.csv",
                                    initialfile="time_series_covid19_confirmed_US.csv",
                                    title="Select a File",
                                    filetypes=(("CSV files", "*.csv*"),("all files", "*.*")), )
    
    #label_file_explorer.configure(text="File Opened: "+f)
   
    case_df = pd.read_csv(str(f))
    window.destroy()
    return case_df


#get_cases()