import pandas as pd 
import numpy as np 
#import conducto as C
from tkinter import Tk
import streamlit as st
from tkinter import filedialog
import Tk

def get_cases():
    f = filedialog.askopenfilename(initialdir = "C19-Predict",
                                        title = "Select a File",
                                        filetypes = (("CSV files",
                                        "*.csv*"),
                                        ("all files",
                                        "*.*")))
    #label_file_explorer.configure(text="File Opened: "+f)
    window = Tk()
    window.title("Case data explorer")
    case_df = pd.read_csv("/Local_case_data/"+str(f))
    return case_df


get_cases()




