"""
02/07/2024
Some functions to process data for the Rice model
"""

import numpy as np
import pandas as pd

def convert_rice_class_to_number(rice_df):
    # Convert the column listing the type of rice from
    # a string to a number
    # Returns a numpy array of the outputs as a number

    # Create Dictionary to hold the numeric data
    class_array = []

    # Cycle through the raw data values
    for value in enumerate(rice_df["Class"]):

        # Add a one or a zero depending on the Rice type
        if 'Cammeo' in str(value):
            class_array.append(0)
        elif 'Osmancik' in str(value):
            class_array.append(1)
            
    return np.array(class_array).reshape(len(class_array), 1)

def normalize_dataframe(df):
    # Normalize the dataframe

    df_dropped = pd.DataFrame.copy(df, deep=True)
    df_normalized = (df_dropped - df_dropped.min())/(df_dropped.max() - df_dropped.min())

    return df_normalized