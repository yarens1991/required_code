# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:24:36 2025

@author: Gamhe
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(csv_file_path):
    """
    Reads data from a .csv file into a DataFrame.
    """
    
    df = pd.read_csv(csv_file_path)
    
    df.rename(columns={'consumed_life': 'remaining_cycles'}, inplace=True)
    df['remaining_cycles'] = 1 - df['remaining_cycles']
    
    #Uncomment if is needed to convert NaN entries to numeric values
    df = df.astype('float64')
    
    return df


def export_to_parquet(dataframe, output_file_path):
    """
    Exports a DataFrame to a Parquet file.
    """
    
    dataframe.to_parquet(output_file_path, index=False)
    
    print(f"Exported {len(dataframe)} rows to {output_file_path}")


folders = os.listdir(os.path.join('..', 'Datasets', 'Original', 'RISE'))
for folder in folders:
    files = os.listdir(os.path.join('..', 'Datasets', 'Original', 'RISE', folder))
    for file in files:
        df = read_csv(os.path.join('..', 'Datasets', 'Original', 'RISE', folder, file))
        export_to_parquet(df, os.path.join('..', 'Datasets', 'Parquet real', 'RISE', file[13:-3] + "parquet"))