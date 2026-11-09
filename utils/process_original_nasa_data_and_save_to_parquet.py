# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:19:50 2025

@author: Gamhe
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read_mat(mat_file_path, eol):
    """
    Reads data from a .mat file into a DataFrame.
    """
    
    #data = loadmat(mat_file_path)['measurement']['steadyState'][0][0][0, 1:][0][2]
    data = loadmat(mat_file_path)['measurement']['steadyState'][0][0][0, 1:][:]['timeDomain']
    
    records = []

    for entry in data:
        struct = entry[0][0]
        record = {}
        for field in struct.dtype.names:
            val = struct[field]

            # Recursively unwrap nested arrays until we reach a scalar
            while isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()

            # Convert non-numeric values (e.g., NaN) safely
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = np.nan

            record[field] = val

        records.append(record)

    df = pd.DataFrame(records)
    df.drop(columns=['heatSinkTemperature', 'internalTemperature', 'ambientTemperature'], inplace=True)
    df = df.head(eol)
    df.insert(0, 'cycle', df.index)
    df['remaining_cycles'] = df.index[::-1].values

    df = df.astype('float64')
    
    return df


def export_to_parquet(dataframe, output_file_path):
    """
    Exports a DataFrame to a Parquet file.
    """
    
    dataframe.to_parquet(output_file_path, index=False)
    
    print(f"Exported {len(dataframe)} rows to {output_file_path}")


df2 = read_mat(os.path.join('..', 'Datasets', 'Original', 'NASA', 'Device 2', 'Device2  1.mat'), eol = 10741)
export_to_parquet(df2, os.path.join('..', 'Datasets', 'Parquet real', 'NASA', 'Device2.parquet'))

df3 = read_mat(os.path.join('..', 'Datasets', 'Original', 'NASA', 'Device 3', 'Device3  1.mat'), eol = 9119)
export_to_parquet(df3, os.path.join('..', 'Datasets', 'Parquet real', 'NASA', 'Device3.parquet'))

df4 = read_mat(os.path.join('..', 'Datasets', 'Original', 'NASA', 'Device 4', 'Device4  1.mat'), eol = 10284)
export_to_parquet(df4, os.path.join('..', 'Datasets', 'Parquet real', 'NASA', 'Device4.parquet'))

df5 = read_mat(os.path.join('..', 'Datasets', 'Original', 'NASA', 'Device 5', 'Device5  1.mat'), eol = 10232)
export_to_parquet(df5, os.path.join('..', 'Datasets', 'Parquet real', 'NASA', 'Device5.parquet'))