# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:39:08 2026

@author: Gamhe
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
import gc
import time

def load_parquet_data(path, output_column='remaining_cycles'):
    data = pd.read_parquet(path)
    
    X = data.drop(columns=[output_column])
    y = data[output_column]
    
    return X, y



devices = ['Device5', 'Device4', 'Device3', 'Device2']

rmse_results = []
r2_results = []

for device in devices:
    
    X_test, y_test = load_parquet_data(os.path.join('Datasets', 'Parquet real', 'NASA', f'{device}.parquet'))
    
    remaining_devices = [x for x in devices if x != device]
    
    
    X_train_ctgan = []
    y_train_ctgan = []
    
    for temp_device in remaining_devices:
        X_temp, y_temp = load_parquet_data(os.path.join('Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'ctgan_synthetic_data_nasa_{temp_device}.parquet'))
        X_train_ctgan.append(X_temp)
        y_train_ctgan.append(y_temp)
        
    X_train_ctgan = pd.concat(X_train_ctgan, ignore_index=True)
    y_train_ctgan = pd.concat(y_train_ctgan, ignore_index=True)
    
    
    y_ctgan_scaler = MinMaxScaler()

    
    y_train_ctgan_scaled = pd.Series(y_ctgan_scaler.fit_transform(y_train_ctgan.values.reshape(-1, 1)).flatten(), name=y_train_ctgan.name, index=y_train_ctgan.index)
    y_test_ctgan_scaled = pd.Series(y_ctgan_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name, index=y_test.index)

    
    training_sets = ['CTGAN']
    
    for training_set in training_sets:
    
        fractions = [1.0]
        quantiles = [0.025, 0.975]
        
        times = []
        
        for fraction in fractions:
        
            print(f"Fitting with {fraction * 100}% of the available {training_set} data for {device}")
            

            if training_set == 'CTGAN':
                X_temp = X_train_ctgan.sample(frac=fraction, random_state=42)
                y_temp = y_train_ctgan_scaled.loc[X_temp.index]
            
            
            xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
            
            t0 = time.perf_counter()
            
            xgb.fit(X_temp, y_temp.squeeze())
            
            predictions = xgb.predict(X_test)
            
            t1 = time.perf_counter()
            
            times.append(t1 - t0)
            
            print(f"Metrics with {fraction * 100}% of the available {training_set} data for {device}:")
            if training_set == 'CTGAN':
                rmse = root_mean_squared_error(y_test_ctgan_scaled, predictions)
                r2 = r2_score(y_test_ctgan_scaled, predictions)
            
            rmse_results.append(rmse)
            print("RMSE:", rmse)
            r2_results.append(r2)
            print("R^2:", r2)
            
            del xgb
            gc.collect()
            torch.cuda.empty_cache()
            
print("Mean RMSE:", np.mean(rmse_results))
print("Std RMSE:", np.std(rmse_results))
print("Mean R^2:", np.mean(r2_results))
print("Std R^2:", np.std(r2_results))
print("Mean time:", np.mean(times))