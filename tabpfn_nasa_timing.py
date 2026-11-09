# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:43:59 2026

@author: Gamhe
"""

import os

with open("key.txt", "r") as f:
    os.environ["TABPFN_TOKEN"] = f.read().strip()

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabpfn import TabPFNRegressor
import time
import gc

SEEDS = [42]

def load_parquet_data(path, output_column='remaining_cycles'):
    data = pd.read_parquet(path)
    
    X = data.drop(columns=[output_column])
    y = data[output_column]
    
    return X, y

if torch.cuda.is_available():
    print("TabPFN will use the GPU")
else:
    print("TabPFN will use the CPU")

devices = ['Device5', 'Device4', 'Device3', 'Device2']

quantiles = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
             0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]
timing = {}  # (training_set, fraction) -> {'fit': [], 'predict': [], 'n_train': []}

for seed in SEEDS:

    for device in devices:
        
        X_test, y_test = load_parquet_data(os.path.join('Datasets', 'Parquet real', 'NASA', f'{device}.parquet'))
        
        remaining_devices = [x for x in devices if x != device]
        
        
        
        X_train_real = []
        y_train_real = []
        
        for temp_device in remaining_devices:
            X_temp, y_temp = load_parquet_data(os.path.join('Datasets', 'Parquet real', 'NASA', f'{temp_device}.parquet'))
            X_train_real.append(X_temp)
            y_train_real.append(y_temp)
            
        X_train_real = pd.concat(X_train_real, ignore_index=True)
        y_train_real = pd.concat(y_train_real, ignore_index=True)
        
        
        
        
        X_train_gaussian = []
        y_train_gaussian = []
        
        for temp_device in remaining_devices:
            X_temp, y_temp = load_parquet_data(os.path.join('Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'gaussian_synthetic_data_nasa_{temp_device}.parquet'))
            X_train_gaussian.append(X_temp)
            y_train_gaussian.append(y_temp)
            
        X_train_gaussian = pd.concat(X_train_gaussian, ignore_index=True)
        y_train_gaussian = pd.concat(y_train_gaussian, ignore_index=True)
        
        
        
        
        X_train_ctgan = []
        y_train_ctgan = []
        
        for temp_device in remaining_devices:
            X_temp, y_temp = load_parquet_data(os.path.join('Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'ctgan_synthetic_data_nasa_{temp_device}.parquet'))
            X_train_ctgan.append(X_temp)
            y_train_ctgan.append(y_temp)
            
        X_train_ctgan = pd.concat(X_train_ctgan, ignore_index=True)
        y_train_ctgan = pd.concat(y_train_ctgan, ignore_index=True)
        
        
        
        
        X_train_real_plus_gaussian = pd.concat([X_train_real, X_train_gaussian], ignore_index=True)
        y_train_real_plus_gaussian = pd.concat([y_train_real, y_train_gaussian], ignore_index=True)
        
        X_train_real_plus_ctgan = pd.concat([X_train_real, X_train_ctgan], ignore_index=True)
        y_train_real_plus_ctgan = pd.concat([y_train_real, y_train_ctgan], ignore_index=True)







        y_real_scaler = MinMaxScaler()
        y_gaussian_scaler = MinMaxScaler()
        y_ctgan_scaler = MinMaxScaler()
        y_real_plus_gaussian_scaler = MinMaxScaler()
        y_real_plus_ctgan_scaler = MinMaxScaler()
        
        y_train_real_scaled = pd.Series(y_real_scaler.fit_transform(y_train_real.values.reshape(-1, 1)).flatten(), name=y_train_real.name, index=y_train_real.index)
        y_train_gaussian_scaled = pd.Series(y_gaussian_scaler.fit_transform(y_train_gaussian.values.reshape(-1, 1)).flatten(), name=y_train_gaussian.name, index=y_train_gaussian.index)
        y_train_ctgan_scaled = pd.Series(y_ctgan_scaler.fit_transform(y_train_ctgan.values.reshape(-1, 1)).flatten(), name=y_train_ctgan.name, index=y_train_ctgan.index)
        y_train_real_plus_gaussian_scaled = pd.Series(y_real_plus_gaussian_scaler.fit_transform(y_train_real_plus_gaussian.values.reshape(-1, 1)).flatten(), name=y_train_real_plus_gaussian.name, index=y_train_real_plus_gaussian.index)
        y_train_real_plus_ctgan_scaled = pd.Series(y_real_plus_ctgan_scaler.fit_transform(y_train_real_plus_ctgan.values.reshape(-1, 1)).flatten(), name=y_train_real_plus_ctgan.name, index=y_train_real_plus_ctgan.index)
        
        y_test_real_scaled = pd.Series(y_real_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name, index=y_test.index)
        y_test_gaussian_scaled = pd.Series(y_gaussian_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name, index=y_test.index)
        y_test_ctgan_scaled = pd.Series(y_ctgan_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name, index=y_test.index)
        y_test_real_plus_gaussian_scaled = pd.Series(y_real_plus_gaussian_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name, index=y_test.index)
        y_test_real_plus_ctgan_scaled = pd.Series(y_real_plus_ctgan_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name, index=y_test.index)
        
        
        
        
        training_sets = ['Real', 'Gaussian copula', 'CTGAN', 'Real + Gaussian copula', 'Real + CTGAN']
        _y_test_map = {
            'Real': y_test_real_scaled, 'Gaussian copula': y_test_gaussian_scaled,
            'CTGAN': y_test_ctgan_scaled, 'Real + Gaussian copula': y_test_real_plus_gaussian_scaled,
            'Real + CTGAN': y_test_real_plus_ctgan_scaled
        }

        for training_set in training_sets:

            fractions = [10, 50, 100, 500, 1000]

            for fraction in fractions:
            
                print(f"Fitting with {fraction} samples of the available {training_set} data for {device}")
                
                if training_set == 'Real':
                    X_temp = X_train_real.sample(n=fraction, random_state=42)
                    y_temp = y_train_real_scaled.loc[X_temp.index]
                elif training_set == 'Gaussian copula':
                    X_temp = X_train_gaussian.sample(n=fraction, random_state=42)
                    y_temp = y_train_gaussian_scaled.loc[X_temp.index]
                elif training_set == 'CTGAN':
                    X_temp = X_train_ctgan.sample(n=fraction, random_state=42)
                    y_temp = y_train_ctgan_scaled.loc[X_temp.index]
                elif training_set == 'Real + Gaussian copula':
                    X_temp = X_train_real_plus_gaussian.sample(n=fraction, random_state=42)
                    y_temp = y_train_real_plus_gaussian_scaled.loc[X_temp.index]
                elif training_set == 'Real + CTGAN':
                    X_temp = X_train_real_plus_ctgan.sample(n=fraction, random_state=42)
                    y_temp = y_train_real_plus_ctgan_scaled.loc[X_temp.index]
                
                
                tabpfn = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu", random_state=seed, ignore_pretraining_limits=True)
                t_key = (training_set, fraction)
                if t_key not in timing:
                    timing[t_key] = {'fit': [], 'predict': [], 'n_train': []}
                t0 = time.perf_counter()
                tabpfn.fit(X_temp, y_temp.squeeze())
                timing[t_key]['fit'].append(time.perf_counter() - t0)
                timing[t_key]['n_train'].append(X_temp.shape[0])

                t0 = time.perf_counter()
                predictions = tabpfn.predict(X_test)
                quantile_predictions = tabpfn.predict(X_test, output_type="quantiles", quantiles=quantiles)
                timing[t_key]['predict'].append(time.perf_counter() - t0)
                
                del tabpfn
                gc.collect()
                torch.cuda.empty_cache()

with open(os.path.join('modeling_results', 'NASA', 'timing_report_extra.txt'), 'w') as f:
    f.write("TabPFN timing report: NASA dataset (mean +/- std across seeds and devices, leave-one-out)\n\n")
    ts_list = ['Real', 'Gaussian copula', 'CTGAN', 'Real + Gaussian copula', 'Real + CTGAN']
    fr_list = [10, 50, 100, 500, 1000]
    for fraction in fr_list:
        f.write(f"Fraction {fraction:.2f}:\n")
        for ts in ts_list:
            t_key = (ts, fraction)
            if t_key in timing and timing[t_key]['fit']:
                n_mean = int(np.mean(timing[t_key]['n_train']))
                ft = timing[t_key]['fit']
                pt = timing[t_key]['predict']
                f.write(f"  {ts} ({fraction} samples, ~{n_mean} samples): "
                        f"fit={np.mean(ft):.4f}+/-{np.std(ft):.4f} s, "
                        f"predict={np.mean(pt):.4f}+/-{np.std(pt):.4f} s\n")
        f.write("\n")
print("Timing report saved to modeling_results/NASA/timing_report_extra.txt")

