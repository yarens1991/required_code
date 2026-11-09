# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 01:08:00 2026

@author: Gamhe
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import gc

def load_parquet_data(path, output_column='remaining_cycles'):
    data = pd.read_parquet(path)

    X = data.drop(columns=[output_column])
    y = data[output_column]

    return X, y



files = os.listdir(os.path.join('Datasets', 'Parquet real', 'RISE'))
test_devices = [x[0:6] for x in files if x.startswith("r5")]
remaining_devices = [x[0:6] for x in files if not x.startswith("r5")]




X_train_real = []
y_train_real = []

for temp_device in remaining_devices:
    X_temp, y_temp = load_parquet_data(os.path.join('Datasets', 'Parquet real', 'RISE', f'{temp_device}.parquet'))
    X_train_real.append(X_temp)
    y_train_real.append(y_temp)

X_train_real = pd.concat(X_train_real, ignore_index=True)
y_train_real = pd.concat(y_train_real, ignore_index=True)



y_real_scaler = MinMaxScaler()

y_train_real_scaled = pd.Series(y_real_scaler.fit_transform(y_train_real.values.reshape(-1, 1)).flatten(), name=y_train_real.name, index=y_train_real.index)



training_sets = ['Real']

for training_set in training_sets:

    fractions = [1.0]
    quantiles = [0.025, 0.975]

    for fraction in fractions:

        print(f"Fitting with {fraction * 100}% of the available {training_set} data")

        if training_set == 'Real':
            X_temp = X_train_real.sample(frac=fraction, random_state=42)
            y_temp = y_train_real_scaled.loc[X_temp.index]


        rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(X_temp, y_temp.squeeze())

        rmse_results = []
        r2_results = []

        for device in test_devices:
            X_test, y_test = load_parquet_data(os.path.join('Datasets', 'Parquet real', 'RISE', f'{device}.parquet'))

            if training_set == 'Real':
                y_test_scaled = pd.Series(y_real_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name, index=y_test.index)


            prediction_batches = []

            for start in range(0, len(X_test), 10000):
                temp_pred = rf.predict(X_test.iloc[start:start + 10000])
                prediction_batches.append(temp_pred)

                del temp_pred
                torch.cuda.empty_cache()

            predictions = np.concatenate(prediction_batches, axis=0)

            # predictions = tabpfn.predict(X_test)
            # quantile_predictions = tabpfn.predict(X_test, output_type="quantiles", quantiles=quantiles)

            print(f"Metrics for {device} using {fraction * 100}% of the available {training_set} data as context:")
            rmse = root_mean_squared_error(y_test_scaled, predictions)
            r2 = r2_score(y_test_scaled, predictions)

            rmse_results.append(rmse)
            print("RMSE:", rmse)
            r2_results.append(r2)
            print("R^2:", r2)

        del rf
        gc.collect()
        torch.cuda.empty_cache()

        print("Mean RMSE:", np.mean(rmse_results))
        print("Std RMSE:", np.std(rmse_results))
        print("Mean R^2:", np.mean(r2_results))
        print("Std R^2:", np.std(r2_results))