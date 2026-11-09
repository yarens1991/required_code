# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:54:26 2026

@author: Gamhe
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
import torch.nn as nn
import gc

TIMESTEPS = 20
TRAINING_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
BATCH_SIZE = 15000

def load_parquet_data(path, output_column='remaining_cycles'):
    data = pd.read_parquet(path)

    X = data.drop(columns=[output_column])
    y = data[output_column]

    return X, y

def create_lstm_sequences(X, y, timesteps=TIMESTEPS):
    # Window i holds rows [i, i + timesteps) and its target is the y of the last row in the window
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(X) < timesteps:
        return np.empty((0, timesteps, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X_seq = np.lib.stride_tricks.sliding_window_view(X, timesteps, axis=0).transpose(0, 2, 1).copy()
    return X_seq, y[timesteps - 1:]

class LSTMModel(nn.Module):
    # LSTM(128) -> Dense(128, relu) -> Dense(1, linear)
    def __init__(self, num_features):
        super().__init__()
        self.lstm = nn.LSTM(num_features, 128, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        h = self.dense(self.lstm(x)[0][:, -1, :])
        return self.output(h).squeeze(1)

def mse_loss(predictions, y):
    return nn.functional.mse_loss(predictions, y)

def fit_lstm(X, y, validation_split=0.05, seed=42):
    torch.manual_seed(seed)
    model = LSTMModel(X.shape[2]).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_val = int(len(X) * validation_split)
    X_tr = torch.tensor(X[:-n_val], device=torch_device)
    y_tr = torch.tensor(y[:-n_val], device=torch_device)
    X_val = torch.tensor(X[-n_val:], device=torch_device)
    y_val = torch.tensor(y[-n_val:], device=torch_device)

    best_val_loss, best_state, epochs_without_improvement = np.inf, None, 0
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        for batch in torch.randperm(len(X_tr), device=torch_device).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = mse_loss(model(X_tr[batch]), y_tr[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(mse_loss(model(xb), yb).item() * len(xb) for xb, yb in zip(X_val.split(4096), y_val.split(4096))) / len(X_val)

        # Early stopping with restore_best_weights=True
        if val_loss < best_val_loss:
            best_val_loss, best_state, epochs_without_improvement = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model

def predict_lstm(model, X):
    model.eval()
    predictions = []
    with torch.no_grad():
        for xb in torch.tensor(X, device=torch_device).split(4096):
            predictions.append(model(xb).cpu().numpy())
    return np.concatenate(predictions)



if torch.cuda.is_available():
    print("LSTM will use the GPU")
    torch_device = torch.device('cuda')
else:
    print("LSTM will use the CPU")
    torch_device = torch.device('cpu')

files = os.listdir(os.path.join('Datasets', 'Parquet real', 'RISE'))
test_devices = [x[0:6] for x in files if x.startswith("r5")]
remaining_devices = [x[0:6] for x in files if not x.startswith("r5")]




X_train_real = []
y_train_real = []

for temp_device in remaining_devices:
    X_temp, y_temp = load_parquet_data(os.path.join('Datasets', 'Parquet real', 'RISE', f'{temp_device}.parquet'))
    X_train_real.append(X_temp)
    y_train_real.append(y_temp)



# The LSTM does not accept NaN: fill missing values with the training medians
X_train_medians = pd.concat(X_train_real, ignore_index=True).median()

y_real_scaler = MinMaxScaler()
y_real_scaler.fit(pd.concat(y_train_real, ignore_index=True).values.reshape(-1, 1))

# Sequences are built per device so that no window crosses from one device into another
X_train_real_seq = []
y_train_real_seq = []

for X_temp, y_temp in zip(X_train_real, y_train_real):
    X_seq, y_seq = create_lstm_sequences(X_temp.fillna(X_train_medians), y_real_scaler.transform(y_temp.values.reshape(-1, 1)).flatten())
    X_train_real_seq.append(X_seq)
    y_train_real_seq.append(y_seq)

X_train_real_seq = np.concatenate(X_train_real_seq)
y_train_real_seq = np.concatenate(y_train_real_seq)



training_sets = ['Real']

for training_set in training_sets:

    fractions = [1.0]
    quantiles = [0.025, 0.975]

    for fraction in fractions:

        print(f"Fitting with {fraction * 100}% of the available {training_set} data")

        if training_set == 'Real':
            # Sample whole sequences (not rows) so the windows stay intact
            sample_idx = np.random.default_rng(42).choice(len(X_train_real_seq), size=int(round(fraction * len(X_train_real_seq))), replace=False)
            X_temp = X_train_real_seq[sample_idx]
            y_temp = y_train_real_seq[sample_idx]


        lstm_model = fit_lstm(X_temp, y_temp)

        rmse_results = []
        r2_results = []

        for device in test_devices:
            X_test, y_test = load_parquet_data(os.path.join('Datasets', 'Parquet real', 'RISE', f'{device}.parquet'))

            if training_set == 'Real':
                X_test_seq, y_test_scaled = create_lstm_sequences(X_test[X_train_medians.index].fillna(X_train_medians), y_real_scaler.transform(y_test.values.reshape(-1, 1)).flatten())


            prediction_batches = []

            for start in range(0, len(X_test_seq), 10000):
                temp_pred = predict_lstm(lstm_model, X_test_seq[start:start + 10000])
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

        del lstm_model
        gc.collect()
        torch.cuda.empty_cache()

        print("Mean RMSE:", np.mean(rmse_results))
        print("Std RMSE:", np.std(rmse_results))
        print("Mean R^2:", np.mean(r2_results))
        print("Std R^2:", np.std(r2_results))