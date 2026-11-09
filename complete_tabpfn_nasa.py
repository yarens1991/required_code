# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 13:20:35 2025

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


def compute_crps(y_true, quantile_preds, quantile_levels):
    """Approximate CRPS via mean pinball loss across quantile levels."""
    y = np.asarray(y_true)
    losses = []
    for i, alpha in enumerate(quantile_levels):
        q = np.asarray(quantile_preds[i])
        losses.append(np.where(y >= q, alpha * (y - q), (1 - alpha) * (q - y)).mean())
    return 2.0 * float(np.mean(losses))


def plot_calibration_curve(y_true, quantile_preds, quantile_levels, save_path):
    """Reliability diagram: nominal quantile level vs empirical fraction below."""
    y = np.asarray(y_true)
    empirical = [(y < np.asarray(quantile_preds[i])).mean() for i in range(len(quantile_levels))]
    plt.figure(figsize=(5, 5))
    plt.plot(quantile_levels, empirical, 'o-', color='blue', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Nominal quantile level')
    plt.ylabel('Empirical fraction below quantile')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

SEEDS = [42, 123, 456, 789, 1000]

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
results = {}  # (training_set, device, fraction) -> {'rmse': [], 'crps': []}
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
        n_max_map = {
            'Real': min(X_train_real.shape[0], 20000),
            'Gaussian copula': min(X_train_gaussian.shape[0], 20000),
            'CTGAN': min(X_train_ctgan.shape[0], 20000),
            'Real + Gaussian copula': min(X_train_real_plus_gaussian.shape[0], 20000),
            'Real + CTGAN': min(X_train_real_plus_ctgan.shape[0], 20000),
        }







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

            fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

            for fraction in fractions:
            
                print(f"Fitting with {fraction * 100}% of the available {training_set} data for {device}")
                
                if training_set == 'Real':
                    X_temp = X_train_real.sample(n=int(fraction * n_max_map['Real']), random_state=42)
                    y_temp = y_train_real_scaled.loc[X_temp.index]
                elif training_set == 'Gaussian copula':
                    X_temp = X_train_gaussian.sample(n=int(fraction * n_max_map['Gaussian copula']), random_state=42)
                    y_temp = y_train_gaussian_scaled.loc[X_temp.index]
                elif training_set == 'CTGAN':
                    X_temp = X_train_ctgan.sample(n=int(fraction * n_max_map['CTGAN']), random_state=42)
                    y_temp = y_train_ctgan_scaled.loc[X_temp.index]
                elif training_set == 'Real + Gaussian copula':
                    X_temp = X_train_real_plus_gaussian.sample(n=int(fraction * n_max_map['Real + Gaussian copula']), random_state=42)
                    y_temp = y_train_real_plus_gaussian_scaled.loc[X_temp.index]
                elif training_set == 'Real + CTGAN':
                    X_temp = X_train_real_plus_ctgan.sample(n=int(fraction * n_max_map['Real + CTGAN']), random_state=42)
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
                
                print(f"Metrics with {fraction * 100}% of the available {training_set} data for {device}:")
                y_test_scaled = _y_test_map[training_set]
                if training_set == 'Real':
                    rmse = root_mean_squared_error(y_test_real_scaled, predictions)
                    r2 = r2_score(y_test_real_scaled, predictions)
                    empirical_coverage = ((y_test_real_scaled >= quantile_predictions[0]) & (y_test_real_scaled <= quantile_predictions[-1])).mean()
                elif training_set == 'Gaussian copula':
                    rmse = root_mean_squared_error(y_test_gaussian_scaled, predictions)
                    r2 = r2_score(y_test_gaussian_scaled, predictions)
                    empirical_coverage = ((y_test_gaussian_scaled >= quantile_predictions[0]) & (y_test_gaussian_scaled <= quantile_predictions[-1])).mean()
                elif training_set == 'CTGAN':
                    rmse = root_mean_squared_error(y_test_ctgan_scaled, predictions)
                    r2 = r2_score(y_test_ctgan_scaled, predictions)
                    empirical_coverage = ((y_test_ctgan_scaled >= quantile_predictions[0]) & (y_test_ctgan_scaled <= quantile_predictions[-1])).mean()
                elif training_set == 'Real + Gaussian copula':
                    rmse = root_mean_squared_error(y_test_real_plus_gaussian_scaled, predictions)
                    r2 = r2_score(y_test_real_plus_gaussian_scaled, predictions)
                    empirical_coverage = ((y_test_real_plus_gaussian_scaled >= quantile_predictions[0]) & (y_test_real_plus_gaussian_scaled <= quantile_predictions[-1])).mean()
                elif training_set == 'Real + CTGAN':
                    rmse = root_mean_squared_error(y_test_real_plus_ctgan_scaled, predictions)
                    r2 = r2_score(y_test_real_plus_ctgan_scaled, predictions)
                    empirical_coverage = ((y_test_real_plus_ctgan_scaled >= quantile_predictions[0]) & (y_test_real_plus_ctgan_scaled <= quantile_predictions[-1])).mean()
                crps = compute_crps(y_test_scaled, quantile_predictions, quantiles)
                key = (training_set, device, fraction)
                if key not in results:
                    results[key] = {'rmse': [], 'r2': [], 'empirical_coverage': [], 'crps': []}
                results[key]['rmse'].append(rmse)
                results[key]['r2'].append(r2)
                results[key]['empirical_coverage'].append(empirical_coverage)
                results[key]['crps'].append(crps)

                print("RMSE:", rmse)
                print("R^2:", r2)
                print("Empirical coverage:", empirical_coverage)
                print("CRPS:", crps)
                
                plt.figure()
                plt.fill_between(range(len(predictions)), quantile_predictions[0], quantile_predictions[-1], color='blue', alpha=0.5, label='95% Confidence Interval')
                plt.plot(predictions, color='red', label='Prediction')
                plt.plot(y_test_scaled, color='black', label='Ground truth')
                plt.legend()
                plt.xlabel('Cycle')
                plt.ylabel('RUL (remaining cycles normalized)')
                #plt.title( f'Using {fraction * 100}% of the available {training_set} data as context', fontsize=12)
                plt.savefig(os.path.join('modeling_results', 'NASA', f'Results for {device} using {fraction * 100}% of the available {training_set} data as context and seed {seed} RMSE = {rmse:.4f} R2 = {r2:.4f} Empirical coverage = {empirical_coverage:.4f}.png'), dpi=300, bbox_inches='tight')
                plt.show()
                ts_safe = training_set.replace(' + ', '_').replace(' ', '_')
                plot_calibration_curve(y_test_scaled, quantile_predictions, quantiles,
                    os.path.join('modeling_results', 'NASA', f'Calibration_{device}_{ts_safe}_{fraction}_seed{seed}.png'))
                
                del tabpfn
                gc.collect()
                torch.cuda.empty_cache()

with open(os.path.join('modeling_results', 'NASA', 'timing_report.txt'), 'w') as f:
    f.write("TabPFN timing report: NASA dataset (mean +/- std across seeds and devices, leave-one-out)\n\n")
    ts_list = ['Real', 'Gaussian copula', 'CTGAN', 'Real + Gaussian copula', 'Real + CTGAN']
    fr_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    for fraction in fr_list:
        f.write(f"Fraction {fraction:.2f}:\n")
        for ts in ts_list:
            t_key = (ts, fraction)
            if t_key in timing and timing[t_key]['fit']:
                n_mean = int(np.mean(timing[t_key]['n_train']))
                ft = timing[t_key]['fit']
                pt = timing[t_key]['predict']
                f.write(f"  {ts} ({fraction*100:.0f}%, ~{n_mean} samples): "
                        f"fit={np.mean(ft):.4f}+/-{np.std(ft):.4f} s, "
                        f"predict={np.mean(pt):.4f}+/-{np.std(pt):.4f} s\n")
        f.write("\n")
print("Timing report saved to modeling_results/NASA/timing_report.txt")

def _wilcoxon(a, b):
    try:
        s, p = stats.wilcoxon(a, b, alternative='two-sided')
        return float(s), float(p)
    except ValueError:
        return float('nan'), float('nan')

def _effect_sizes(a, b, stat):
    """Cohen's d (paired) and rank-biserial r for Wilcoxon signed-rank test."""
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    n = len(diff)
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) != 0 else float('nan')
    r = float(1.0 - (4.0 * stat) / (n * (n + 1))) if not np.isnan(stat) else float('nan')
    return d, r

# Statistical significance: metrics per fraction, pooled Wilcoxon across all fractions
training_sets = ['Real', 'Gaussian copula', 'CTGAN', 'Real + Gaussian copula', 'Real + CTGAN']
fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
with open(os.path.join('modeling_results', 'NASA', 'statistical_significance.txt'), 'w') as f:
    f.write("Statistical significance: training set comparisons (Wilcoxon signed-rank)\n")
    f.write(f"Metrics: {len(SEEDS)} seeds x {len(devices)} devices per fraction\n\n")
    for fraction in fractions:
        f.write(f"Fraction {fraction:.2f}:\n")
        for ts in training_sets:
            all_rmse = [v for dev in devices for v in results.get((ts, dev, fraction), {}).get('rmse', [])]
            all_r2 = [v for dev in devices for v in results.get((ts, dev, fraction), {}).get('r2', [])]
            all_cov = [v for dev in devices for v in results.get((ts, dev, fraction), {}).get('empirical_coverage', [])]
            all_crps = [v for dev in devices for v in results.get((ts, dev, fraction), {}).get('crps', [])]
            _n = int(np.mean(timing.get((ts, fraction), {}).get('n_train', [0])))
            f.write(f"  {ts} ({fraction*100:.0f}%, ~{_n} samples): RMSE {np.mean(all_rmse):.8f}+/-{np.std(all_rmse):.8f}, "
                    f"R2 {np.mean(all_r2):.8f}+/-{np.std(all_r2):.8f}, "
                    f"Coverage {np.mean(all_cov):.8f}+/-{np.std(all_cov):.8f}, "
                    f"CRPS {np.mean(all_crps):.8f}+/-{np.std(all_crps):.8f}\n")
        f.write("\n")
    n_pool = len(SEEDS) * len(devices) * len(fractions)
    f.write(f"Pooled Wilcoxon tests vs Real (n={n_pool} pairs = {len(SEEDS)} seeds x {len(devices)} devices x {len(fractions)} fractions):\n")
    real_rmse_pool = [v for fraction in fractions for dev in devices for v in results.get(('Real', dev, fraction), {}).get('rmse', [])]
    real_r2_pool = [v for fraction in fractions for dev in devices for v in results.get(('Real', dev, fraction), {}).get('r2', [])]
    real_cov_pool = [v for fraction in fractions for dev in devices for v in results.get(('Real', dev, fraction), {}).get('empirical_coverage', [])]
    real_crps_pool = [v for fraction in fractions for dev in devices for v in results.get(('Real', dev, fraction), {}).get('crps', [])]
    for ts_b in ['Gaussian copula', 'CTGAN', 'Real + Gaussian copula', 'Real + CTGAN']:
        b_rmse_pool = [v for fraction in fractions for dev in devices for v in results.get((ts_b, dev, fraction), {}).get('rmse', [])]
        b_r2_pool = [v for fraction in fractions for dev in devices for v in results.get((ts_b, dev, fraction), {}).get('r2', [])]
        b_cov_pool = [v for fraction in fractions for dev in devices for v in results.get((ts_b, dev, fraction), {}).get('empirical_coverage', [])]
        b_crps_pool = [v for fraction in fractions for dev in devices for v in results.get((ts_b, dev, fraction), {}).get('crps', [])]
        if len(real_rmse_pool) >= 2 and len(real_rmse_pool) == len(b_rmse_pool):
            s_rmse, p_rmse = _wilcoxon(real_rmse_pool, b_rmse_pool)
            d_rmse, r_rmse = _effect_sizes(real_rmse_pool, b_rmse_pool, s_rmse)
            s_r2, p_r2 = _wilcoxon(real_r2_pool, b_r2_pool)
            d_r2, r_r2 = _effect_sizes(real_r2_pool, b_r2_pool, s_r2)
            s_cov, p_cov = _wilcoxon(real_cov_pool, b_cov_pool)
            d_cov, r_cov = _effect_sizes(real_cov_pool, b_cov_pool, s_cov)
            s_crps, p_crps = _wilcoxon(real_crps_pool, b_crps_pool)
            d_crps, r_crps = _effect_sizes(real_crps_pool, b_crps_pool, s_crps)
            f.write(f"  Real vs {ts_b}:\n")
            f.write(f"    RMSE:     W={s_rmse:.8f}  p={p_rmse:.8f}  Cohen's d={d_rmse:.8f}, rank-biserial r={r_rmse:.8f}\n")
            f.write(f"    R2:       W={s_r2:.8f}  p={p_r2:.8f}  Cohen's d={d_r2:.8f}, rank-biserial r={r_r2:.8f}\n")
            f.write(f"    Coverage: W={s_cov:.8f}  p={p_cov:.8f}  Cohen's d={d_cov:.8f}, rank-biserial r={r_cov:.8f}\n")
            f.write(f"    CRPS:     W={s_crps:.8f}  p={p_crps:.8f}  Cohen's d={d_crps:.8f}, rank-biserial r={r_crps:.8f}\n")
print("Statistical significance saved to modeling_results/NASA/statistical_significance.txt")