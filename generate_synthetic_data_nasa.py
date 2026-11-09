# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:11:32 2025

@author: Gamhe
"""

import os
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality, get_column_plot
import random
import numpy as np
from scipy import stats

SEEDS = [42, 123, 456, 789, 1000]

def seed_everything(seed: int) -> None:
    """Make every stochastic library reproducible for this run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_parquet(parquet_file_path):
    
    dataframe = pd.read_parquet(parquet_file_path, engine="pyarrow")
    
    return dataframe

def export_to_parquet(dataframe, output_file_path):
    """
    Exports a DataFrame to a Parquet file.
    """
    
    dataframe.to_parquet(output_file_path, index=False)
    
    print(f"Exported {len(dataframe)} rows to {output_file_path}")
    
def generate_synthetic_data(metadata, real_data, synthesizer, seed=None):
    num_rows=real_data.shape[0]

    seed_everything(seed)

    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data = synthetic_data.sort_values(by=['remaining_cycles', 'cycle'], ascending=[False, True])

    print(f"Synthetic data head:\n{synthetic_data.head()}")

    quality_report = evaluate_quality(real_data, synthetic_data, metadata)

    quality_report.get_details('Column Shapes')
    quality_report.get_visualization('Column Shapes')
    
    return synthetic_data, quality_report


temp = read_parquet(os.path.join('Datasets', 'Parquet real', 'NASA', 'Device2.parquet'))

metadata = Metadata.detect_from_dataframe(data=temp)
print(metadata)

metadata.update_columns(
    column_names=['cycle', 'supplyVoltage', 'node1Voltage', 'node2Voltage', 'collectorEmitterCurrent', 'packageTemperature', 'remaining_cycles'],
    sdtype='numerical',
    computer_representation='Float'
)
metadata.remove_primary_key()

print(metadata)

print(f"Validation errors: {metadata.validate()}")

#metadata.save_to_json('nasa_metadata.json')

metadata.visualize()


devices = ['Device2', 'Device3', 'Device4', 'Device5']

gaussian_ksc = {d: [] for d in devices}
gaussian_vps = {d: [] for d in devices}
ctgan_ksc = {d: [] for d in devices}
ctgan_vps = {d: [] for d in devices}

for device in devices:
    real = read_parquet(os.path.join('Datasets', 'Parquet real', 'NASA', f'{device}.parquet'))
    
    gaussian_synthesizer = GaussianCopulaSynthesizer(metadata)
    gaussian_synthesizer.fit(real)

    if torch.cuda.is_available():
        print("CTGAN will use the GPU")
    else:
        print("CTGAN will use the CPU")
            
    ctgan_synthesizer = CTGANSynthesizer(metadata, epochs=15000, verbose=True)
    ctgan_synthesizer.fit(real)

    for seed in SEEDS:
    
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        
        gaussian_synthetic_data, gaussian_quality_report = generate_synthetic_data(metadata, real, gaussian_synthesizer, seed=seed)
        
        #export_to_parquet(gaussian_synthetic_data, os.path.join('Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'gaussian_synthetic_data_nasa_{device}.parquet'))
        
        gaussian_quality_report.get_details('Column Shapes')
        gaussian_quality_report.get_visualization('Column Shapes')
        
        with open(os.path.join('synthetic_data_generation_results', f'nasa_gaussian_scores_for_seed_{seed}_{device}.txt'), 'w') as f:
            f.write(f"{gaussian_quality_report.get_properties().values[0, 0]}: {gaussian_quality_report.get_properties().values[0, 1]}\n")
            f.write(f"{gaussian_quality_report.get_properties().values[1, 0]}: {gaussian_quality_report.get_properties().values[1, 1]}\n")
            f.write(f"Overall Score: {gaussian_quality_report.get_score()}\n")
        gaussian_ksc[device].append(gaussian_quality_report.get_properties().values[0, 1])
        gaussian_vps[device].append(gaussian_quality_report.get_properties().values[1, 1])
        
        # fig1 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='supplyVoltage', metadata=metadata)
        # fig2 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='collectorEmitterCurrent', metadata=metadata)
        # fig3 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='packageTemperature', metadata=metadata)
        # fig1.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'supplyVoltage_{device}.html'))
        # fig2.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'collectorEmitterCurrent_{device}.html'))
        # fig3.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'packageTemperature_{device}.html'))
        
        
        
        
        

        
        ctgan_synthetic_data, ctgan_quality_report = generate_synthetic_data(metadata, real, ctgan_synthesizer, seed=seed)
        
        #export_to_parquet(ctgan_synthetic_data, os.path.join('Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'ctgan_synthetic_data_nasa_{device}.parquet'))
        
        ctgan_quality_report.get_details('Column Shapes')
        ctgan_quality_report.get_visualization('Column Shapes')
        
        with open(os.path.join('synthetic_data_generation_results', f'nasa_ctgan_scores_for_seed_{seed}_{device}.txt'), 'w') as f:
            f.write(f"{ctgan_quality_report.get_properties().values[0, 0]}: {ctgan_quality_report.get_properties().values[0, 1]}\n")
            f.write(f"{ctgan_quality_report.get_properties().values[1, 0]}: {ctgan_quality_report.get_properties().values[1, 1]}\n")
            f.write(f"Overall Score: {ctgan_quality_report.get_score()}\n")
        ctgan_ksc[device].append(ctgan_quality_report.get_properties().values[0, 1])
        ctgan_vps[device].append(ctgan_quality_report.get_properties().values[1, 1])
        
        # fig4 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='supplyVoltage', metadata=metadata)
        # fig5 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='collectorEmitterCurrent', metadata=metadata)
        # fig6 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='packageTemperature', metadata=metadata)
        # fig4.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'supply_voltage_{device}.html'))
        # fig5.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'collectorEmitterCurrent_{device}.html'))
        # fig6.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'packageTemperature_{device}.html'))

# Statistical significance: Wilcoxon signed-rank test for KSC and VPS, per device and pooled
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

all_g_ksc, all_c_ksc, all_g_vps, all_c_vps = [], [], [], []

with open(os.path.join('synthetic_data_generation_results', 'nasa_statistical_significance.txt'), 'w') as f:
    f.write("Wilcoxon signed-rank test: Gaussian Copula vs CTGAN\n")
    f.write(f"n = {len(SEEDS)} replicas per device\n\n")

    for device in devices:
        gk, ck = gaussian_ksc[device], ctgan_ksc[device]
        gv, cv = gaussian_vps[device], ctgan_vps[device]
        all_g_ksc.extend(gk); all_c_ksc.extend(ck)
        all_g_vps.extend(gv); all_c_vps.extend(cv)
        sk, pk = _wilcoxon(gk, ck)
        sv, pv = _wilcoxon(gv, cv)
        dk, rk = _effect_sizes(gk, ck, sk)
        dv, rv = _effect_sizes(gv, cv, sv)
        f.write(f"Device {device}:\n")
        f.write(f"  KSC  Gaussian: {np.mean(gk):.8f} +/- {np.std(gk):.8f}  CTGAN: {np.mean(ck):.8f} +/- {np.std(ck):.8f}  W={sk:.8f}  p={pk:.8f}\n")
        f.write(f"       Effect size: Cohen's d = {dk:.8f}, rank-biserial r = {rk:.8f}\n")
        f.write(f"  VPS  Gaussian: {np.mean(gv):.8f} +/- {np.std(gv):.8f}  CTGAN: {np.mean(cv):.8f} +/- {np.std(cv):.8f}  W={sv:.8f}  p={pv:.8f}\n")
        f.write(f"       Effect size: Cohen's d = {dv:.8f}, rank-biserial r = {rv:.8f}\n\n")

    sk_pool, pk_pool = _wilcoxon(all_g_ksc, all_c_ksc)
    sv_pool, pv_pool = _wilcoxon(all_g_vps, all_c_vps)
    dk_pool, rk_pool = _effect_sizes(all_g_ksc, all_c_ksc, sk_pool)
    dv_pool, rv_pool = _effect_sizes(all_g_vps, all_c_vps, sv_pool)
    f.write(f"Pooled across all devices (n={len(all_g_ksc)} pairs):\n")
    f.write(f"  KSC  Gaussian: {np.mean(all_g_ksc):.8f} +/- {np.std(all_g_ksc):.8f}  CTGAN: {np.mean(all_c_ksc):.8f} +/- {np.std(all_c_ksc):.8f}  W={sk_pool:.8f}  p={pk_pool:.8f}\n")
    f.write(f"       Effect size: Cohen's d = {dk_pool:.8f}, rank-biserial r = {rk_pool:.8f}\n")
    f.write(f"  VPS  Gaussian: {np.mean(all_g_vps):.8f} +/- {np.std(all_g_vps):.8f}  CTGAN: {np.mean(all_c_vps):.8f} +/- {np.std(all_c_vps):.8f}  W={sv_pool:.8f}  p={pv_pool:.8f}\n")
    f.write(f"       Effect size: Cohen's d = {dv_pool:.8f}, rank-biserial r = {rv_pool:.8f}\n")

print("Statistical significance results saved to nasa_statistical_significance.txt")