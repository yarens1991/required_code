# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:52:45 2025

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

SEEDS = [42]

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


temp = read_parquet(os.path.join('Datasets', 'Parquet real', 'RISE', 'r4-d01.parquet'))

metadata = Metadata.detect_from_dataframe(data=temp)
print(metadata)

metadata.update_columns(
    column_names=['cycle', 'remaining_cycles', 'end_voltage(V)', 'end_resistance(mOhm)', 'min_leg_temperature(C)', 'max_leg_temperature(C)', 'resistance_from_max_leg_temperature(mOhm)', 'diff_resistance(mOhm)', 'cleaned_diff_resistance(mOhm)', 'min_temperature(C)', 'max_temperature(C)', 'min_block_temperature1(C)', 'max_block_temperature1(C)', 'min_block_temperature2(C)', 'max_block_temperature2(C)', 'min_water_inlet_temperature(C)', 'max_water_inlet_temperature(C)', 'min_water_outlet_temperature(C)', 'max_water_outlet_temperature(C)', 'mean_block_temperature(C)', 'voltage_from_mean_block_temperature(V)', 'diff_voltage(V)', 'cleaned_diff_voltage(V)'],
    sdtype='numerical',
    computer_representation='Float'
)
metadata.remove_primary_key()

print(metadata)

print(f"Validation errors: {metadata.validate()}")

#metadata.save_to_json('rise_metadata.json')

metadata.visualize()


    
files = os.listdir(os.path.join('Datasets', 'Parquet real', 'RISE'))

gaussian_ksc = {}
gaussian_vps = {}
ctgan_ksc = {}
ctgan_vps = {}

for file in files[:5] + files[15:]:
    real = read_parquet(os.path.join('Datasets', 'Parquet real', 'RISE', file))
    real = real.sample(frac=0.25, random_state=42)

    gaussian_synthesizer = GaussianCopulaSynthesizer(metadata)
    gaussian_synthesizer.fit(real)

    if torch.cuda.is_available():
        print("CTGAN will use the GPU")
    else:
        print("CTGAN will use the CPU")
            
    ctgan_synthesizer = CTGANSynthesizer(metadata, epochs=1500, verbose=True)
    ctgan_synthesizer.fit(real)
    
    for seed in SEEDS:
    
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        
        gaussian_synthetic_data, gaussian_quality_report = generate_synthetic_data(metadata, real, gaussian_synthesizer, seed=seed)
        
        #export_to_parquet(gaussian_synthetic_data, os.path.join('Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'gaussian_synthetic_data_rise_{file[:6]}.parquet'))
        
        gaussian_quality_report.get_details('Column Shapes')
        gaussian_quality_report.get_visualization('Column Shapes')
        
        with open(os.path.join('synthetic_data_generation_results', f'rise_gaussian_scores_for_seed_{seed}_{file[:6]}.txt'), 'w') as f:
            f.write(f"{gaussian_quality_report.get_properties().values[0, 0]}: {gaussian_quality_report.get_properties().values[0, 1]}\n")
            f.write(f"{gaussian_quality_report.get_properties().values[1, 0]}: {gaussian_quality_report.get_properties().values[1, 1]}\n")
            f.write(f"Overall Score: {gaussian_quality_report.get_score()}\n")
        gaussian_ksc.setdefault(file[:6], []).append(gaussian_quality_report.get_properties().values[0, 1])
        gaussian_vps.setdefault(file[:6], []).append(gaussian_quality_report.get_properties().values[1, 1])

        # fig1 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='cleaned_diff_voltage(V)', metadata=metadata)
        # fig2 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='cleaned_diff_resistance(mOhm)', metadata=metadata)
        # fig3 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='mean_block_temperature(C)', metadata=metadata)
        # fig1.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'Cleaned_diff_voltage_{file[:6]}.html'))
        # fig2.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'Cleaned_diff_resistance_{file[:6]}.html'))
        # fig3.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'Mean_block_temperature_{file[:6]}.html'))

        del gaussian_synthetic_data, gaussian_quality_report
        
        
        
        
        
        
        ctgan_synthetic_data, ctgan_quality_report = generate_synthetic_data(metadata, real, ctgan_synthesizer, seed=seed)
        
        #export_to_parquet(ctgan_synthetic_data, os.path.join('Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'ctgan_synthetic_data_rise_{file[:6]}.parquet'))
        
        ctgan_quality_report.get_details('Column Shapes')
        ctgan_quality_report.get_visualization('Column Shapes')
        
        with open(os.path.join('synthetic_data_generation_results', f'rise_ctgan_scores_for_seed_{seed}_{file[:6]}.txt'), 'w') as f:
            f.write(f"{ctgan_quality_report.get_properties().values[0, 0]}: {ctgan_quality_report.get_properties().values[0, 1]}\n")
            f.write(f"{ctgan_quality_report.get_properties().values[1, 0]}: {ctgan_quality_report.get_properties().values[1, 1]}\n")
            f.write(f"Overall Score: {ctgan_quality_report.get_score()}\n")
        ctgan_ksc.setdefault(file[:6], []).append(ctgan_quality_report.get_properties().values[0, 1])
        ctgan_vps.setdefault(file[:6], []).append(ctgan_quality_report.get_properties().values[1, 1])

        # fig4 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='cleaned_diff_voltage(V)', metadata=metadata)
        # fig5 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='cleaned_diff_resistance(mOhm)', metadata=metadata)
        # fig6 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='mean_block_temperature(C)', metadata=metadata)
        # fig4.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'Cleaned_diff_voltage_{file[:6]}.html'))
        # fig5.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'Cleaned_diff_resistance_{file[:6]}.html'))
        # fig6.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'Mean_block_temperature_{file[:6]}.html'))

        del ctgan_synthetic_data, ctgan_quality_report

    del gaussian_synthesizer, ctgan_synthesizer

# Statistical significance: Wilcoxon signed-rank test for KSC and VPS, pooled across all files
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
for key in gaussian_ksc:
    all_g_ksc.extend(gaussian_ksc[key])
    all_c_ksc.extend(ctgan_ksc.get(key, []))
    all_g_vps.extend(gaussian_vps[key])
    all_c_vps.extend(ctgan_vps.get(key, []))

sk_pool, pk_pool = _wilcoxon(all_g_ksc, all_c_ksc)
sv_pool, pv_pool = _wilcoxon(all_g_vps, all_c_vps)
dk_pool, rk_pool = _effect_sizes(all_g_ksc, all_c_ksc, sk_pool)
dv_pool, rv_pool = _effect_sizes(all_g_vps, all_c_vps, sv_pool)

with open(os.path.join('synthetic_data_generation_results', 'rise_statistical_significance.txt'), 'w') as f:
    f.write("Wilcoxon signed-rank test: Gaussian Copula vs CTGAN\n")
    f.write(f"n = {len(SEEDS)} replicas per file, {len(gaussian_ksc)} files processed\n\n")
    for key in gaussian_ksc:
        gk, ck = gaussian_ksc[key], ctgan_ksc.get(key, [])
        gv, cv = gaussian_vps[key], ctgan_vps.get(key, [])
        f.write(f"File {key}:  KSC Gaussian {gk[0]:.8f}  CTGAN {ck[0]:.8f}  |  "
                f"VPS Gaussian {gv[0]:.8f}  CTGAN {cv[0]:.8f}\n")
    f.write(f"\nPooled across all files (n={len(all_g_ksc)} pairs):\n")
    f.write(f"  KSC  Gaussian: {np.mean(all_g_ksc):.8f} +/- {np.std(all_g_ksc):.8f}  "
            f"CTGAN: {np.mean(all_c_ksc):.8f} +/- {np.std(all_c_ksc):.8f}  W={sk_pool:.8f}  p={pk_pool:.8f}\n")
    f.write(f"       Effect size: Cohen's d = {dk_pool:.8f}, rank-biserial r = {rk_pool:.8f}\n")
    f.write(f"  VPS  Gaussian: {np.mean(all_g_vps):.8f} +/- {np.std(all_g_vps):.8f}  "
            f"CTGAN: {np.mean(all_c_vps):.8f} +/- {np.std(all_c_vps):.8f}  W={sv_pool:.8f}  p={pv_pool:.8f}\n")
    f.write(f"       Effect size: Cohen's d = {dv_pool:.8f}, rank-biserial r = {rv_pool:.8f}\n")

print("Statistical significance results saved to rise_statistical_significance.txt")