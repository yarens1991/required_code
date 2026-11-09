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

def read_parquet(parquet_file_path):
    
    dataframe = pd.read_parquet(parquet_file_path, engine="pyarrow")
    
    return dataframe

def export_to_parquet(dataframe, output_file_path):
    """
    Exports a DataFrame to a Parquet file.
    """
    
    dataframe.to_parquet(output_file_path, index=False)
    
    print(f"Exported {len(dataframe)} rows to {output_file_path}")
    
def generate_synthetic_data(metadata, real_data, synthesizer):
    synthesizer.fit(real_data)

    synthetic_data = synthesizer.sample(num_rows=real_data.shape[0])
    synthetic_data = synthetic_data.sort_values(by=['remaining_cycles', 'cycle'], ascending=[False, True])

    print(f"Synthetic data head:\n{synthetic_data.head()}")

    quality_report = evaluate_quality(real_data, synthetic_data, metadata)

    quality_report.get_details('Column Shapes')
    quality_report.get_visualization('Column Shapes')
    
    return synthetic_data, quality_report

temp = read_parquet(os.path.join('..', 'Datasets', 'Parquet real', 'RISE', 'r4-d01.parquet'))

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


files = os.listdir(os.path.join('..', 'Datasets', 'Parquet real', 'RISE'))
#for file in files:
for file in files[11:]:
    real = read_parquet(os.path.join('..', 'Datasets', 'Parquet real', 'RISE', file))

    gaussian_synthesizer = GaussianCopulaSynthesizer(metadata)
    
    gaussian_synthetic_data, gaussian_quality_report = generate_synthetic_data(metadata, real, gaussian_synthesizer)
    
    export_to_parquet(gaussian_synthetic_data, os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'gaussian_synthetic_data_rise_{file[:6]}.parquet'))
    
    gaussian_quality_report.get_details('Column Shapes')
    gaussian_quality_report.get_visualization('Column Shapes')
    
    with open(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'scores_{file[:6]}.txt'), 'w') as f:
        f.write(f"{gaussian_quality_report.get_properties().values[0, 0]}: {gaussian_quality_report.get_properties().values[0, 1]}\n")
        f.write(f"{gaussian_quality_report.get_properties().values[1, 0]}: {gaussian_quality_report.get_properties().values[1, 1]}\n")
        f.write(f"Overall Score: {gaussian_quality_report.get_score()}\n")
    
    fig1 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='cleaned_diff_voltage(V)', metadata=metadata)
    fig2 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='cleaned_diff_resistance(mOhm)', metadata=metadata)
    fig3 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='mean_block_temperature(C)', metadata=metadata)
    fig1.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'Cleaned_diff_voltage_{file[:6]}.html'))
    fig2.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'Cleaned_diff_resistance_{file[:6]}.html'))
    fig3.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'Gaussian copula', f'Mean_block_temperature_{file[:6]}.html'))
    
    del gaussian_synthesizer, gaussian_synthetic_data, gaussian_quality_report
    
    
    
    
    
    if torch.cuda.is_available():
        print("CTGAN will use the GPU")
    else:
        print("CTGAN will use the CPU")
            
    ctgan_synthesizer = CTGANSynthesizer(metadata, epochs=1500, verbose=True)
    
    ctgan_synthetic_data, ctgan_quality_report = generate_synthetic_data(metadata, real, ctgan_synthesizer)
    
    export_to_parquet(ctgan_synthetic_data, os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'ctgan_synthetic_data_rise_{file[:6]}.parquet'))
    
    ctgan_quality_report.get_details('Column Shapes')
    ctgan_quality_report.get_visualization('Column Shapes')
    
    with open(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'scores_{file[:6]}.txt'), 'w') as f:
        f.write(f"{ctgan_quality_report.get_properties().values[0, 0]}: {ctgan_quality_report.get_properties().values[0, 1]}\n")
        f.write(f"{ctgan_quality_report.get_properties().values[1, 0]}: {ctgan_quality_report.get_properties().values[1, 1]}\n")
        f.write(f"Overall Score: {ctgan_quality_report.get_score()}\n")
    
    fig4 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='cleaned_diff_voltage(V)', metadata=metadata)
    fig5 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='cleaned_diff_resistance(mOhm)', metadata=metadata)
    fig6 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='mean_block_temperature(C)', metadata=metadata)
    fig4.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'Cleaned_diff_voltage_{file[:6]}.html'))
    fig5.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'Cleaned_diff_resistance_{file[:6]}.html'))
    fig6.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'RISE', 'CTGAN', f'Mean_block_temperature_{file[:6]}.html'))
    
    del ctgan_synthesizer, ctgan_synthetic_data, ctgan_quality_report