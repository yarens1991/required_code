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



temp = read_parquet(os.path.join('..', 'Datasets', 'Parquet real', 'NASA', 'Device2.parquet'))

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

for device in devices:
    real = read_parquet(os.path.join('..', 'Datasets', 'Parquet real', 'NASA', f'{device}.parquet'))
    
    gaussian_synthesizer = GaussianCopulaSynthesizer(metadata)
    
    gaussian_synthetic_data, gaussian_quality_report = generate_synthetic_data(metadata, real, gaussian_synthesizer)
    
    export_to_parquet(gaussian_synthetic_data, os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'gaussian_synthetic_data_nasa_{device}.parquet'))
    
    gaussian_quality_report.get_details('Column Shapes')
    gaussian_quality_report.get_visualization('Column Shapes')
    
    with open(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'scores_{device}.txt'), 'w') as f:
        f.write(f"{gaussian_quality_report.get_properties().values[0, 0]}: {gaussian_quality_report.get_properties().values[0, 1]}\n")
        f.write(f"{gaussian_quality_report.get_properties().values[1, 0]}: {gaussian_quality_report.get_properties().values[1, 1]}\n")
        f.write(f"Overall Score: {gaussian_quality_report.get_score()}\n")
    
    fig1 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='supplyVoltage', metadata=metadata)
    fig2 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='collectorEmitterCurrent', metadata=metadata)
    fig3 = get_column_plot(real_data=real, synthetic_data=gaussian_synthetic_data, column_name='packageTemperature', metadata=metadata)
    fig1.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'supplyVoltage_{device}.html'))
    fig2.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'collectorEmitterCurrent_{device}.html'))
    fig3.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'Gaussian copula', f'packageTemperature_{device}.html'))
    
    
    
    
    
    if torch.cuda.is_available():
        print("CTGAN will use the GPU")
    else:
        print("CTGAN will use the CPU")
            
    ctgan_synthesizer = CTGANSynthesizer(metadata, epochs=15000, verbose=True)
    
    ctgan_synthetic_data, ctgan_quality_report = generate_synthetic_data(metadata, real, ctgan_synthesizer)
    
    export_to_parquet(ctgan_synthetic_data, os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'ctgan_synthetic_data_nasa_{device}.parquet'))
    
    ctgan_quality_report.get_details('Column Shapes')
    ctgan_quality_report.get_visualization('Column Shapes')
    
    with open(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'scores_{device}.txt'), 'w') as f:
        f.write(f"{ctgan_quality_report.get_properties().values[0, 0]}: {ctgan_quality_report.get_properties().values[0, 1]}\n")
        f.write(f"{ctgan_quality_report.get_properties().values[1, 0]}: {ctgan_quality_report.get_properties().values[1, 1]}\n")
        f.write(f"Overall Score: {ctgan_quality_report.get_score()}\n")
    
    fig4 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='supplyVoltage', metadata=metadata)
    fig5 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='collectorEmitterCurrent', metadata=metadata)
    fig6 = get_column_plot(real_data=real, synthetic_data=ctgan_synthetic_data, column_name='packageTemperature', metadata=metadata)
    fig4.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'supply_voltage_{device}.html'))
    fig5.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'collectorEmitterCurrent_{device}.html'))
    fig6.write_html(os.path.join('..', 'Datasets', 'Parquet synthetic', 'NASA', 'CTGAN', f'packageTemperature_{device}.html'))