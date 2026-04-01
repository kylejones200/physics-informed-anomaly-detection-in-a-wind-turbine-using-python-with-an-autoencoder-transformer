#!/usr/bin/env python3
"""
Physics-Informed Anomaly Detection in Wind Turbine

Main entry point for running anomaly detection analysis.
"""

import argparse
import yaml
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    preprocess_data,
    detect_anomalies,
    create_dataset,
    reshape_for_tensor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Wind Turbine Anomaly Detection')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, required=True, help='Path to wind turbine data CSV')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(config['output']['data_dir'])
    data_dir.mkdir(exist_ok=True)
    
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    df = pd.read_csv(args.data_path)
    features = config['data']['features']
    
    if config['analysis']['run_preprocessing']:
                df_train, scaler = preprocess_data(df, features, config['preprocessing']['wavelet'])
        
        if config['analysis']['run_anomaly_detection']:
                        anomaly_scores, anomalies = detect_anomalies(
                df_train,
                contamination=config['preprocessing']['contamination'],
                n_estimators=config['preprocessing']['n_estimators'],
                random_state=config['preprocessing']['random_state']
            )
            logging.info(f"Found {len(anomalies)} anomalies")
            
            plot_anomaly_detection(df_train, anomalies, 'voltage',
                                  output_dir / 'anomaly_detection.png')
        
        if config['analysis']['run_correlation']:
                        plot_correlation_heatmap(df_train, output_dir / 'correlation_heatmap.png')
        
        plot_denoised_data(df_train, output_dir / 'denoised_normalized_data.png')
        
        if config['analysis']['create_tensors']:
                        time_steps = config['tensor']['time_steps_multiplier'] * config['tensor']['interval']
            X = create_dataset(df_train, time_steps, config['tensor']['step'])
            X = np.nan_to_num(X, copy=True)
            
            n_cols = len(df_train.columns)
            X_reshaped = reshape_for_tensor(
                X, 
                n_cols, 
                tuple(config['tensor']['target_shape'])
            )
            
            for i, x in enumerate(np.array_split(X_reshaped, config['tensor']['n_chunks'])):
                np.save(data_dir / f'wind_turbine_{i:02d}.npy', x)
            logging.info(f"Saved {config['tensor']['n_chunks']} tensor chunks to {data_dir}")
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

