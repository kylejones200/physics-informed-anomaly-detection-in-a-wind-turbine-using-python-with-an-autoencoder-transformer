"""Core functions for wind turbine anomaly detection."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pywt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def wavelet_denoise(data: np.ndarray, wavelet: str = 'db6', noise_sigma: float = None) -> np.ndarray:
    """Apply wavelet denoising to data."""
    wavelet_obj = pywt.Wavelet(wavelet)
    levels = min(5, int(np.floor(np.log2(data.shape[0]))))
    coeffs = pywt.wavedec(data, wavelet_obj, level=levels)
    
    if noise_sigma is None:
        noise_sigma = np.std(data)
    
    threshold = noise_sigma * np.sqrt(2 * np.log2(data.size))
    new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(new_coeffs, wavelet_obj)

def preprocess_data(df: pd.DataFrame, features: List[str], 
                   wavelet: str = 'db6') -> Tuple[pd.DataFrame, StandardScaler]:
    """Denoise and normalize data."""
    df_train = df.copy()
    raw_std = df_train[features].std()
    
    for f in features:
        df_train[f] = wavelet_denoise(df_train[f].values, wavelet, raw_std[f])
    
    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train[features]),
        columns=features,
        index=df_train.index
    )
    
    return df_train_scaled, scaler

def detect_anomalies(df_train: pd.DataFrame, contamination: float = 0.01,
                    n_estimators: int = 100, random_state: int = 42) -> Tuple[np.ndarray, pd.DataFrame]:
    """Detect anomalies using Isolation Forest."""
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    iso_forest.fit(df_train)
    anomaly_scores = iso_forest.decision_function(df_train)
    threshold = np.percentile(anomaly_scores, 1)
    anomalies = df_train[anomaly_scores < threshold]
    return anomaly_scores, anomalies

def create_dataset(X: pd.DataFrame, time_steps: int = 100, step: int = 10) -> np.ndarray:
    """Create time series dataset with sliding windows."""
    Xs = []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

def reshape_for_tensor(X: np.ndarray, n_cols: int, target_shape: Tuple[int, int] = (10, 10)) -> np.ndarray:
    """Reshape data for tensor operations."""
    X = np.transpose(X, (0, 2, 1))
    return X.reshape(X.shape[0], n_cols, target_shape[0], target_shape[1])

def plot_denoised_data(df_train: pd.DataFrame, output_path: Path):
    """Plot denoised and normalized data """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in df_train.columns:
        ax.plot(df_train.index, df_train[col], label=col, linewidth=1.0, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Value")
    ax.legend(loc='best', ncol=2)
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_anomaly_detection(df: pd.DataFrame, anomalies: pd.DataFrame,
                          feature: str, output_path: Path):
 """Plot anomaly detection results """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    normal_mask = ~df.index.isin(anomalies.index)
    normal_data = df[normal_mask]
    
    ax.scatter(normal_data.index, normal_data[feature], 
              c='#4A90A4', label='Normal', s=20, alpha=0.6)
    ax.scatter(anomalies.index, anomalies[feature], 
              c='#D4A574', label='Anomaly', s=30, alpha=0.8)
    
    ax.set_xlabel("Time")
    ax.set_ylabel(feature)
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot correlation heatmap """
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

