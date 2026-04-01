# wind_turbine_anomaly_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import glob
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

# Helper function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def wavelet_denoise(data, wavelet, noise_sigma):
    wavelet = pywt.Wavelet(wavelet)
    levels = min(5, int(np.floor(np.log2(data.shape[0]))))
    coeffs = pywt.wavedec(data, wavelet, level=levels)
    threshold = noise_sigma * np.sqrt(2 * np.log2(data.size))
    new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(new_coeffs, wavelet)

# Example data load (replace with your CSV)
# df = pd.read_csv("wind_turbine.csv")

# For this script, assume df is already loaded and includes necessary columns
features = [
    'temp', 'pressure', 'humidity', 'altitude',
    'voltage', 'power', 'rpm',
    'gearbox_vibration'
]

# Denoise and normalize
df_train = df.copy()
raw_std = df_train[features].std()
for f in features:
    df_train[f] = wavelet_denoise(df_train[f].values, 'db6', raw_std[f])
scaler = StandardScaler()
df_train = pd.DataFrame(scaler.fit_transform(df_train[features]), columns=features)

# Plot
plt.figure(figsize=(20,10))
df_train.plot(figsize=(20,10))
plt.title("Denoised & Normalized Data")
plt.tight_layout()
plt.savefig('denoised_normalized_data.png')
plt.close()

# Anomaly Detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(df_train)
anomaly_scores = iso_forest.decision_function(df_train)
threshold = np.percentile(anomaly_scores, 1)
anomalies = df.loc[anomaly_scores < threshold]

plt.figure(figsize=(15, 8))
plt.scatter(df.index, df['voltage'], c='b', label='Normal')
plt.scatter(anomalies.index, anomalies['voltage'], c='r', label='Anomaly')
plt.title('Anomaly Detection in Voltage')
plt.legend()
plt.savefig('anomaly_detection.png')
plt.close()

# Correlation Heatmap
corr = df_train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(15, 8))
sns.heatmap(corr, annot=True, mask=mask)
plt.title("Correlation Heatmap")
plt.savefig('correlation_heatmap.png')
plt.close()

# Create tensors
def create_dataset(X, time_steps=1, step=1):
    Xs = []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

INTERVAL = 5
TIME_STEPS = 20 * INTERVAL
STEP = 10
n_cols = len(df_train.columns)
X = create_dataset(df_train, TIME_STEPS, STEP)
X = np.nan_to_num(X, copy=True)

X = np.transpose(X, (0, 2, 1)).reshape(X.shape[0], n_cols, 10, 10)

# Save as .npy chunks
os.makedirs('data', exist_ok=True)
for i, x in enumerate(np.array_split(X, 60)):
    np.save(f'data/wind_turbine_{i:02d}.npy', x)