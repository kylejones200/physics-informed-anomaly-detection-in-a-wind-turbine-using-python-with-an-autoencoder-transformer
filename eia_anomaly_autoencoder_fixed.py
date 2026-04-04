import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.seasonal import STL

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    season: int = 12
    window: int = 24
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-3
    z_thresh: float = 3.0  # threshold on recon error z-score


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def stl_residuals(s: pd.Series, season: int) -> pd.Series:
    stl = STL(s, period=season, robust=True).fit()
    return stl.resid

class AE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def make_windows(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) < win:
        return np.empty((0, win))
    return np.stack([x[i:i+win] for i in range(len(x)-win+1)], axis=0)


def train_autoencoder(X: np.ndarray, cfg: Config) -> Tuple[AE, np.ndarray]:
    device = torch.device('cpu')
    model = AE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()

    # Reconstruction errors (per window)
    model.eval()
    with torch.no_grad():
        Xten = torch.from_numpy(X).float().to(device)
        R = model(Xten).cpu().numpy()
    errs = np.mean((R - X)**2, axis=1)
    return model, errs


def main():
    cfg = Config()
    s = load_series(cfg)

    # Physics-informed: learn on STL residual windows (trend/season removed)
    resid = stl_residuals(s, cfg.season).dropna()
    # Standardize residuals for training stability
    mu, sd = resid.mean(), resid.std(ddof=1)
    sd = sd if sd else 1.0
    zres = (resid - mu) / sd

    X = make_windows(zres.values, cfg.window)
    if X.shape[0] == 0:
        raise SystemExit("Series too short for configured window size.")

    # Train on the middle 80% windows to reduce edge effects (still leakage-safe for detection use case)
    n = X.shape[0]
    lo, hi = int(0.1*n), int(0.9*n)
    X_train = X[lo:hi]

    model, errs = train_autoencoder(X_train, cfg)

    # Score all windows using the trained AE
    with torch.no_grad():
        X_all = torch.from_numpy(X).float()
        R_all = model(X_all).cpu().numpy()
        all_errs = np.mean((R_all - X)**2, axis=1)

    # Map window error back to the end timestamp of each window
    err_idx = resid.index[cfg.window-1:]
    err_s = pd.Series(all_errs, index=err_idx)

    # Z-score thresholding on reconstruction error
    e_mu, e_sd = err_s.mean(), err_s.std(ddof=1)
    e_sd = e_sd if e_sd else 1.0
    z = (err_s - e_mu) / e_sd
    anomalies = z > cfg.z_thresh

    # Plot on original series
    plt.figure(figsize=(10,5))
    plt.plot(s.index, s.values, label='EIA series', alpha=0.7)
    if anomalies.any():
        ts_anom = err_s.index[anomalies]
        vals = s.reindex(ts_anom).values
        plt.scatter(ts_anom, vals, color='red', s=24, label='AE anomaly')
    plt.legend()
    save_fig('eia_anomaly_autoencoder.png')

    # Also show error time series
    plt.figure(figsize=(10,3))
    plt.plot(err_s.index, err_s.values, label='Recon error')
    plt.axhline(e_mu + cfg.z_thresh*e_sd, color='red', lw=0.8, linestyle='--', label='threshold')
    plt.legend()
    save_fig('eia_anomaly_autoencoder_error.png')

if __name__ == '__main__':
    main()
