# Physics-Informed Anomaly Detection in Wind Turbine

This project demonstrates physics-informed anomaly detection for wind turbine data using wavelet denoising, Isolation Forest, and tensor preparation for deep learning models.

## Article

Medium article: [Physics-Informed Anomaly Detection in Wind Turbine](https://medium.com/@kylejones_47003/physics-informed-anomaly-detection-in-a-wind-turbine-using-python-with-an-autoencoder-transformer-06eb68aeb0e8)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Anomaly detection functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files and tensor chunks
└── images/            # Generated plots and figures
```

## Data Format

The input CSV should contain the following columns (or adjust in config.yaml):
- temp
- pressure
- humidity
- altitude
- voltage
- power
- rpm
- gearbox_vibration

## Configuration

Edit `config.yaml` to customize:
- Feature names
- Preprocessing parameters (wavelet type, contamination level)
- Tensor creation parameters
- Which analyses to run

## Caveats

- The script requires a CSV file with wind turbine sensor data.
- Wavelet denoising uses Daubechies 6 (db6) wavelet by default.
- Isolation Forest contamination parameter controls the expected proportion of anomalies.
- Tensor chunks are saved as .npy files for use with deep learning models.
