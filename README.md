# Physics-Informed Anomaly Detection in Wind Turbine

This project demonstrates physics-informed anomaly detection for wind turbine data using wavelet denoising, Isolation Forest, and tensor preparation for deep learning models.

## Business context

The challenge we're trying to address here is to detect anomalies in the components of a Wind Turbine. Each wind turbine has many sensors that reads data like:

- external temperature - Rotor speed - Air pressure - Voltage (or current) in the generator - Vibration in the GearBox, Generator, and Tower

Depending on the type of the anomalies we want to detect, we need to select one or more features and then prepare a dataset that 'explains' the anomalies. We are interested in three types of anomalies:

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
├── images/            # Generated plots and figures
├── rust/                   # Rust port (core + PyO3 + CLI bench)
├── benchmark_rust.py       # Python vs Rust benchmark
├── src/compute_kernel.py   # Python/numpy reference kernel
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

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — reconstruction errors. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p physics_informed_anomaly_detection_in_a_wind_turbine_using_python_with_an_autoencoder_transformer_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).