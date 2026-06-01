#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import reconstruction_errors  # noqa: E402

def main() -> None:
    n = 5000
    actual = np.ascontiguousarray(np.sin(np.arange(n) * 0.01) + 1.0)
    predicted = actual * 0.95 + 0.02
    t0 = time.perf_counter()
    for _ in range(200):
        reconstruction_errors(actual, predicted)
    py_s = time.perf_counter() - t0
    try:
        import physics_informed_anomaly_detection_in_a_wind_turbine_using_python_with_an_autoencoder_transformer_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(actual, predicted, 10000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        reconstruction_errors(actual, predicted),
        np.asarray(rs.reconstruction_errors_py(actual, predicted)),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
