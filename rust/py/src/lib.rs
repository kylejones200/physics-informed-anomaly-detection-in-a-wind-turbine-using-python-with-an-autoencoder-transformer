use physics_informed_anomaly_detection_in_a_wind_turbine_using_python_with_an_autoencoder_transformer_core::reconstruction_errors;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn reconstruction_errors_py<'py>(
    py: Python<'py>,
    actual: PyReadonlyArray1<f64>,
    predicted: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(reconstruction_errors(actual.as_slice()?, predicted.as_slice()?).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (actual, predicted, iterations=10_000))]
fn bench_kernel_py(
    actual: PyReadonlyArray1<f64>,
    predicted: PyReadonlyArray1<f64>,
    iterations: usize,
) -> PyResult<f64> {
    let a = actual.as_slice()?.to_vec();
    let p = predicted.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = reconstruction_errors(&a, &p);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn physics_informed_anomaly_detection_in_a_wind_turbine_using_python_with_an_autoencoder_transformer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reconstruction_errors_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
