use physics_informed_anomaly_detection_in_a_wind_turbine_using_python_with_an_autoencoder_transformer_core::reconstruction_errors;

fn main() {
    let n = 5000usize;
    let actual: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin() + 1.0).collect();
    let predicted: Vec<f64> = actual.iter().map(|&a| a * 0.95 + 0.02).collect();
    for _ in 0..10000 {
        let _ = reconstruction_errors(&actual, &predicted);
    }
}
