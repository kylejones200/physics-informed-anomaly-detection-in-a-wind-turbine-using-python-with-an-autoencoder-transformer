//! Reconstruction error magnitudes for anomaly detection.

pub fn reconstruction_errors(actual: &[f64], predicted: &[f64]) -> Vec<f64> {
    assert_eq!(actual.len(), predicted.len());
    actual
        .iter()
        .zip(predicted)
        .map(|(&a, &p)| (a - p).abs())
        .collect()
}
