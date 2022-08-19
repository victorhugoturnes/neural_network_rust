use ndarray::Array2;

pub fn medium_squared_error(correct: f64, calculated: Array2<f64>) -> f64 {
    (correct - calculated).map(|x| x.powi(2)).mean().unwrap()
}

pub fn medium_squared_error_derivative(correct: f64, calculated: Array2<f64>) -> Array2<f64> {
    let size: f64 = calculated.len() as f64;
    (2.0 * (calculated - correct)) / size
}
