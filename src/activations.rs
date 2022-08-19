use ndarray::Array2;

use crate::activation_layer::Layer;

pub fn tanh_activator(size: usize) -> Layer {
    Layer::init(tanh_function, tanh_derivative, size)
}

fn tanh_function(x: Array2<f64>) -> Array2<f64> {
    x.map(|x| -> f64 { x.tanh() })
}
fn tanh_derivative(x: Array2<f64>) -> Array2<f64> {
    1.0 - x.map(|x| -> f64 { x.tanh().powi(2) })
}
//possible other activators, currently not in use
/*
pub fn sigmod_activator(size: usize) -> Layer {
    Layer::init(sigmoid_function, sigmoid_derivative, size)
}

fn sigmoid_derivative(x: Array2<f64>) -> Array2<f64> {
    let sigmoid = sigmoid_function(x);
    sigmoid.clone() * (1.0 - sigmoid)
}

fn sigmoid_function(x: Array2<f64>) -> Array2<f64> {
    x.map(|x| -> f64 { 1.0 / (1.0 + f64::exp(-x)) - 0.5 })
}
*/
