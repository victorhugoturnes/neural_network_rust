use ndarray::Array2;
use ndarray::Ix2;

use crate::dense_layer::Propagation;

pub struct Layer {
    activation: fn(Array2<f64>) -> Array2<f64>,
    activation_derivate: fn(Array2<f64>) -> Array2<f64>,
    input: Array2<f64>,
}

impl Layer {
    pub fn init(
        activation: fn(Array2<f64>) -> Array2<f64>,
        activation_derivate: fn(Array2<f64>) -> Array2<f64>,
        nodes_in: usize,
    ) -> Layer {
        Layer {
            activation,
            activation_derivate,
            input: Array2::<f64>::default(Ix2(nodes_in, 1)),
        }
    }
}
impl Propagation for Layer {
    fn forward_propagation(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.input = input.clone();
        (self.activation)(input)
    }
    fn backward_propagation(&mut self, output_gradient: Array2<f64>, _: f64) -> Array2<f64> {
        output_gradient * (self.activation_derivate)(self.input.clone())
    }
}
