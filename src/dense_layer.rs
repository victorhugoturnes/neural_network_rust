use ndarray::Array2;
use ndarray::Ix2;
use rand::Rng;

pub trait Propagation {
    fn forward_propagation(&mut self, input: Array2<f64>) -> Array2<f64>;
    fn backward_propagation(&mut self, output: Array2<f64>, learning_rate: f64) -> Array2<f64>;
}

pub struct Layer {
    input: Array2<f64>,
    weights: Array2<f64>,
    biases: Array2<f64>,
}

impl Layer {
    pub fn init(nodes_in: usize, nodes_out: usize) -> Layer {
        Layer {
            input: Array2::<f64>::default(Ix2(nodes_in, 1)),
            weights: Array2::<f64>::from_shape_simple_fn(Ix2(nodes_out, nodes_in), &rng_float),
            biases: Array2::<f64>::from_shape_simple_fn(Ix2(nodes_out, 1), &rng_float),
        }
    }
}
impl Propagation for Layer {
    fn forward_propagation(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.input = input;
        self.weights.dot(&self.input) + &self.biases
    }
    fn backward_propagation(&mut self, output: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let weights_gradient = output.dot(&self.input.clone().reversed_axes());
        let new_weights = self.weights.clone().reversed_axes().dot(&output);
        self.weights = &self.weights - (learning_rate * &weights_gradient);
        self.biases = &self.biases - (learning_rate * &output);
        new_weights
    }
}

fn rng_float() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0.0..1.0)
}
