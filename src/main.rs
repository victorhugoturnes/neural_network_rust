use ndarray::Array2;
mod activation_layer;
mod activations;
mod dense_layer;
use dense_layer::Propagation;
mod nn_error;

fn main() {
    let learning_matrix: Vec<Array2<f64>> = vec![
        Array2::from(vec![[0., 0.]]).reversed_axes(),
        Array2::from(vec![[0., 1.]]).reversed_axes(),
        Array2::from(vec![[1., 0.]]).reversed_axes(),
        Array2::from(vec![[1., 1.]]).reversed_axes(),
    ];
    let true_output: Array2<f64> = Array2::from(vec![[0., 1., 1., 0.]]).reversed_axes();

    let mut network: Vec<Box<dyn Propagation>> = vec![
        Box::new(dense_layer::Layer::init(2, 3)),
        Box::new(activations::tanh_activator(3)),
        Box::new(dense_layer::Layer::init(3, 1)),
        Box::new(activations::tanh_activator(1)),
    ];

    let epochs = 10000;
    let learning_rate = 0.1;

    for i in 0..epochs {
        let mut error = 0.0;
        for (x, y) in Iterator::zip(learning_matrix.iter(), true_output.iter()) {
            let mut output = x.clone();
            for layer in &mut network {
                output = layer.forward_propagation(output);
            }
            error += nn_error::medium_squared_error(*y, output.clone());

            if i == epochs - 1 {
                unsafe {
                    print!("Calculated: {}", output.uget([0, 0]));
                    println!(", true answer: {y}");
                }
            }

            let mut grad = nn_error::medium_squared_error_derivative(*y, output);
            for layer in &mut network.iter_mut().rev() {
                grad = layer.backward_propagation(grad, learning_rate);
            }
            error /= x.len() as f64;
        }
        println!("error: {}, epoch {} of {}", error, i + 1, epochs);
    }
}
