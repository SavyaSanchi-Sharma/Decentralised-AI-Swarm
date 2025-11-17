// src/model.rs
use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DynamicModel {
    pub layer_sizes: Vec<usize>,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
}

impl DynamicModel {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..(layer_sizes.len() - 1) {
            let out = layer_sizes[i + 1];
            let inp = layer_sizes[i];

            let mut w = vec![0.0; out * inp];
            let b = vec![0.0; out];

            for v in w.iter_mut() {
                *v = rng.random::<f32>() * 0.2 - 0.1;
            }

            weights.push(w);
            biases.push(b);
        }

        Self { layer_sizes, weights, biases }
    }

    pub fn forward(&self, x: &Vec<f32>) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut activations = vec![x.clone()];
        let mut inp = x.clone();

        for (layer_idx, w) in self.weights.iter().enumerate() {
            let out_n = self.layer_sizes[layer_idx + 1];
            let in_n = self.layer_sizes[layer_idx];

            let mut out = vec![0.0; out_n];

            for i in 0..out_n {
                let mut s = self.biases[layer_idx][i];
                for j in 0..in_n {
                    s += w[i * in_n + j] * inp[j];
                }
                // final layer: linear, else ReLU
                out[i] = if layer_idx + 1 == self.layer_sizes.len() - 1 {
                    s
                } else {
                    if s > 0.0 { s } else { 0.0 }
                };
            }

            inp = out.clone();
            activations.push(inp.clone());
        }

        (inp, activations)
    }

    pub fn train_step(&mut self, xs: &Vec<Vec<f32>>, ys: &Vec<Vec<f32>>, lr: f32) -> f32 {
        if xs.is_empty() { return 0.0; }
        let batch = xs.len() as f32;
        let mut total_loss = 0.0;

        let mut gw: Vec<Vec<f32>> =
            self.weights.iter().map(|w| vec![0.0; w.len()]).collect();
        let mut gb: Vec<Vec<f32>> =
            self.biases.iter().map(|b| vec![0.0; b.len()]).collect();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let (pred, acts) = self.forward(x);

            let mut loss = 0.0;
            let mut dout = vec![0.0; pred.len()];

            for i in 0..pred.len() {
                let e = pred[i] - y[i];
                loss += 0.5 * e * e;
                dout[i] = e;
            }

            total_loss += loss;

            let mut upstream = dout;

            for layer_idx in (0..self.weights.len()).rev() {
                let out_n = self.layer_sizes[layer_idx + 1];
                let in_n = self.layer_sizes[layer_idx];

                for i in 0..out_n {
                    gb[layer_idx][i] += upstream[i];
                    for j in 0..in_n {
                        let a = acts[layer_idx][j];
                        gw[layer_idx][i * in_n + j] += upstream[i] * a;
                    }
                }

                if layer_idx > 0 {
                    let mut downstream = vec![0.0; in_n];
                    for j in 0..in_n {
                        let mut s = 0.0;
                        for i in 0..out_n {
                            s += self.weights[layer_idx][i * in_n + j] * upstream[i];
                        }
                        // ReLU backward
                        downstream[j] = if acts[layer_idx][j] > 0.0 { s } else { 0.0 };
                    }
                    upstream = downstream;
                }
            }
        }

        for li in 0..self.weights.len() {
            for i in 0..self.weights[li].len() {
                self.weights[li][i] -= lr * gw[li][i] / batch;
            }
            for i in 0..self.biases[li].len() {
                self.biases[li][i] -= lr * gb[li][i] / batch;
            }
        }

        total_loss / batch
    }

    pub fn merge_inplace(&mut self, other: &DynamicModel, alpha: f32) {
        assert_eq!(self.layer_sizes, other.layer_sizes);

        for li in 0..self.weights.len() {
            for i in 0..self.weights[li].len() {
                self.weights[li][i] = alpha * self.weights[li][i] + (1.0 - alpha) * other.weights[li][i];
            }
            for i in 0..self.biases[li].len() {
                self.biases[li][i] = alpha * self.biases[li][i] + (1.0 - alpha) * other.biases[li][i];
            }
        }
    }
}
