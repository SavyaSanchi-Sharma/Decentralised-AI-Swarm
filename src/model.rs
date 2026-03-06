// src/model.rs
use rand::Rng;
use serde::{Serialize, Deserialize};

/// A fully-connected feed-forward network with arbitrary depth.
///
/// Architecture
/// ─────────────
/// • All hidden layers: ReLU activation.
/// • Output layer: linear (no activation).
///
/// Weight storage (flat row-major): `weights[l]` has length `out_n * in_n`
/// where `out_n = layer_sizes[l+1]`, `in_n = layer_sizes[l]`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DynamicModel {
    pub layer_sizes: Vec<usize>,
    pub weights:     Vec<Vec<f32>>,
    pub biases:      Vec<Vec<f32>>,
}

impl DynamicModel {
    /// Construct a randomly-initialised network.
    /// `layer_sizes` must have at least 2 elements (input, output).
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least input + output layer");
        let mut rng = rand::rng();
        let mut weights = Vec::new();
        let mut biases  = Vec::new();

        for i in 0..(layer_sizes.len() - 1) {
            let out = layer_sizes[i + 1];
            let inp = layer_sizes[i];

            // He-like init (small uniform) — good enough for shallow nets
            let mut w = vec![0.0f32; out * inp];
            for v in w.iter_mut() {
                *v = rng.random::<f32>() * 0.2 - 0.1;
            }
            weights.push(w);
            biases.push(vec![0.0f32; out]);
        }

        Self { layer_sizes, weights, biases }
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Run one forward pass.
    ///
    /// Returns `(output, activations)` where `activations[i]` is the
    /// post-activation vector of layer `i` (layer 0 == raw input).
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut activations = vec![x.to_vec()];
        let mut inp = x.to_vec();

        for (layer_idx, w) in self.weights.iter().enumerate() {
            let out_n = self.layer_sizes[layer_idx + 1];
            let in_n  = self.layer_sizes[layer_idx];
            let is_last = layer_idx + 1 == self.layer_sizes.len() - 1;

            let mut out = vec![0.0f32; out_n];
            for i in 0..out_n {
                let mut s = self.biases[layer_idx][i];
                for j in 0..in_n {
                    s += w[i * in_n + j] * inp[j];
                }
                // Last layer: linear; hidden layers: ReLU
                out[i] = if is_last { s } else { s.max(0.0) };
            }

            inp = out.clone();
            activations.push(inp.clone());
        }

        (inp, activations)
    }

    /// Convenience: forward pass without keeping activations.
    #[allow(dead_code)]
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        self.forward(x).0
    }

    // -----------------------------------------------------------------------
    // Gradient computation (data-parallel path)
    // -----------------------------------------------------------------------

    /// Compute MSE gradients over a batch **without** modifying weights.
    ///
    /// Returns `(dw, db, mean_loss)`.  The caller ships `dw`/`db` to the
    /// server which aggregates across all workers before applying one
    /// optimiser step.
    pub fn compute_gradients(
        &self,
        xs: &[Vec<f32>],
        ys: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f32) {
        if xs.is_empty() {
            let gw: Vec<Vec<f32>> = self.weights.iter().map(|w| vec![0.0; w.len()]).collect();
            let gb: Vec<Vec<f32>> = self.biases.iter().map(|b| vec![0.0; b.len()]).collect();
            return (gw, gb, 0.0);
        }

        let batch = xs.len() as f32;
        let mut total_loss = 0.0f32;

        let mut gw: Vec<Vec<f32>> = self.weights.iter().map(|w| vec![0.0; w.len()]).collect();
        let mut gb: Vec<Vec<f32>> = self.biases.iter().map(|b| vec![0.0; b.len()]).collect();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let (pred, acts) = self.forward(x);

            // MSE forward + output-layer delta
            let mut upstream: Vec<f32> = pred.iter().zip(y.iter())
                .map(|(p, t)| {
                    let e = p - t;
                    total_loss += 0.5 * e * e;
                    e
                })
                .collect();

            // Backprop through layers (reverse)
            for layer_idx in (0..self.weights.len()).rev() {
                let out_n = self.layer_sizes[layer_idx + 1];
                let in_n  = self.layer_sizes[layer_idx];

                // Accumulate weight/bias gradients
                for i in 0..out_n {
                    gb[layer_idx][i] += upstream[i];
                    for j in 0..in_n {
                        gw[layer_idx][i * in_n + j] += upstream[i] * acts[layer_idx][j];
                    }
                }

                // Propagate upstream through ReLU (skip for input layer)
                if layer_idx > 0 {
                    let mut downstream = vec![0.0f32; in_n];
                    for j in 0..in_n {
                        let mut s = 0.0f32;
                        for i in 0..out_n {
                            s += self.weights[layer_idx][i * in_n + j] * upstream[i];
                        }
                        // ReLU backward: pass gradient only where activation was positive
                        downstream[j] = if acts[layer_idx][j] > 0.0 { s } else { 0.0 };
                    }
                    upstream = downstream;
                }
            }
        }

        // Normalise by batch size
        for li in 0..gw.len() {
            for v in gw[li].iter_mut() { *v /= batch; }
            for v in gb[li].iter_mut() { *v /= batch; }
        }

        (gw, gb, total_loss / batch)
    }

    // -----------------------------------------------------------------------
    // Weight update
    // -----------------------------------------------------------------------

    /// Apply pre-averaged gradients (`dw`, `db`) with learning rate `lr`.
    /// Called on the server after the weighted aggregation step.
    pub fn apply_gradients(&mut self, dw: &[Vec<f32>], db: &[Vec<f32>], lr: f32) {
        for li in 0..self.weights.len() {
            for i in 0..self.weights[li].len() {
                self.weights[li][i] -= lr * dw[li][i];
            }
            for i in 0..self.biases[li].len() {
                self.biases[li][i] -= lr * db[li][i];
            }
        }
    }

    /// Single-process SGD — compute gradients and apply in one shot.
    /// Kept for local testing / benchmarking; not used in the parallelised path.
    #[allow(dead_code)]
    pub fn train_step(&mut self, xs: &[Vec<f32>], ys: &[Vec<f32>], lr: f32) -> f32 {
        let (dw, db, loss) = self.compute_gradients(xs, ys);
        self.apply_gradients(&dw, &db, lr);
        loss
    }

    // -----------------------------------------------------------------------
    // Legacy federated averaging helper
    // -----------------------------------------------------------------------

    /// Merge `other`'s weights into `self` via exponential moving average.
    /// `alpha` controls how much of `self` to retain (0 = full replacement).
    pub fn merge_inplace(&mut self, other: &DynamicModel, alpha: f32) {
        assert_eq!(self.layer_sizes, other.layer_sizes,
            "Cannot merge models with different architectures");

        for li in 0..self.weights.len() {
            for i in 0..self.weights[li].len() {
                self.weights[li][i] =
                    alpha * self.weights[li][i] + (1.0 - alpha) * other.weights[li][i];
            }
            for i in 0..self.biases[li].len() {
                self.biases[li][i] =
                    alpha * self.biases[li][i] + (1.0 - alpha) * other.biases[li][i];
            }
        }
    }
}
