use rand::Rng;

#[derive(Clone, Debug)]
pub struct Model {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl Model {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut w1 = vec![0.0; hidden_dim * input_dim];
        let mut b1 = vec![0.0; hidden_dim];
        let mut w2 = vec![0.0; output_dim * hidden_dim];
        let mut b2 = vec![0.0; output_dim];
        for v in w1.iter_mut() { *v = rng.gen::<f32>() * 0.2 - 0.1; }
        for v in w2.iter_mut() { *v = rng.gen::<f32>() * 0.2 - 0.1; }
        Self { input_dim, hidden_dim, output_dim, w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut s = self.b1[i];
            for j in 0..self.input_dim {
                s += self.w1[i * self.input_dim + j] * x[j];
            }
            h[i] = if s > 0.0 { s } else { 0.0 };
        }

        let mut out = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut s = self.b2[i];
            for j in 0..self.hidden_dim {
                s += self.w2[i * self.hidden_dim + j] * h[j];
            }
            out[i] = s;
        }
        (out, h)
    }

    pub fn train_step(&mut self, xs: &Vec<Vec<f32>>, ys: &Vec<Vec<f32>>, lr: f32) -> f32 {
        let batch = xs.len();
        let mut total_loss = 0.0;
        let mut gw2 = vec![0.0; self.w2.len()];
        let mut gb2 = vec![0.0; self.b2.len()];
        let mut gw1 = vec![0.0; self.w1.len()];
        let mut gb1 = vec![0.0; self.b1.len()];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let (pred, h) = self.forward(x);
            let mut dout = vec![0.0; self.output_dim];
            let mut loss = 0.0;

            for i in 0..self.output_dim {
                let e = pred[i] - y[i];
                loss += e * e * 0.5;
                dout[i] = e;
            }
            total_loss += loss;

            for i in 0..self.output_dim {
                gb2[i] += dout[i];
                for j in 0..self.hidden_dim {
                    gw2[i * self.hidden_dim + j] += dout[i] * h[j];
                }
            }

            let mut dh = vec![0.0; self.hidden_dim];
            for j in 0..self.hidden_dim {
                let mut s = 0.0;
                for i in 0..self.output_dim {
                    s += self.w2[i * self.hidden_dim + j] * dout[i];
                }
                dh[j] = if h[j] > 0.0 { s } else { 0.0 };
            }

            for i in 0..self.hidden_dim {
                gb1[i] += dh[i];
                for j in 0..self.input_dim {
                    gw1[i * self.input_dim + j] += dh[i] * x[j];
                }
            }
        }

        let b = batch as f32;
        for i in 0..self.w2.len() { self.w2[i] -= lr * gw2[i] / b; }
        for i in 0..self.b2.len() { self.b2[i] -= lr * gb2[i] / b; }
        for i in 0..self.w1.len() { self.w1[i] -= lr * gw1[i] / b; }
        for i in 0..self.b1.len() { self.b1[i] -= lr * gb1[i] / b; }

        total_loss / b
    }

    pub fn to_flat_vec(&self) -> Vec<f32> {
        let mut v = Vec::new();
        v.push(self.input_dim as f32);
        v.push(self.hidden_dim as f32);
        v.push(self.output_dim as f32);
        v.extend(&self.w1);
        v.extend(&self.b1);
        v.extend(&self.w2);
        v.extend(&self.b2);
        v
    }

    pub fn from_flat_vec(v: &Vec<f32>) -> Result<Self, String> {
        if v.len() < 3 { return Err("too short".into()); }
        let input_dim = v[0] as usize;
        let hidden_dim = v[1] as usize;
        let output_dim = v[2] as usize;
        let mut idx = 3;
        let w1 = v[idx..idx + hidden_dim * input_dim].to_vec();
        idx += hidden_dim * input_dim;
        let b1 = v[idx..idx + hidden_dim].to_vec();
        idx += hidden_dim;
        let w2 = v[idx..idx + output_dim * hidden_dim].to_vec();
        idx += output_dim * hidden_dim;
        let b2 = v[idx..idx + output_dim].to_vec();
        Ok(Self { input_dim, hidden_dim, output_dim, w1, b1, w2, b2 })
    }

    pub fn merge_inplace(&mut self, other: &Model, alpha: f32) {
        for (a, b) in self.w1.iter_mut().zip(&other.w1) { *a = alpha * *a + (1. - alpha) * *b; }
        for (a, b) in self.b1.iter_mut().zip(&other.b1) { *a = alpha * *a + (1. - alpha) * *b; }
        for (a, b) in self.w2.iter_mut().zip(&other.w2) { *a = alpha * *a + (1. - alpha) * *b; }
        for (a, b) in self.b2.iter_mut().zip(&other.b2) { *a = alpha * *a + (1. - alpha) * *b; }
    }
}
