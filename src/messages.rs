use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Message {
    Weights(Vec<f32>),
    Ack(String),
    RequestSync,
}
