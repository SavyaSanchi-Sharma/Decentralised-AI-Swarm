// src/messages.rs
use serde::{Serialize, Deserialize};
use crate::model::DynamicModel;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "cmd")]
pub enum TcpMessage {
    Start,

    /// Worker asks server for the initial model
    RequestModel {
        worker_id: String,
    },

    /// Worker sends weight update
    ModelUpdate {
        worker_id: String,
        round: u64,
        model: DynamicModel,
    },

    /// Server broadcasts merged model
    Model {
        model: DynamicModel,
    },

    /// Optional heartbeat
    Heartbeat {
        worker_id: String,
    },
}
