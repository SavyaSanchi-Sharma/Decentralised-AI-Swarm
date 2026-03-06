// src/messages.rs
use serde::{Serialize, Deserialize};
use crate::model::DynamicModel;

/// All messages exchanged over the TCP control channel between server and workers.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "cmd")]
pub enum TcpMessage {
    // ----------------------------------------------------------------
    // Server → Worker
    // ----------------------------------------------------------------

    /// Server signals all registered workers to begin a training session.
    Start,

    /// Server broadcasts the (possibly updated) global model to a worker.
    /// Sent: (a) immediately after RequestModel if a model exists,
    ///        (b) after every successful gradient aggregation round,
    ///        (c) after a legacy ModelUpdate merge.
    Model {
        model: DynamicModel,
    },

    // ----------------------------------------------------------------
    // Worker → Server
    // ----------------------------------------------------------------

    /// Worker registers itself and requests the current model (if any).
    /// Sent once right after TCP connection is established.
    RequestModel {
        worker_id: String,
    },

    /// [DATA-PARALLEL — primary path]
    /// Worker sends raw per-layer gradients for one round.
    /// The server collects contributions from all expected workers,
    /// computes a sample-count-weighted average, applies a single
    /// optimiser step to the global model, and broadcasts the result.
    GradUpdate {
        worker_id: String,
        round:     u64,
        /// Per-layer weight gradients — same shape as DynamicModel::weights.
        dw: Vec<Vec<f32>>,
        /// Per-layer bias gradients — same shape as DynamicModel::biases.
        db: Vec<Vec<f32>>,
        /// Number of local samples used; drives the weighted average.
        n_samples: usize,
    },

    /// [LEGACY — federated averaging mode]
    /// Worker sends its locally-trained full model weights.
    /// The server merges via exponential moving average (alpha = 0.5)
    /// and broadcasts the merged model.
    /// Kept for backwards compatibility; prefer GradUpdate for new work.
    ModelUpdate {
        worker_id: String,
        round:     u64,
        model:     DynamicModel,
    },

    // ----------------------------------------------------------------
    // Bidirectional
    // ----------------------------------------------------------------

    /// Keep-alive ping.  Workers send this every ~2 s.
    /// Server reflects it on the WS broadcast channel.
    Heartbeat {
        worker_id: String,
    },
}
