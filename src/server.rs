use crate::model::DynamicModel;
use crate::messages::Message;
use anyhow::Result;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};

use serde_json::Value;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use axum::http::{HeaderMap, HeaderValue, StatusCode};

use std::net::SocketAddr;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

#[derive(Clone)]
pub struct TrainingState {
    pub active: bool,
    pub epochs_run: usize,
    pub max_epochs: usize,
    pub stop_loss: f32,
    pub checkpoint_interval: usize,
    pub version: String,
    pub sync_interval_ms: u64,
}

#[derive(Clone)]
pub struct ServerState {
    pub model: Arc<Mutex<Option<DynamicModel>>>,
    pub dataset: Arc<Mutex<Vec<(Vec<f32>, Vec<f32>)>>>,
    pub training: Arc<Mutex<TrainingState>>,
    pub log_file: String,
}

pub struct Server;

impl Server {
    pub async fn run(tcp_addr: &str, http_addr: &str, sync_interval_ms: u64, log_file: String) -> Result<()> {
        tracing::info!("Starting server: TCP={}  HTTP={}", tcp_addr, http_addr);

        let model = Arc::new(Mutex::new(None));
        let dataset = Arc::new(Mutex::new(Vec::new()));

        let training = Arc::new(Mutex::new(TrainingState {
            active: false,
            epochs_run: 0,
            max_epochs: 0,
            stop_loss: 0.01,
            checkpoint_interval: 5,
            version: "v0".to_string(),
            sync_interval_ms,
        }));


        // --- Spawn TCP worker server ---
        let tcp_addr_owned = tcp_addr.to_string();
        let model_for_tcp = model.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::tcp_worker_loop(&tcp_addr_owned, model_for_tcp).await {
                tracing::error!("TCP worker loop failed: {:?}", e);
            }
        });

        // --- HTTP REST server ---
        let state = ServerState {
            model: model.clone(),
            dataset: dataset.clone(),
            training: training.clone(),
            log_file: log_file.clone(),
        };

        let addr: SocketAddr = http_addr.parse().expect("invalid http addr");

        // Build router and attach shared state
        let app = Router::new()
            .route("/create_model", post(Self::create_model))
            .route("/upload_dataset", post(Self::upload_dataset))
            .route("/dataset", get(Self::get_dataset))
            .route("/start_training", post(Self::start_training))
            .route("/stop_training", post(Self::stop_training))
            .route("/training_status", get(Self::training_status))
            .route("/train", post(Self::train_handler)) // keep for single-batch server-side training
            .route("/download_model", get(Self::download_model))
            .with_state(state);

        // Bind listener and run axum 0.7 style
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("HTTP server running on {}", addr);

        // Note: axum::serve takes the listener + router
        axum::serve(listener, app).await.unwrap();
        Ok(())
    }

    // ======================================================================
    // TCP Federated Worker Loop (unchanged behaviour)
    // ======================================================================
    async fn tcp_worker_loop(addr: &str, model: Arc<Mutex<Option<DynamicModel>>>) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("TCP workers listening on {}", addr);

        loop {
            let (stream, peer) = listener.accept().await?;
            tracing::info!("Worker connected: {}", peer);

            let model_for_task = model.clone();

            tokio::spawn(async move {
                let (r, mut w) = stream.into_split();
                let mut reader = BufReader::new(r);
                let mut line = String::new();

                // --- Send model on connect (clone, then send) ---
                {
                    let maybe_model = { model_for_task.lock().unwrap().clone() };
                    if let Some(m) = maybe_model {
                        let msg = Message::Model(m);
                        if let Ok(s) = serde_json::to_string(&msg) {
                            if let Err(e) = w.write_all(format!("{}\n", s).as_bytes()).await {
                                tracing::error!("Failed to send initial model to {}: {:?}", peer, e);
                                return;
                            }
                        }
                    }
                }

                loop {
                    line.clear();

                    match reader.read_line(&mut line).await {
                        Ok(0) => {
                            tracing::info!("Worker {} disconnected", peer);
                            break;
                        }
                        Ok(_) => {
                            match serde_json::from_str::<Message>(&line) {
                                Ok(Message::ModelUpdate(client_model)) => {
                                    {
                                        let mut global = model_for_task.lock().unwrap();
                                        if let Some(g) = &mut *global {
                                            g.merge_inplace(&client_model, 0.5);
                                        } else {
                                            *global = Some(client_model.clone());
                                        }
                                    } // lock dropped
                                    // reply with fresh copy
                                    let maybe_global = { model_for_task.lock().unwrap().clone() };
                                    if let Some(g) = maybe_global {
                                        let reply = Message::Model(g);
                                        if let Ok(s) = serde_json::to_string(&reply) {
                                            if let Err(e) = w.write_all(format!("{}\n", s).as_bytes()).await {
                                                tracing::error!("Failed to send updated model to {}: {:?}", peer, e);
                                                break;
                                            }
                                        }
                                    }
                                }
                                Ok(Message::RequestModel) => {
                                    let maybe_global = { model_for_task.lock().unwrap().clone() };
                                    if let Some(g) = maybe_global {
                                        let reply = Message::Model(g);
                                        if let Ok(s) = serde_json::to_string(&reply) {
                                            if let Err(e) = w.write_all(format!("{}\n", s).as_bytes()).await {
                                                tracing::error!("Failed to send model to {}: {:?}", peer, e);
                                                break;
                                            }
                                        }
                                    }
                                }
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::error!("Malformed message from {}: {:?} -- raw: {}", peer, e, line.trim());
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("TCP read error from {}: {:?}", peer, e);
                            break;
                        }
                    }
                }
            });
        }
    }

    // ======================================================================
    // POST /create_model (same)
    // ======================================================================
    async fn create_model(
        State(state): State<ServerState>,
        Json(payload): Json<Value>,
    ) -> Json<String> {
        let input_dim = payload.get("input_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

        let hidden_layers_json = payload
            .get("hidden_layers")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut hidden_layers = Vec::new();
        for h in hidden_layers_json {
            if let Some(n) = h.as_u64() {
                hidden_layers.push(n as usize);
            }
        }

        let output_dim = payload.get("output_dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(hidden_layers);
        layer_sizes.push(output_dim);

        let model = DynamicModel::new(layer_sizes);
        *state.model.lock().unwrap() = Some(model);

        tracing::info!("Created new global model");
        Json("model created".to_string())
    }

    // ======================================================================
    // POST /upload_dataset
    //   body: [ [[x...],[y...]], ... ]
    // ======================================================================
    async fn upload_dataset(
        State(state): State<ServerState>,
        Json(payload): Json<Vec<(Vec<f32>, Vec<f32>)>>,
    ) -> Json<String> {
        let mut ds = state.dataset.lock().unwrap();
        *ds = payload;
        tracing::info!("Uploaded dataset with {} samples", ds.len());
        Json(format!("dataset uploaded, {} samples", ds.len()))
    }

    // GET /dataset  (workers fetch)
    async fn get_dataset(State(state): State<ServerState>) -> Json<Vec<(Vec<f32>, Vec<f32>)>> {
        let ds = state.dataset.lock().unwrap();
        Json(ds.clone())
    }

    // ======================================================================
    // POST /start_training
    // payload: {"max_epochs":50,"stop_loss":0.01,"checkpoint_interval":5,"version":"exp1"}
    // ======================================================================
    async fn start_training(
        State(state): State<ServerState>,
        Json(payload): Json<Value>,
    ) -> Json<String> {
        // parse params
        let max_epochs = payload.get("max_epochs").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let stop_loss = payload.get("stop_loss").and_then(|v| v.as_f64()).unwrap_or(0.01) as f32;
        let checkpoint_interval = payload.get("checkpoint_interval").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let version = payload.get("version").and_then(|v| v.as_str()).unwrap_or("run").to_string();

        {
            let mut ts = state.training.lock().unwrap();
            ts.active = true;
            ts.max_epochs = max_epochs;
            ts.stop_loss = stop_loss;
            ts.checkpoint_interval = checkpoint_interval;
            ts.version = version.clone();
            ts.epochs_run = 0;
        }

        // spawn monitor task
        let model_clone = state.model.clone();
        let dataset_clone = state.dataset.clone();
        let training_clone = state.training.clone();
        let log_file = state.log_file.clone();

        tokio::spawn(async move {
            // training monitor loop
            loop {
                {
                    let ts = training_clone.lock().unwrap().clone();
                    if !ts.active {
                        tracing::info!("Training stopped by external signal");
                        break;
                    }
                }

                // Evaluate loss on dataset if model exists
                let maybe_model = { model_clone.lock().unwrap().clone() };
                let ds = { dataset_clone.lock().unwrap().clone() };

                let mut current_loss = std::f32::INFINITY;

                if let Some(m) = maybe_model {
                    if !ds.is_empty() {
                        let mut s = 0.0f32;
                        for (x, y) in ds.iter() {
                            let (pred, _) = m.forward(x);
                            for i in 0..pred.len() {
                                let e = pred[i] - y[i];
                                s += 0.5 * e * e;
                            }
                        }
                        current_loss = s / (ds.len() as f32);
                    }
                }

                // increment epoch count
                {
                    let mut ts = training_clone.lock().unwrap();
                    ts.epochs_run += 1;
                    let epoch = ts.epochs_run;
                    // checkpoint and logging
                    if epoch % ts.checkpoint_interval == 0 || current_loss.is_finite() && current_loss < ts.stop_loss {
                        // save checkpoint
                        if let Some(g) = model_clone.lock().unwrap().clone() {
                            let fname = format!("trained_model_{}_epoch{}.json", ts.version, epoch);
                            if let Ok(json) = serde_json::to_string_pretty(&g) {
                                if let Ok(mut f) = File::create(&fname) {
                                    let _ = f.write_all(json.as_bytes());
                                    tracing::info!("Saved checkpoint {}", fname);
                                }
                            }
                        }
                        // append training log
                        let _ = append_training_log(&log_file, ts.epochs_run, current_loss);
                    }

                    // check stopping criteria
                    if (current_loss.is_finite() && current_loss < ts.stop_loss) || ts.epochs_run >= ts.max_epochs {
                        ts.active = false;
                        tracing::info!("Stopping training: epoch={} loss={}", ts.epochs_run, current_loss);
                        // final save
                        if let Some(g) = model_clone.lock().unwrap().clone() {
                            let fname = format!("trained_model_{}_final.json", ts.version);
                            if let Ok(json) = serde_json::to_string_pretty(&g) {
                                if let Ok(mut f) = File::create(&fname) {
                                    let _ = f.write_all(json.as_bytes());
                                    tracing::info!("Saved final model {}", fname);
                                }
                            }
                        }
                        // append final log
                        let _ = append_training_log(&log_file, ts.epochs_run, current_loss);
                        break;
                    }
                }

                let interval_ms = { training_clone.lock().unwrap().sync_interval_ms };
                tokio::time::sleep(Duration::from_millis(interval_ms)).await;
            }
        });

        Json(format!("training started (version={})", version))
    }

    // POST /stop_training
    async fn stop_training(State(state): State<ServerState>) -> Json<String> {
        let mut ts = state.training.lock().unwrap();
        ts.active = false;
        Json("stopping training".to_string())
    }

    // GET /training_status
    async fn training_status(State(state): State<ServerState>) -> Json<Value> {
        let ts = state.training.lock().unwrap().clone();
        let obj = serde_json::json!({
            "active": ts.active,
            "epochs_run": ts.epochs_run,
            "max_epochs": ts.max_epochs,
            "stop_loss": ts.stop_loss,
            "checkpoint_interval": ts.checkpoint_interval,
            "version": ts.version,
        });
        Json(obj)
    }

    // server-side /train (single-batch helper, unchanged)
    async fn train_handler(
        State(state): State<ServerState>,
        Json(payload): Json<Vec<(Vec<f32>, Vec<f32>)>>,
    ) -> Json<String> {
        let mut model_guard = state.model.lock().unwrap();

        if model_guard.is_none() {
            return Json("No model created".to_string());
        }

        let model = model_guard.as_mut().unwrap();

        let xs: Vec<Vec<f32>> = payload.iter().map(|p| p.0.clone()).collect();
        let ys: Vec<Vec<f32>> = payload.iter().map(|p| p.1.clone()).collect();

        let loss = model.train_step(&xs, &ys, 0.01);

        tracing::info!("Server trained one batch, loss={}", loss);

        Json(format!("trained global model, loss={:.4}", loss))
    }

    // GET /download_model (final)
    async fn download_model(
        State(state): State<ServerState>,
    ) -> (StatusCode, (HeaderMap, String)) {
        let model_guard = state.model.lock().unwrap();

        if model_guard.is_none() {
            return (
                StatusCode::BAD_REQUEST,
                (HeaderMap::new(), "no model".to_string())
            );
        }

        let model = model_guard.as_ref().unwrap();
        let json = serde_json::to_string_pretty(model).unwrap();

        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        headers.insert("Content-Disposition", HeaderValue::from_static("attachment; filename=trained_model.json"));

        (StatusCode::OK, (headers, json))
    }
}

// helper to append logs
fn append_training_log(path: &str, epoch: usize, loss: f32) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "epoch={}, loss={}", epoch, loss)?;
    Ok(())
}
