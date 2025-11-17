// src/server.rs
use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
    sync::mpsc::UnboundedSender,
};

use crate::{messages::TcpMessage, model::DynamicModel};

#[derive(Clone)]
pub struct TrainingParams {
    pub stop_loss: f32,
    pub max_epochs: usize,
}

#[derive(Clone)]
pub struct ServerState {
    pub model: Arc<Mutex<Option<DynamicModel>>>,
    pub datasets: Arc<Mutex<HashMap<String, Vec<(Vec<f32>, Vec<f32>)>>>>,
    pub worker_sync: Arc<Mutex<HashMap<String, usize>>>,
    pub params: Arc<Mutex<TrainingParams>>,
    pub tcp_senders: Arc<Mutex<HashMap<String, UnboundedSender<String>>>>,
    pub api_key: String,
}

pub struct Server;

impl Server {
    pub async fn run(tcp_addr: &str, http_addr: &str, api_key: String) -> Result<()> {
        println!("\n====== FEDERATED SERVER STARTED ======");
        println!("HTTP: {}", http_addr);
        println!("TCP : {}", tcp_addr);
        println!("API : {}", api_key);
        println!("=====================================");

        let state = ServerState {
            model: Arc::new(Mutex::new(None)),
            datasets: Arc::new(Mutex::new(HashMap::new())),
            worker_sync: Arc::new(Mutex::new(HashMap::new())),
            params: Arc::new(Mutex::new(TrainingParams { stop_loss: 0.01, max_epochs: 20 })),
            tcp_senders: Arc::new(Mutex::new(HashMap::new())),
            api_key,
        };

        // spawn tcp loop
        let tcp_addr_owned = tcp_addr.to_string();
        let tcp_state = state.clone(); // Clone the local `state` variable, not `self.state`

        // 2. Spawn the task, moving the owned data
        tokio::spawn(async move {
            if let Err(e) = Self::tcp_accept_loop(&tcp_addr_owned, tcp_state).await {
                // Handle the error instead of unwrapping
                eprintln!("TCP loop failed: {:?}", e); 
            }
        });

        // build router
        let router = Router::new()
            .route("/create_model", post(Self::create_model))
            .route("/upload_dataset", post(Self::upload_dataset))
            .route("/register_worker", post(Self::register_worker))
            .route("/set_training_params", post(Self::set_training_params))
            .route("/start_training", post(Self::start_training))
            .route("/get_dataset", get(Self::get_dataset))
            .route("/get_model", get(Self::get_model))
            .route("/get_training_params", get(Self::get_training_params))
            .route("/sync_status", get(Self::sync_status))
            .route("/download_model", get(Self::download_model))
            .with_state(state.clone());

        // start HTTP
        let addr: SocketAddr = http_addr.parse()?;
let listener = TcpListener::bind(addr).await?;

axum::serve(listener, router).await?;

        Ok(())
    }

    // ------------------ TCP accept loop ------------------
    async fn tcp_accept_loop(addr: &str, state: ServerState) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        println!("TCP listening on {}", addr);

        loop {
            let (stream, peer) = listener.accept().await?;
            println!("TCP connection from {}", peer);
            let st = state.clone();
            tokio::spawn(async move {
                if let Err(e) = Server::handle_tcp(stream, st).await {
                    eprintln!("handle_tcp error: {:?}", e);
                }
            });
        }
    }

    async fn handle_tcp(stream: TcpStream, state: ServerState) -> Result<()> {
        let peer = stream.peer_addr()?.to_string();
        let (r, w) = stream.into_split();
        let mut reader = BufReader::new(r);
        
        // create local channel for sending server->worker messages
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let mut worker_id_opt: Option<String> = None;
        
        // spawn writer task
        let mut writer = w;
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let _ = writer.write_all(msg.as_bytes()).await;
                let _ = writer.write_all(b"\n").await;
            }
        });
        
        // read loop
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                println!("TCP {} disconnected", peer);
                break;
            }
            
            let raw = line.trim();
            if raw.is_empty() { continue; }
            
            match serde_json::from_str::<TcpMessage>(raw) {
                Ok(TcpMessage::RequestModel { worker_id }) => {
                    println!("TCP register: {}", worker_id);
                    worker_id_opt = Some(worker_id.clone());
                    state.tcp_senders.lock().unwrap().insert(worker_id.clone(), tx.clone());
                    
                    // send current model if exists
                    if let Some(m) = &*state.model.lock().unwrap() {
                        let msg = TcpMessage::Model { model: m.clone() };
                        let s = serde_json::to_string(&msg).unwrap();
                        let _ = tx.send(s);
                    }
                }
                Ok(TcpMessage::ModelUpdate { worker_id, round, model }) => {
                    println!("TCP update from {} round {}", worker_id, round);
                    
                    // merge into global
                    {
                        let mut global = state.model.lock().unwrap();
                        if let Some(g) = &mut *global {
                            g.merge_inplace(&model, 0.5);
                        } else {
                            *global = Some(model.clone());
                        }
                    }
                    
                    // track updates count
                    *state.worker_sync.lock().unwrap().entry(worker_id).or_insert(0) += 1;
                    
                    // broadcast merged model to all workers
                    let broadcast_model = {
                        let guard = state.model.lock().unwrap();
                        guard.as_ref().map(|m| TcpMessage::Model { model: m.clone() })
                    };
                    
                    if let Some(msg) = broadcast_model {
                        let s = serde_json::to_string(&msg).unwrap();
                        let mut dead = Vec::new();
                        
                        {
                            let senders = state.tcp_senders.lock().unwrap();
                            for (wid, sender) in senders.iter() {
                                if sender.send(s.clone()).is_err() {
                                    dead.push(wid.clone());
                                }
                            }
                        }
                        
                        if !dead.is_empty() {
                            let mut map = state.tcp_senders.lock().unwrap();
                            for d in dead { 
                                map.remove(&d); 
                                println!("Removed dead tcp {}", d); 
                            }
                        }
                    }
                }
                Ok(TcpMessage::Heartbeat { worker_id }) => {
                    // ignore or update last seen
                    println!("heartbeat {}", worker_id);
                }
                Ok(TcpMessage::Start) => {
                    // Workers should never send Start messages; ignore
                    println!("Unexpected Start message from worker");
                }
                Ok(TcpMessage::Model { .. }) => {
                    // Workers should never send Model messages; ignore
                    println!("Unexpected Model message from worker");
                }
                Err(e) => {
                    println!("TCP parse error: {} | raw: {}", e, raw);
                }
            }
        }
        
        // cleanup
        if let Some(wid) = worker_id_opt {
            state.tcp_senders.lock().unwrap().remove(&wid);
        }
        
        Ok(())
    }

    // ------------------ HTTP endpoints ------------------

    fn check_api(headers: &HeaderMap, expected: &str) -> Result<(), StatusCode> {
        if expected.is_empty() { return Ok(()); }
        let got = headers.get("x-api-key").and_then(|v| v.to_str().ok());
        match got {
            Some(k) if k == expected => Ok(()),
            _ => {
                println!("⛔ Unauthorized (got={:?}, expected=...)", got);
                Err(StatusCode::UNAUTHORIZED)
            }
        }
    }

    async fn create_model(
        State(state): State<ServerState>,
        headers: HeaderMap,
        Json(payload): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;

        let input_dim = payload.get("input_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
        let hidden = payload.get("hidden_layers").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let output_dim = payload.get("output_dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let mut layers = vec![input_dim];
        for h in hidden { if let Some(n) = h.as_u64() { layers.push(n as usize); } }
        layers.push(output_dim);

        let model = DynamicModel::new(layers.clone());
        *state.model.lock().unwrap() = Some(model);

        println!("✔ Model created: {:?}", layers);
        Ok(Json(json!({"status":"ok","layer_sizes": layers})))
    }

    async fn upload_dataset(
        State(state): State<ServerState>,
        headers: HeaderMap,
        Json(payload): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;

        let worker_id = payload["worker_id"].as_str().ok_or(StatusCode::BAD_REQUEST)?.to_string();
        let data = payload["data"].as_array().ok_or(StatusCode::BAD_REQUEST)?;

        let mut parsed: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for sample in data.iter() {
            let arr_x = sample["x"].as_array().ok_or(StatusCode::BAD_REQUEST)?;
            let arr_y = sample["y"].as_array().ok_or(StatusCode::BAD_REQUEST)?;
            let x = arr_x.iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let y = arr_y.iter().map(|v| v.as_f64().unwrap() as f32).collect();
            parsed.push((x, y));
        }

        state.datasets.lock().unwrap().insert(worker_id.clone(), parsed);
        println!("📥 Valid dataset uploaded for {}", worker_id);
        Ok(Json(json!({"status":"ok"})))
    }

    async fn register_worker(
        State(state): State<ServerState>,
        headers: HeaderMap,
        Json(payload): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;
        let worker_id = payload["worker_id"].as_str().ok_or(StatusCode::BAD_REQUEST)?.to_string();

        state.datasets.lock().unwrap().entry(worker_id.clone()).or_insert(vec![]);
        state.worker_sync.lock().unwrap().entry(worker_id.clone()).or_insert(0usize);

        println!("🟢 Worker registered: {}", worker_id);
        Ok(Json(json!({"status":"registered","worker_id":worker_id})))
    }

    async fn set_training_params(
        State(state): State<ServerState>,
        headers: HeaderMap,
        Json(payload): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;
        let mut p = state.params.lock().unwrap();
        p.stop_loss = payload.get("stop_loss").and_then(|v| v.as_f64()).unwrap_or(0.01) as f32;
        p.max_epochs = payload.get("max_epochs").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        println!("⚙ Updated training params stop_loss={} max_epochs={}", p.stop_loss, p.max_epochs);
        Ok(Json(json!({"status":"ok"})))
    }

    async fn start_training(
        State(state): State<ServerState>,
        headers: HeaderMap,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;
        println!("🚀 Broadcast START to workers");

        let msg = serde_json::to_string(&TcpMessage::Start).unwrap();
        let mut dead = Vec::new();
        let senders = state.tcp_senders.lock().unwrap();
        for (wid, tx) in senders.iter() {
            if tx.send(msg.clone()).is_err() {
                dead.push(wid.clone());
            }
        }
        drop(senders);
        if !dead.is_empty() {
            let mut map = state.tcp_senders.lock().unwrap();
            for d in dead { map.remove(&d); }
        }

        Ok(Json(json!({"status":"started"})))
    }

    async fn get_dataset(
        State(state): State<ServerState>,
        Query(q): Query<HashMap<String,String>>,
    ) -> Result<Json<Value>, StatusCode> {
        let worker = q.get("worker_id").ok_or(StatusCode::BAD_REQUEST)?;
        let map = state.datasets.lock().unwrap();
        if let Some(ds) = map.get(worker) {
            let arr: Vec<Value> = ds.iter().map(|(x,y)| json!({"x": x, "y": y})).collect();
            return Ok(Json(Value::Array(arr)));
        }
        Ok(Json(Value::Array(vec![])))
    }

    async fn get_model(
        State(state): State<ServerState>,
        headers: HeaderMap,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;
        let guard = state.model.lock().unwrap();
        if let Some(m) = &*guard {
            return Ok(Json(serde_json::to_value(m).unwrap()));
        }
        Err(StatusCode::NO_CONTENT)
    }

    async fn get_training_params(
        State(state): State<ServerState>
    ) -> Result<Json<Value>, StatusCode> {
        let p = state.params.lock().unwrap();
        Ok(Json(json!({"stop_loss": p.stop_loss, "max_epochs": p.max_epochs})))
    }

    async fn sync_status(State(state): State<ServerState>) -> Result<Json<Value>, StatusCode> {
        let map = state.worker_sync.lock().unwrap();
        let mut out = serde_json::Map::new();
        for (k, v) in map.iter() {
            out.insert(k.clone(), json!({"updates": v}));
        }
        Ok(Json(Value::Object(out)))
    }

    async fn download_model(State(state): State<ServerState>) -> (StatusCode, (axum::http::HeaderMap, String)) {
        let guard = state.model.lock().unwrap();
        if guard.is_none() {
            return (StatusCode::BAD_REQUEST, (axum::http::HeaderMap::new(), "no model".into()));
        }
        let body = serde_json::to_string_pretty(guard.as_ref().unwrap()).unwrap();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("Content-Type", axum::http::HeaderValue::from_static("application/json"));
        headers.insert("Content-Disposition", axum::http::HeaderValue::from_static("attachment; filename=model.json"));
        (StatusCode::OK, (headers, body))
    }
}
