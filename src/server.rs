use anyhow::Result;
use axum::{
    extract::{Query, State, ws::WebSocketUpgrade, ws::Message, ws::WebSocket},
    http::{HeaderMap, Method, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::time::{sleep, Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream, UdpSocket},
    sync::{broadcast, mpsc::unbounded_channel, mpsc::UnboundedSender},
};
use tower_http::cors::{Any, CorsLayer};
use chrono;
use crate::{messages::TcpMessage, model::DynamicModel};

// ---------------------------------------------------------------------------
// Training hyperparameters
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TrainingParams {
    pub stop_loss: f32,
    pub max_epochs: usize,
    pub learning_rate: f32,
}

// ---------------------------------------------------------------------------
// Worker bookkeeping
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum WorkerState {
    Idle,
    Training,
}

#[derive(Clone, Debug)]
pub struct WorkerInfo {
    pub last_seen: u64,
    pub updates: usize,
    pub addr: Option<String>,
    pub state: WorkerState,
}

// ---------------------------------------------------------------------------
// Pending gradient entry: one per worker per round
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct PendingGrad {
    dw: Vec<Vec<f32>>,
    db: Vec<Vec<f32>>,
    n_samples: usize,
}

// ---------------------------------------------------------------------------
// Shared server state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ServerState {
    pub model: Arc<Mutex<Option<DynamicModel>>>,
    pub model_versions: Arc<Mutex<Vec<(u64, DynamicModel, String)>>>, // (ver, model, ts)
    pub next_model_version: Arc<Mutex<u64>>,
    pub datasets: Arc<Mutex<HashMap<String, Vec<(Vec<f32>, Vec<f32>)>>>>,
    pub worker_sync: Arc<Mutex<HashMap<String, WorkerInfo>>>,
    pub params: Arc<Mutex<TrainingParams>>,
    pub tcp_senders: Arc<Mutex<HashMap<String, UnboundedSender<String>>>>,
    pub api_key: String,
    pub broadcaster: broadcast::Sender<Value>,
    pub http_addr: String,
    pub tcp_addr: String,

    // Data-parallel barrier: collects one GradUpdate per worker per round.
    // Key = worker_id, Value = (dw, db, n_samples).
    // When all workers in `round_expected_workers` have submitted → aggregate.
    pub pending_grads: Arc<Mutex<HashMap<String, PendingGrad>>>,

    // Snapshot of workers expected to participate in the *current* round.
    // Set atomically when /start_training is called so late-joiners don't
    // prevent the barrier from ever closing.
    pub round_expected_workers: Arc<Mutex<HashSet<String>>>,
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

pub struct Server;

impl Server {
    pub async fn run(
        tcp_addr: &str,
        http_addr: &str,
        udp_addr: &str,
        api_key: String,
    ) -> Result<()> {
        println!("\n====== FEDERATED SERVER STARTED ======");
        println!("HTTP: {}", http_addr);
        println!("TCP : {}", tcp_addr);
        println!("UDP : {}", udp_addr);
        println!("API : {}", api_key);
        println!("=====================================");

        let (tx, _rx) = broadcast::channel(128);

        let state = ServerState {
            model: Arc::new(Mutex::new(None)),
            model_versions: Arc::new(Mutex::new(Vec::new())),
            next_model_version: Arc::new(Mutex::new(1)),
            datasets: Arc::new(Mutex::new(HashMap::new())),
            worker_sync: Arc::new(Mutex::new(HashMap::new())),
            params: Arc::new(Mutex::new(TrainingParams {
                stop_loss: 0.01,
                max_epochs: 20,
                learning_rate: 0.01,
            })),
            tcp_senders: Arc::new(Mutex::new(HashMap::new())),
            api_key,
            broadcaster: tx,
            http_addr: http_addr.to_string(),
            tcp_addr: tcp_addr.to_string(),
            pending_grads: Arc::new(Mutex::new(HashMap::new())),
            round_expected_workers: Arc::new(Mutex::new(HashSet::new())),
        };

        // spawn TCP accept loop
        let tcp_addr_owned = tcp_addr.to_string();
        let tcp_state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::tcp_accept_loop(&tcp_addr_owned, tcp_state).await {
                eprintln!("TCP loop failed: {:?}", e);
            }
        });

        // spawn UDP discovery responder
        let udp_state = state.clone();
        let udp_addr_owned = udp_addr.to_string();
        tokio::spawn(async move {
            if let Err(e) = Self::udp_discovery_responder(&udp_addr_owned, udp_state).await {
                eprintln!("UDP discovery failed: {:?}", e);
            }
        });

        // CORS: allow everything (development-friendly; restrict in production)
        let cors = CorsLayer::new()
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([header::CONTENT_TYPE, header::HeaderName::from_static("x-api-key")])
            .allow_origin(Any);

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
            .route("/ws", get(Self::ws_handler))
            .layer(cors)
            .with_state(state.clone());

        let addr: SocketAddr = http_addr.parse()?;
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("HTTP server listening on {}", addr);
        axum::serve(listener, router).await?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // UDP discovery
    // -----------------------------------------------------------------------

    async fn udp_discovery_responder(bind_addr: &str, state: ServerState) -> Result<()> {
        let sock = UdpSocket::bind(bind_addr).await?;
        let _ = sock.set_broadcast(true);
        println!("UDP discovery responder bound to {}", bind_addr);

        let mut buf = [0u8; 1024];

        loop {
            match sock.recv_from(&mut buf).await {
                Ok((n, src)) => {
                    let req = String::from_utf8_lossy(&buf[..n]).trim().to_string();
                    if req == "DISCOVER_SWARM" {
                        println!("UDP discovery ping from {}", src);
                        let info = json!({
                            "http": state.http_addr,
                            "tcp": state.tcp_addr,
                            "note": "swarm_server"
                        });
                        let _ = sock.send_to(info.to_string().as_bytes(), &src).await;
                    }
                }
                Err(e) => {
                    println!("UDP recv error: {}", e);
                }
            }
            sleep(Duration::from_millis(50)).await;
        }
    }

    // -----------------------------------------------------------------------
    // TCP accept loop
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // TCP connection handler
    // -----------------------------------------------------------------------

    async fn handle_tcp(stream: TcpStream, state: ServerState) -> Result<()> {
        let peer = stream.peer_addr()?.to_string();
        let (r, w) = stream.into_split();
        let mut reader = BufReader::new(r);

        let (tx, mut rx) = unbounded_channel::<String>();
        let mut writer = w;
        let mut worker_id_opt: Option<String> = None;

        // writer task — drains the mpsc channel into the TCP socket
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if writer.write_all(msg.as_bytes()).await.is_err() { break; }
                if writer.write_all(b"\n").await.is_err() { break; }
            }
        });

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
                // -- Registration --
                Ok(TcpMessage::RequestModel { worker_id }) => {
                    println!("TCP register: {}", worker_id);
                    worker_id_opt = Some(worker_id.clone());
                    state.tcp_senders.lock().unwrap().insert(worker_id.clone(), tx.clone());

                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    state.worker_sync.lock().unwrap()
                        .entry(worker_id.clone())
                        .or_insert(WorkerInfo {
                            last_seen: now,
                            updates: 0,
                            addr: Some(peer.clone()),
                            state: WorkerState::Idle,
                        });
                    state.datasets.lock().unwrap()
                        .entry(worker_id.clone())
                        .or_insert(vec![]);

                    let _ = state.broadcaster.send(
                        json!({"event": "worker_connect", "worker": worker_id.clone()})
                    );

                    // push current model if exists
                    if let Some(m) = &*state.model.lock().unwrap() {
                        let msg = TcpMessage::Model { model: m.clone() };
                        let s = serde_json::to_string(&msg).unwrap();
                        let _ = tx.send(s);
                    }
                }

                // -- Data-parallel gradient update (primary path) --
                Ok(TcpMessage::GradUpdate { worker_id, round, dw, db, n_samples }) => {
                    println!("GradUpdate from {} round {} n_samples={}", worker_id, round, n_samples);

                    // 1. Store this worker's gradient contribution
                    state.pending_grads.lock().unwrap().insert(
                        worker_id.clone(),
                        PendingGrad { dw, db, n_samples },
                    );

                    // 2. Update worker bookkeeping
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    state.worker_sync.lock().unwrap()
                        .entry(worker_id.clone())
                        .and_modify(|wi| { wi.updates += 1; wi.last_seen = now; wi.state = WorkerState::Training; })
                        .or_insert(WorkerInfo {
                            last_seen: now,
                            updates: 1,
                            addr: Some(peer.clone()),
                            state: WorkerState::Training,
                        });

                    // 3. Barrier check: have all *expected* workers for this round submitted?
                    //    We use round_expected_workers (set at start_training) so late-joining
                    //    workers don't hold the barrier open indefinitely.
                    let all_submitted = {
                        let grads = state.pending_grads.lock().unwrap();
                        let expected = state.round_expected_workers.lock().unwrap();
                        // If no snapshot yet (e.g. training started before snapshot was recorded),
                        // fall back to all currently-connected workers.
                        if expected.is_empty() {
                            let senders = state.tcp_senders.lock().unwrap();
                            !senders.is_empty() && senders.keys().all(|wid| grads.contains_key(wid))
                        } else {
                            expected.iter().all(|wid| grads.contains_key(wid))
                        }
                    };

                    if all_submitted {
                        println!("Barrier reached — aggregating gradients from all workers for round {}", round);

                        // 4. Weighted average of gradients
                        let grads_snapshot: Vec<(String, PendingGrad)> = {
                            state.pending_grads.lock().unwrap()
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        };

                        let total_n: usize = grads_snapshot.iter().map(|(_, g)| g.n_samples).sum();

                        if let Some((_, first)) = grads_snapshot.first() {
                            let n_layers = first.dw.len();

                            // Initialise accumulators
                            let mut avg_dw: Vec<Vec<f32>> =
                                first.dw.iter().map(|w| vec![0.0f32; w.len()]).collect();
                            let mut avg_db: Vec<Vec<f32>> =
                                first.db.iter().map(|b| vec![0.0f32; b.len()]).collect();

                            // Weighted sum
                            for (_, g) in &grads_snapshot {
                                let weight = g.n_samples as f32 / total_n as f32;
                                for li in 0..n_layers {
                                    for i in 0..avg_dw[li].len() {
                                        avg_dw[li][i] += weight * g.dw[li][i];
                                    }
                                    for i in 0..avg_db[li].len() {
                                        avg_db[li][i] += weight * g.db[li][i];
                                    }
                                }
                            }

                            // 5. Apply averaged gradient to global model
                            let lr = state.params.lock().unwrap().learning_rate;
                            let mut global = state.model.lock().unwrap();
                            if let Some(m) = &mut *global {
                                m.apply_gradients(&avg_dw, &avg_db, lr);
                            }

                            // 6. Save version
                            {
                                let g = global.as_ref().unwrap().clone();
                                let mut ver = state.next_model_version.lock().unwrap();
                                let v = *ver;
                                *ver += 1;
                                let ts = SystemTime::now()
                                    .duration_since(UNIX_EPOCH).unwrap().as_secs().to_string();
                                state.model_versions.lock().unwrap().push((v, g, ts));
                            }

                            // 7. Broadcast updated model to all workers
                            let broadcast_msg = {
                                global.as_ref().map(|m| {
                                    let msg = TcpMessage::Model { model: m.clone() };
                                    serde_json::to_string(&msg).unwrap()
                                })
                            };
                            drop(global);

                            if let Some(s) = broadcast_msg {
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
                                    for d in &dead {
                                        map.remove(d);
                                        println!("Removed dead worker tcp: {}", d);
                                        let _ = state.broadcaster.send(
                                            json!({"event": "worker_dead", "worker": d})
                                        );
                                    }
                                }
                            }

                            // 8. Clear barrier for next round
                            state.pending_grads.lock().unwrap().clear();

                            let _ = state.broadcaster.send(json!({
                                "event": "model_updated",
                                "round": round,
                                "n_workers": grads_snapshot.len(),
                                "total_samples": total_n,
                            }));

                            // Update worker states back to Training (still in training loop)
                            {
                                let now2 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                                let mut ws = state.worker_sync.lock().unwrap();
                                for (wid, _) in &grads_snapshot {
                                    if let Some(wi) = ws.get_mut(wid) {
                                        wi.last_seen = now2;
                                        wi.state = WorkerState::Training;
                                    }
                                }
                            }
                        }
                    }
                }

                // -- Legacy: full weight update (federated averaging mode) --
                Ok(TcpMessage::ModelUpdate { worker_id, round, model }) => {
                    println!("ModelUpdate (legacy fedavg) from {} round {}", worker_id, round);

                    {
                        let mut global = state.model.lock().unwrap();
                        if let Some(g) = &mut *global {
                            g.merge_inplace(&model, 0.5);
                        } else {
                            *global = Some(model.clone());
                        }

                        let mut ver_guard = state.next_model_version.lock().unwrap();
                        let version = *ver_guard;
                        *ver_guard += 1;
                        let ts = SystemTime::now()
                            .duration_since(UNIX_EPOCH).unwrap().as_secs().to_string();
                        state.model_versions.lock().unwrap()
                            .push((version, model.clone(), ts));
                    }

                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    state.worker_sync.lock().unwrap()
                        .entry(worker_id.clone())
                        .and_modify(|wi| { wi.updates += 1; wi.last_seen = now; wi.state = WorkerState::Idle; })
                        .or_insert(WorkerInfo {
                            last_seen: now, updates: 1,
                            addr: Some(peer.clone()), state: WorkerState::Idle,
                        });

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
                                if sender.send(s.clone()).is_err() { dead.push(wid.clone()); }
                            }
                        }
                        if !dead.is_empty() {
                            let mut map = state.tcp_senders.lock().unwrap();
                            for d in dead {
                                map.remove(&d);
                                println!("Removed dead tcp {}", d);
                                let _ = state.broadcaster.send(
                                    json!({"event": "worker_dead", "worker": d.clone()})
                                );
                            }
                        }
                    }

                    let _ = state.broadcaster.send(json!({"event": "model_updated"}));
                }

                // -- Heartbeat --
                Ok(TcpMessage::Heartbeat { worker_id }) => {
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    state.worker_sync.lock().unwrap()
                        .entry(worker_id.clone())
                        .and_modify(|wi| { wi.last_seen = now; })
                        .or_insert(WorkerInfo {
                            last_seen: now, updates: 0,
                            addr: Some(peer.clone()), state: WorkerState::Idle,
                        });
                    let _ = state.broadcaster.send(
                        json!({"event": "heartbeat", "worker": worker_id})
                    );
                }

                Ok(_) => {}
                Err(e) => {
                    println!("TCP parse error: {} | raw: {}", e, raw);
                }
            }
        }

        // cleanup
        if let Some(wid) = worker_id_opt {
            state.tcp_senders.lock().unwrap().remove(&wid);
            state.worker_sync.lock().unwrap().remove(&wid);
            // Remove dangling grad so barrier isn't stuck if worker disconnects mid-round
            state.pending_grads.lock().unwrap().remove(&wid);
            // Remove from expected set so barrier can still close
            state.round_expected_workers.lock().unwrap().remove(&wid);
            let _ = state.broadcaster.send(
                json!({"event": "worker_disconnect", "worker": wid})
            );
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // HTTP endpoints
    // -----------------------------------------------------------------------

    fn check_api(headers: &HeaderMap, expected: &str) -> Result<(), StatusCode> {
        if expected.is_empty() { return Ok(()); }
        let got = headers.get("x-api-key").and_then(|v| v.to_str().ok());
        match got {
            Some(k) if k == expected => Ok(()),
            _ => {
                println!("Unauthorized (got={:?})", got);
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

        let input_dim  = payload.get("input_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
        let hidden     = payload.get("hidden_layers").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let output_dim = payload.get("output_dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let mut layers = vec![input_dim];
        for h in hidden { if let Some(n) = h.as_u64() { layers.push(n as usize); } }
        layers.push(output_dim);

        let model = DynamicModel::new(layers.clone());
        *state.model.lock().unwrap() = Some(model.clone());

        let mut ver_guard = state.next_model_version.lock().unwrap();
        let version = *ver_guard;
        *ver_guard += 1;
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string();
        state.model_versions.lock().unwrap().push((version, model.clone(), ts));

        let _ = state.broadcaster.send(json!({"event": "model_created", "version": version}));
        println!("Model created: {:?} (version {})", layers, version);
        Ok(Json(json!({"status": "ok", "layer_sizes": layers, "version": version})))
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
            let arr_x = sample.get("x").and_then(|v| v.as_array()).ok_or(StatusCode::BAD_REQUEST)?;
            let arr_y = sample.get("y").and_then(|v| v.as_array()).ok_or(StatusCode::BAD_REQUEST)?;
            let x = arr_x.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
            let y = arr_y.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
            parsed.push((x, y));
        }

        state.datasets.lock().unwrap().insert(worker_id.clone(), parsed);
        println!("Dataset uploaded for {}", worker_id);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        if let Some(wi) = state.worker_sync.lock().unwrap().get_mut(&worker_id) {
            wi.last_seen = now;
        }

        Ok(Json(json!({"status": "ok"})))
    }

    async fn register_worker(
        State(state): State<ServerState>,
        headers: HeaderMap,
        Json(payload): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;
        let worker_id = payload["worker_id"].as_str().ok_or(StatusCode::BAD_REQUEST)?.to_string();

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        state.datasets.lock().unwrap().entry(worker_id.clone()).or_insert(vec![]);
        state.worker_sync.lock().unwrap().entry(worker_id.clone()).or_insert(WorkerInfo {
            last_seen: now,
            updates: 0usize,
            addr: None,
            state: WorkerState::Idle,
        });

        println!("Worker registered via HTTP: {}", worker_id);
        let _ = state.broadcaster.send(
            json!({"event": "worker_register", "worker": worker_id.clone()})
        );
        Ok(Json(json!({"status": "registered", "worker_id": worker_id})))
    }

    async fn set_training_params(
        State(state): State<ServerState>,
        headers: HeaderMap,
        Json(payload): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;
        let mut p = state.params.lock().unwrap();
        p.stop_loss     = payload.get("stop_loss").and_then(|v| v.as_f64()).unwrap_or(0.01) as f32;
        p.max_epochs    = payload.get("max_epochs").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        p.learning_rate = payload.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.01) as f32;
        println!("Updated training params stop_loss={} max_epochs={} lr={}",
            p.stop_loss, p.max_epochs, p.learning_rate);
        Ok(Json(json!({"status": "ok"})))
    }

    async fn start_training(
        State(state): State<ServerState>,
        headers: HeaderMap,
    ) -> Result<Json<Value>, StatusCode> {
        Self::check_api(&headers, &state.api_key)?;

        if state.model.lock().unwrap().is_none() {
            return Err(StatusCode::BAD_REQUEST);
        }

        println!("Broadcasting START to all workers");

        // Snapshot the set of workers we expect to train this round.
        // This prevents late-joiners from holding the barrier open.
        let start_msg = serde_json::to_string(&TcpMessage::Start).unwrap();
        let mut dead: Vec<String> = Vec::new();
        let mut active_workers: HashSet<String> = HashSet::new();

        {
            let senders = state.tcp_senders.lock().unwrap();
            let mut worker_meta = state.worker_sync.lock().unwrap();

            for (wid, tx) in senders.iter() {
                if tx.send(start_msg.clone()).is_err() {
                    dead.push(wid.clone());
                } else {
                    active_workers.insert(wid.clone());
                    if let Some(meta) = worker_meta.get_mut(wid) {
                        meta.state = WorkerState::Training;
                        meta.last_seen = chrono::Utc::now().timestamp() as u64;
                    }
                }
            }
        }

        // Store the round snapshot
        *state.round_expected_workers.lock().unwrap() = active_workers.clone();

        if !dead.is_empty() {
            let mut map = state.tcp_senders.lock().unwrap();
            for d in &dead {
                map.remove(d);
                state.worker_sync.lock().unwrap().remove(d);
                println!("Removed dead worker: {}", d);
            }
        }

        // Clear any stale pending grads from a previous run
        state.pending_grads.lock().unwrap().clear();

        let _ = state.broadcaster.send(json!({
            "event": "training_started",
            "expected_workers": active_workers.len(),
        }));
        Ok(Json(json!({"status": "started", "workers": active_workers.len()})))
    }

    async fn get_dataset(
        State(state): State<ServerState>,
        headers: HeaderMap,
        Query(q): Query<HashMap<String, String>>,
    ) -> Result<Json<Value>, StatusCode> {
        // Require API key so arbitrary clients can't read shard data
        Self::check_api(&headers, &state.api_key)?;
        let worker = q.get("worker_id").ok_or(StatusCode::BAD_REQUEST)?;
        let map = state.datasets.lock().unwrap();
        if let Some(ds) = map.get(worker) {
            let arr: Vec<Value> = ds.iter().map(|(x, y)| json!({"x": x, "y": y})).collect();
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
        State(state): State<ServerState>,
    ) -> Result<Json<Value>, StatusCode> {
        let p = state.params.lock().unwrap();
        Ok(Json(json!({
            "stop_loss": p.stop_loss,
            "max_epochs": p.max_epochs,
            "learning_rate": p.learning_rate,
        })))
    }

    async fn sync_status(State(state): State<ServerState>) -> Result<Json<Value>, StatusCode> {
        let map = state.worker_sync.lock().unwrap();
        let pending = state.pending_grads.lock().unwrap();
        let expected = state.round_expected_workers.lock().unwrap();
        let mut workers = serde_json::Map::new();
        for (k, wi) in map.iter() {
            workers.insert(k.clone(), json!({
                "updates": wi.updates,
                "last_seen": wi.last_seen,
                "addr": wi.addr.clone().unwrap_or_default(),
                "state": match wi.state { WorkerState::Idle => "idle", WorkerState::Training => "training" },
                "grad_submitted": pending.contains_key(k),
                "in_current_round": expected.contains(k),
            }));
        }
        Ok(Json(json!({
            "workers": workers,
            "pending_grad_count": pending.len(),
            "expected_this_round": expected.len(),
        })))
    }

    async fn download_model(
        State(state): State<ServerState>,
        Query(q): Query<HashMap<String, String>>,
    ) -> (StatusCode, (axum::http::HeaderMap, String)) {
        let guard = state.model.lock().unwrap();
        if let Some(ver_str) = q.get("version") {
            if let Ok(ver) = ver_str.parse::<u64>() {
                let versions = state.model_versions.lock().unwrap();
                if let Some((_, m, _)) = versions.iter().find(|(v, _, _)| *v == ver) {
                    let body = serde_json::to_string_pretty(m).unwrap();
                    let mut headers = axum::http::HeaderMap::new();
                    headers.insert("Content-Type", axum::http::HeaderValue::from_static("application/json"));
                    headers.insert("Content-Disposition", axum::http::HeaderValue::from_static("attachment; filename=model.json"));
                    return (StatusCode::OK, (headers, body));
                } else {
                    return (StatusCode::BAD_REQUEST, (axum::http::HeaderMap::new(), "version not found".into()));
                }
            }
        }
        if guard.is_none() {
            return (StatusCode::BAD_REQUEST, (axum::http::HeaderMap::new(), "no model".into()));
        }
        let body = serde_json::to_string_pretty(guard.as_ref().unwrap()).unwrap();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("Content-Type", axum::http::HeaderValue::from_static("application/json"));
        headers.insert("Content-Disposition", axum::http::HeaderValue::from_static("attachment; filename=model.json"));
        (StatusCode::OK, (headers, body))
    }

    async fn ws_handler(
        ws: WebSocketUpgrade,
        State(state): State<ServerState>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |mut socket: WebSocket| async move {
            let mut rx = state.broadcaster.subscribe();

            let snapshot = {
                let map = state.worker_sync.lock().unwrap();
                let expected = state.round_expected_workers.lock().unwrap();
                let mut workers = serde_json::Map::new();
                for (k, wi) in map.iter() {
                    workers.insert(k.clone(), json!({
                        "updates": wi.updates,
                        "last_seen": wi.last_seen,
                        "addr": wi.addr.clone().unwrap_or_default(),
                        "state": match wi.state { WorkerState::Idle => "idle", WorkerState::Training => "training" },
                        "in_current_round": expected.contains(k),
                    }));
                }
                let versions = state.model_versions.lock().unwrap();
                let versions_meta: Vec<Value> = versions.iter()
                    .map(|(v, _, ts)| json!({"version": v, "ts": ts})).collect();
                json!({
                    "event": "snapshot",
                    "workers": workers,
                    "model_versions": versions_meta,
                    "expected_this_round": expected.len(),
                })
            };

            if let Ok(txt) = serde_json::to_string(&snapshot) {
                let _ = socket.send(Message::Text(txt.into())).await;
            }

            loop {
                tokio::select! {
                    res = rx.recv() => {
                        match res {
                            Ok(val) => {
                                if let Ok(txt) = serde_json::to_string(&val) {
                                    if socket.send(Message::Text(txt.into())).await.is_err() { break; }
                                }
                            }
                            Err(_) => break,
                        }
                    }

                    msg = socket.recv() => {
                        match msg {
                            Some(Ok(m)) => {
                                match m {
                                    Message::Ping(p) => { let _ = socket.send(Message::Pong(p)).await; }
                                    Message::Close(_) => { break; }
                                    _ => {}
                                }
                            }
                            Some(Err(_)) => break,
                            None => break,
                        }
                    }
                }
            }
        })
    }
}
