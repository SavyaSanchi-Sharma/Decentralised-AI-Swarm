// src/server.rs
use anyhow::Result;
use axum::{
    extract::{Query, State, ws::WebSocketUpgrade, ws::Message, ws::WebSocket},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
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
use crate::{messages::TcpMessage, model::DynamicModel};

#[derive(Clone)]
pub struct TrainingParams {
    pub stop_loss: f32,
    pub max_epochs: usize,
}

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
}

pub struct Server;

impl Server {
    pub async fn run(
    tcp_addr: &str,
    http_addr: &str,
    udp_addr: &str,
    api_key: String
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
        params: Arc::new(Mutex::new(TrainingParams { stop_loss: 0.01, max_epochs: 20 })),
        tcp_senders: Arc::new(Mutex::new(HashMap::new())),
        api_key,
        broadcaster: tx,
    };

    // -------------------------------
    // TCP Accept Loop
    // -------------------------------
    let tcp_addr_owned = tcp_addr.to_string();
    let tcp_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = Self::tcp_accept_loop(&tcp_addr_owned, tcp_state).await {
            eprintln!("TCP loop failed: {:?}", e);
        }
    });

    // -------------------------------
    // UDP Discovery
    // -------------------------------
    let udp_addr_owned = udp_addr.to_string();
    let udp_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = Self::udp_discovery_responder(&udp_addr_owned, udp_state).await {
            eprintln!("UDP discovery failed: {:?}", e);
        }
    });

    // -------------------------------
    // HTTP Router (AXUM)
    // -------------------------------
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
        .with_state(state.clone());

    // -------------------------------
    // Bind HTTP Server
    // -------------------------------
    let addr: SocketAddr = http_addr.parse()?;
    axum::Server::bind(&addr)
        .serve(router.into_make_service())
        .await?;

    Ok(())
}


async fn udp_discovery_responder(bind_addr: &str, _state: ServerState) -> Result<()> {
    let sock = UdpSocket::bind(bind_addr).await?;
    println!("UDP discovery responder bound to {}", bind_addr);

    let mut buf = [0u8; 1024];

    loop {
        match sock.recv_from(&mut buf).await {
            Ok((n, src)) => {
                let req = String::from_utf8_lossy(&buf[..n]).to_string();

                if req.trim() == "DISCOVER_SWARM" {
                    let info = json!({
                        "http": "available",
                        "note": "swarm_server"
                    })
                    .to_string();

                    let _ = sock.send_to(info.as_bytes(), &src).await;
                }
            }
            Err(e) => {
                println!("UDP recv error: {}", e);
            }
        }

        // small async sleep
        sleep(Duration::from_millis(100)).await;
    }
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

        let (tx, mut rx) = unbounded_channel::<String>();
        let mut writer = w;
        let mut worker_id_opt: Option<String> = None;

        // writer task
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let _ = writer.write_all(msg.as_bytes()).await;
                let _ = writer.write_all(b"\n").await;
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
                Ok(TcpMessage::RequestModel { worker_id }) => {
                    println!("TCP register: {}", worker_id);
                    worker_id_opt = Some(worker_id.clone());
                    state.tcp_senders.lock().unwrap().insert(worker_id.clone(), tx.clone());

                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    state.worker_sync.lock().unwrap().entry(worker_id.clone()).or_insert(WorkerInfo {
                        last_seen: now,
                        updates: 0,
                        addr: Some(peer.clone()),
                        state: WorkerState::Idle,
                    });
                    state.datasets.lock().unwrap().entry(worker_id.clone()).or_insert(vec![]);

                    let _ = state.broadcaster.send(json!({"type":"worker_connect","worker": worker_id.clone()}));

                    if let Some(m) = &*state.model.lock().unwrap() {
                        let msg = TcpMessage::Model { model: m.clone() };
                        let s = serde_json::to_string(&msg).unwrap();
                        let _ = tx.send(s);
                    }
                }
                Ok(TcpMessage::ModelUpdate { worker_id, round, model }) => {
                    println!("TCP update from {} round {}", worker_id, round);

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
                        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string();
                        state.model_versions.lock().unwrap().push((version, model.clone(), ts));
                    }

                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    state.worker_sync.lock().unwrap().entry(worker_id.clone())
                        .and_modify(|wi| { wi.updates += 1; wi.last_seen = now; wi.state = WorkerState::Idle; })
                        .or_insert(WorkerInfo { last_seen: now, updates:1, addr: None, state: WorkerState::Idle});

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
                                let _ = state.broadcaster.send(json!({"type":"worker_dead","worker": d.clone()}));
                            }
                        }
                    }

                    let _ = state.broadcaster.send(json!({"type":"model_updated"}));
                }
                Ok(TcpMessage::Heartbeat { worker_id }) => {
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    state.worker_sync.lock().unwrap().entry(worker_id.clone()).and_modify(|wi| { wi.last_seen = now; wi.state = WorkerState::Idle; })
                        .or_insert(WorkerInfo { last_seen: now, updates:0, addr: Some(peer.clone()), state: WorkerState::Idle });
                    let _ = state.broadcaster.send(json!({"type":"heartbeat","worker": worker_id}));
                }
                Ok(_) => {}
                Err(e) => {
                    println!("TCP parse error: {} | raw: {}", e, raw);
                }
            }
        }

        if let Some(wid) = worker_id_opt {
            state.tcp_senders.lock().unwrap().remove(&wid);
            state.worker_sync.lock().unwrap().remove(&wid);
            let _ = state.broadcaster.send(json!({"type":"worker_disconnect","worker": wid}));
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
                println!("⛔ Unauthorized (got={:?})", got);
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
        *state.model.lock().unwrap() = Some(model.clone());

        let mut ver_guard = state.next_model_version.lock().unwrap();
        let version = *ver_guard;
        *ver_guard += 1;
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string();
        state.model_versions.lock().unwrap().push((version, model.clone(), ts));

        let _ = state.broadcaster.send(json!({"type":"model_created","version": version}));

        println!("✔ Model created: {:?} (version {})", layers, version);
        Ok(Json(json!({"status":"ok","layer_sizes": layers, "version": version})))
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
        println!("📥 Valid dataset uploaded for {}", worker_id);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        if let Some( wi) = state.worker_sync.lock().unwrap().get_mut(&worker_id) {
            wi.last_seen = now;
        }

        Ok(Json(json!({"status":"ok"})))
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

        println!("🟢 Worker registered via HTTP: {}", worker_id);
        let _ = state.broadcaster.send(json!({"type":"worker_register","worker": worker_id.clone()}));
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
        // Block start if no model exists
        if state.model.lock().unwrap().is_none() {
            return Err(StatusCode::BAD_REQUEST);
        }

        println!("🚀 Broadcast START to workers");

        let msg = serde_json::to_string(&TcpMessage::Start).unwrap();
        let mut dead = Vec::new();
        let senders = state.tcp_senders.lock().unwrap();
        for (wid, tx) in senders.iter() {
            if tx.send(msg.clone()).is_err() {
                dead.push(wid.clone());
            } else {
                if let Some(mut wi) = state.worker_sync.lock().unwrap().get_mut(wid) {
                    wi.state = WorkerState::Training;
                }
            }
        }
        drop(senders);
        if !dead.is_empty() {
            let mut map = state.tcp_senders.lock().unwrap();
            for d in dead { map.remove(&d); }
        }

        let _ = state.broadcaster.send(json!({"type":"start"}));
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
        // No model → return 204 No Content
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
        let mut workers = serde_json::Map::new();
        for (k, wi) in map.iter() {
            workers.insert(k.clone(), json!({
                "updates": wi.updates,
                "last_seen": wi.last_seen,
                "addr": wi.addr.clone().unwrap_or_default(),
                "state": match wi.state { WorkerState::Idle => "idle", WorkerState::Training => "training" }
            }));
        }
        Ok(Json(json!({"workers": workers})))
    }

    async fn download_model(
        State(state): State<ServerState>,
        Query(q): Query<HashMap<String,String>>,
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

            // send initial snapshot
            let snapshot = {
                let map = state.worker_sync.lock().unwrap();
                let mut workers = serde_json::Map::new();
                for (k, wi) in map.iter() {
                    workers.insert(k.clone(), json!({
                        "updates": wi.updates,
                        "last_seen": wi.last_seen,
                        "addr": wi.addr.clone().unwrap_or_default(),
                        "state": match wi.state { WorkerState::Idle => "idle", WorkerState::Training => "training" }
                    }));
                }
                let versions = state.model_versions.lock().unwrap();
                let versions_meta: Vec<Value> = versions.iter().map(|(v, _, ts)| json!({"version": v, "ts": ts})).collect();
                json!({"type":"snapshot","workers": workers, "model_versions": versions_meta})
            };

            if let Ok(txt) = serde_json::to_string(&snapshot) {
                let _ = socket.send(Message::Text(txt)).await;
            }

            loop {
                tokio::select! {
                    res = rx.recv() => {
                        match res {
                            Ok(val) => {
                                if let Ok(txt) = serde_json::to_string(&val) {
                                    if socket.send(Message::Text(txt)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }

                    msg = socket.recv() => {
                        match msg {
                            Some(Ok( m)) => {
                                // handle incoming WS messages from client if needed
                                // for now, ignore text/ping etc. but respond to ping if present
                                match m {
                                    Message::Ping(p) => {
                                        let _ = socket.send(Message::Pong(p)).await;
                                    }
                                    Message::Close(_) => {
                                        break;
                                    }
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
