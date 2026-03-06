use anyhow::Result;
use reqwest::Client;
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpStream, UdpSocket},
    time::{sleep, Duration},
};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{messages::TcpMessage, model::DynamicModel};

fn default_broadcast_addr() -> &'static str {
    "255.255.255.255:9999"
}

pub async fn run_client(id: &str, tcp_addr: &str, http_addr: &str) -> Result<()> {
    println!("Worker `{}` starting. TCP={} HTTP={}", id, tcp_addr, http_addr);

    let api_key = std::env::var("API_KEY").expect("API_KEY missing");
    let http = Client::new();

    // ------------------------------------------------------------------
    // UDP auto-discovery (only when tcp_addr is empty)
    // ------------------------------------------------------------------
    let mut tcp_target = tcp_addr.to_string();
    if tcp_target.is_empty() {
        println!("Attempting UDP discovery...");
        let sock = UdpSocket::bind("0.0.0.0:0").await?;
        sock.set_broadcast(true)?;
        let broadcast = std::env::var("BROADCAST_ADDR")
            .unwrap_or_else(|_| default_broadcast_addr().to_string());
        let _ = sock.send_to(b"DISCOVER_SWARM", &broadcast).await;
        let mut buf = [0u8; 1024];
        match tokio::time::timeout(Duration::from_secs(2), sock.recv_from(&mut buf)).await {
            Ok(Ok((n, _src))) => {
                let resp = String::from_utf8_lossy(&buf[..n]).to_string();
                if let Ok(parsed) = serde_json::from_str::<Value>(&resp) {
                    if let Some(t) = parsed.get("tcp").and_then(|v| v.as_str()) {
                        tcp_target = t.to_string();
                        println!("Discovered TCP server: {}", tcp_target);
                    }
                }
            }
            _ => {
                println!("UDP discovery failed/timeout — using default TCP addr");
            }
        }
    }

    // ------------------------------------------------------------------
    // Outer reconnect loop
    // ------------------------------------------------------------------
    loop {
        println!("Connecting to TCP {} ...", tcp_target);

        match TcpStream::connect(&tcp_target).await {
            Ok(stream) => {
                println!("Connected to server at {}", tcp_target);

                let (reader_half, writer_half) = stream.into_split();
                let reader = BufReader::new(reader_half);
                let writer = Arc::new(Mutex::new(writer_half));

                // -----------------------------------------------------------
                // Register — triggers the server to push the current model
                // -----------------------------------------------------------
                {
                    let reg = TcpMessage::RequestModel { worker_id: id.to_string() };
                    let msg = serde_json::to_string(&reg)? + "\n";
                    writer.lock().await.write_all(msg.as_bytes()).await?;
                }

                // -----------------------------------------------------------
                // Heartbeat task (runs until the writer channel closes)
                // -----------------------------------------------------------
                {
                    let hb_writer = writer.clone();
                    let wid = id.to_string();
                    tokio::spawn(async move {
                        loop {
                            let hb = TcpMessage::Heartbeat { worker_id: wid.clone() };
                            let msg = serde_json::to_string(&hb).unwrap() + "\n";
                            let mut w = hb_writer.lock().await;
                            if w.write_all(msg.as_bytes()).await.is_err() {
                                println!("Heartbeat failed; exiting hb task");
                                break;
                            }
                            drop(w);
                            sleep(Duration::from_secs(2)).await;
                        }
                    });
                }

                let mut local_model: Option<DynamicModel> = None;
                let mut reader = reader;

                // -----------------------------------------------------------
                // Per-session training loop
                // -----------------------------------------------------------
                'session: loop {
                    // -------------------------------------------------------
                    // 1. Fetch dataset shard — retries until data arrives.
                    //    The dataset must have been uploaded by the pipeline
                    //    controller via POST /upload_dataset before training
                    //    can start.
                    // -------------------------------------------------------
                    println!("[{}] Waiting for dataset shard...", id);
                    let dataset_url = format!("{}/get_dataset?worker_id={}", http_addr, id);
                    let (xs, ys) = loop {
                        match http.get(&dataset_url)
                            .header("x-api-key", api_key.clone())
                            .send().await
                        {
                            Ok(resp) if resp.status().is_success() => {
                                let val: Value = resp.json().await.unwrap_or(Value::Null);
                                if let Some(arr) = val.as_array() {
                                    if !arr.is_empty() {
                                        let mut xs: Vec<Vec<f32>> = vec![];
                                        let mut ys: Vec<Vec<f32>> = vec![];
                                        for s in arr {
                                            xs.push(
                                                s["x"].as_array().unwrap_or(&vec![])
                                                    .iter().map(|v| v.as_f64().unwrap_or(0.0) as f32)
                                                    .collect()
                                            );
                                            ys.push(
                                                s["y"].as_array().unwrap_or(&vec![])
                                                    .iter().map(|v| v.as_f64().unwrap_or(0.0) as f32)
                                                    .collect()
                                            );
                                        }
                                        break (xs, ys);
                                    }
                                }
                            }
                            _ => {}
                        }
                        println!("[{}] No dataset yet, retrying in 1s...", id);
                        sleep(Duration::from_secs(1)).await;
                    };
                    println!("[{}] Got {} samples", id, xs.len());

                    // -------------------------------------------------------
                    // 2. Wait for Start signal from server.
                    //    Also accept a Model push that may arrive before Start.
                    // -------------------------------------------------------
                    println!("[{}] Waiting for Start signal...", id);
                    'wait_start: loop {
                        let mut raw = String::new();
                        let n = reader.read_line(&mut raw).await?;
                        if n == 0 {
                            println!("[{}] TCP closed by server", id);
                            break 'session;
                        }
                        let trimmed = raw.trim();
                        if trimmed.is_empty() { continue; }

                        match serde_json::from_str::<TcpMessage>(trimmed) {
                            Ok(TcpMessage::Start) => {
                                println!("[{}] Start received", id);
                                break 'wait_start;
                            }
                            Ok(TcpMessage::Model { model }) => {
                                println!("[{}] Model pushed from server (pre-start)", id);
                                local_model = Some(model);
                            }
                            Ok(TcpMessage::Heartbeat { .. }) => {}
                            Ok(_) => {}
                            Err(e) => {
                                println!("[{}] TCP parse error: {} | raw: {}", id, e, trimmed);
                            }
                        }
                    }

                    // -------------------------------------------------------
                    // 3. Ensure we have a model (HTTP fallback).
                    // -------------------------------------------------------
                    if local_model.is_none() {
                        println!("[{}] Fetching initial model via HTTP...", id);
                        match http.get(format!("{}/get_model", http_addr))
                            .header("x-api-key", api_key.clone())
                            .send().await
                        {
                            Ok(r) if r.status().is_success() => {
                                if let Ok(m) = r.json::<DynamicModel>().await {
                                    local_model = Some(m);
                                }
                            }
                            _ => {
                                println!("[{}] Failed to fetch model; retrying session", id);
                                continue;
                            }
                        }
                    }

                    let mut model = local_model.take().unwrap();

                    // -------------------------------------------------------
                    // 4. Fetch training hyper-params
                    // -------------------------------------------------------
                    let params = http
                        .get(format!("{}/get_training_params", http_addr))
                        .header("x-api-key", api_key.clone())
                        .send().await?
                        .json::<Value>().await?;

                    let stop_loss  = params["stop_loss"].as_f64().unwrap_or(0.01) as f32;
                    let max_rounds = params["max_epochs"].as_u64().unwrap_or(20) as usize;
                    let lr_log     = params["learning_rate"].as_f64().unwrap_or(0.01) as f32;

                    println!(
                        "[{}] Training: max_rounds={} stop_loss={} lr={} (applied server-side)",
                        id, max_rounds, stop_loss, lr_log
                    );

                    // -------------------------------------------------------
                    // 5. Data-parallel training loop
                    //
                    //    Each round:
                    //      a) Compute gradients over the full local shard
                    //         (no weight update here).
                    //      b) Send GradUpdate to server.
                    //      c) Wait for server to broadcast the aggregated Model
                    //         (server waits until ALL expected workers submit).
                    //      d) Replace local weights with the fused server model.
                    //      e) Check early-stop criterion on the new weights.
                    // -------------------------------------------------------
                    let mut training_done = false;

                    'rounds: for round in 1..=max_rounds {
                        // a) Gradient computation (no weight mutation)
                        let (dw, db, loss) = model.compute_gradients(&xs, &ys);
                        println!("[{}] round {}/{} local_loss={:.6}", id, round, max_rounds, loss);

                        // b) Send gradient update to server
                        let upd = TcpMessage::GradUpdate {
                            worker_id: id.to_string(),
                            round: round as u64,
                            dw,
                            db,
                            n_samples: xs.len(),
                        };
                        let msg = serde_json::to_string(&upd)? + "\n";
                        {
                            let mut w = writer.lock().await;
                            if w.write_all(msg.as_bytes()).await.is_err() {
                                println!("[{}] Failed to send GradUpdate; reconnecting", id);
                                break 'rounds;
                            }
                        }

                        // c) Wait for aggregated model broadcast
                        //    (server only sends this once all expected workers
                        //     for this round have submitted their GradUpdate)
                        loop {
                            let mut raw = String::new();
                            let n = reader.read_line(&mut raw).await?;
                            if n == 0 {
                                println!("[{}] TCP closed while waiting for aggregated model", id);
                                break 'rounds;
                            }
                            let trimmed = raw.trim();
                            if trimmed.is_empty() { continue; }

                            match serde_json::from_str::<TcpMessage>(trimmed) {
                                Ok(TcpMessage::Model { model: updated }) => {
                                    // d) Replace local weights with server-fused model
                                    model = updated;
                                    println!("[{}] Received aggregated model for round {}", id, round);
                                    break;
                                }
                                Ok(TcpMessage::Heartbeat { .. }) => {}
                                Ok(_) => {}
                                Err(e) => {
                                    println!("[{}] TCP parse error: {} | raw: {}", id, e, trimmed);
                                }
                            }
                        }

                        // e) Early-stop check on the now-fused model
                        let (_, _, updated_loss) = model.compute_gradients(&xs, &ys);
                        if updated_loss <= stop_loss {
                            println!(
                                "[{}] Early stop: loss {:.6} <= {} after round {}",
                                id, updated_loss, stop_loss, round
                            );
                            training_done = true;
                            break 'rounds;
                        }

                        // Small backoff to avoid sender flooding
                        sleep(Duration::from_millis(50)).await;
                    }

                    if training_done {
                        println!("[{}] Training converged. Waiting for next Start...", id);
                    } else {
                        println!("[{}] Reached max_rounds. Waiting for next Start...", id);
                    }

                    // Carry the trained model into the next session
                    local_model = Some(model);
                }

                // TCP closed — brief pause, then reconnect
                println!("[{}] Reconnecting in 2s...", id);
                sleep(Duration::from_secs(2)).await;
            }
            Err(e) => {
                println!("[{}] Connect failed: {} — retrying in 2s", id, e);
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}
