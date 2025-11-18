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

fn default_broadcast_for_subnet() -> &'static str {
    // adjust if you want auto-detection; for your subnet from earlier: 10.0.15.255
    // We keep this simple; user should set UDP_ADDR in project.env and server should respond with tcp addr.
    "255.255.255.255:9999"
}

pub async fn run_client(id: &str, tcp_addr: &str, http_addr: &str) -> Result<()> {
    println!("🤖 Worker `{}` starting. TCP={} HTTP={}", id, tcp_addr, http_addr);

    let api_key = std::env::var("API_KEY").expect("API_KEY missing");
    let http = Client::new();

    // If tcp_addr is empty, try UDP discovery
    let mut tcp_target = tcp_addr.to_string();
    if tcp_target.is_empty() {
        println!("🔍 Attempting UDP discovery...");
        let sock = UdpSocket::bind("0.0.0.0:0").await?;
        sock.set_broadcast(true)?;
        let broadcast = std::env::var("BROADCAST_ADDR").unwrap_or_else(|_| default_broadcast_for_subnet().to_string());
        let _ = sock.send_to(b"DISCOVER_SWARM", &broadcast).await;
        let mut buf = [0u8; 1024];
        match tokio::time::timeout(Duration::from_secs(2), sock.recv_from(&mut buf)).await {
            Ok(Ok((n, _src))) => {
                let resp = String::from_utf8_lossy(&buf[..n]).to_string();
                if let Ok(parsed) = serde_json::from_str::<Value>(&resp) {
                    if let Some(t) = parsed.get("tcp").and_then(|v| v.as_str()) {
                        tcp_target = t.to_string();
                        println!("🔗 Discovered TCP server: {}", tcp_target);
                    }
                }
            }
            _ => {
                println!("❌ UDP discovery failed/timeout");
            }
        }
    }

    loop {
        println!("🔌 Connecting to TCP {} ...", tcp_target);

        match TcpStream::connect(&tcp_target).await {
            Ok(stream) => {
                println!("🟢 Connected to server at {}", tcp_target);

                let (reader_half, writer_half) = stream.into_split();
                let reader = BufReader::new(reader_half);
                let writer = Arc::new(Mutex::new(writer_half));

                // register
                {
                    let reg = TcpMessage::RequestModel { worker_id: id.to_string() };
                    let msg = serde_json::to_string(&reg)? + "\n";
                    let mut w = writer.lock().await;
                    w.write_all(msg.as_bytes()).await?;
                }

                // heartbeat task
                {
                    let hb_writer = writer.clone();
                    let wid = id.to_string();
                    tokio::spawn(async move {
                        loop {
                            let hb = TcpMessage::Heartbeat { worker_id: wid.clone() };
                            let msg = serde_json::to_string(&hb).unwrap() + "\n";
                            let mut w = hb_writer.lock().await;
                            if w.write_all(msg.as_bytes()).await.is_err() {
                                println!("💔 Heartbeat failed; exiting hb task");
                                break;
                            }
                            drop(w);
                            sleep(Duration::from_secs(2)).await;
                        }
                    });
                }

                let mut local_model: Option<DynamicModel> = None;
                let mut reader = reader;

                // outer training/reception loop
                loop {
                    println!("⏳ Waiting for dataset for {}", id);
                    let dataset_url = format!("{}/get_dataset?worker_id={}", http_addr, id);
                    let (xs, ys) = loop {
                        match http.get(&dataset_url).header("x-api-key", api_key.clone()).send().await {
                            Ok(resp) if resp.status().is_success() => {
                                let val: Value = resp.json().await.unwrap_or(Value::Null);
                                if let Some(arr) = val.as_array() {
                                    if !arr.is_empty() {
                                        let mut xs = vec![];
                                        let mut ys = vec![];
                                        for s in arr {
                                            xs.push(s["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>());
                                            ys.push(s["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>());
                                        }
                                        break (xs, ys);
                                    }
                                }
                            }
                            _ => {}
                        }
                        println!("⏳ No dataset yet, retrying...");
                        sleep(Duration::from_secs(1)).await;
                    };

                    println!("⏳ Waiting for Start or Model push...");
                    loop {
                        let mut raw = String::new();
                        let n = reader.read_line(&mut raw).await?;
                        if n == 0 {
                            println!("❌ TCP closed by server");
                            break;
                        }
                        let trimmed = raw.trim();
                        if trimmed.is_empty() { continue; }
                        match serde_json::from_str::<TcpMessage>(trimmed) {
                            Ok(TcpMessage::Start) => {
                                println!("🚀 Start received");
                                break;
                            }
                            Ok(TcpMessage::Model { model }) => {
                                println!("📦 Model pushed from server");
                                local_model = Some(model);
                            }
                            Ok(TcpMessage::Heartbeat { .. }) => {}
                            Ok(_) => {}
                            Err(e) => {
                                println!("⚠ TCP parse error: {} | raw: {}", e, trimmed);
                            }
                        }
                    }

                    // get model if none
                    if local_model.is_none() {
                        println!("⬇ Fetching model via HTTP...");
                        match http.get(format!("{}/get_model", http_addr)).header("x-api-key", api_key.clone()).send().await {
                            Ok(r) if r.status().is_success() => {
                                if let Ok(m) = r.json::<DynamicModel>().await {
                                    local_model = Some(m);
                                }
                            }
                            _ => {
                                println!("❌ Failed to fetch model");
                                continue;
                            }
                        }
                    }

                    let mut model = local_model.take().unwrap();

                    // fetch params
                    let params = http.get(format!("{}/get_training_params", http_addr)).header("x-api-key", api_key.clone()).send().await?.json::<Value>().await?;
                    let stop_loss = params["stop_loss"].as_f64().unwrap_or(0.01) as f32;
                    let max_epochs = params["max_epochs"].as_u64().unwrap_or(20) as usize;

                    println!("🎯 Training loop starting");
                    for epoch in 1..=max_epochs {
                        let loss = model.train_step(&xs, &ys, 0.01);
                        println!("[{}] epoch {}/{} loss={:.6}", id, epoch, max_epochs, loss);

                        // send update
                        let upd = TcpMessage::ModelUpdate { worker_id: id.to_string(), round: epoch as u64, model: model.clone() };
                        let msg = serde_json::to_string(&upd)? + "\n";
                        let mut w = writer.lock().await;
                        if w.write_all(msg.as_bytes()).await.is_err() {
                            println!("❌ Failed to send update; connection likely closed");
                            break;
                        }
                        drop(w);

                        if loss <= stop_loss {
                            println!("🏁 Early stop: loss {:.6} <= {}", loss, stop_loss);
                            break;
                        }
                        sleep(Duration::from_millis(300)).await;
                    }

                    println!("🔄 Training iteration complete");
                    // loop to wait for next start
                }
                // if we reach here TCP closed — reconnect
                println!("🔁 Reconnecting to server in 2s...");
                sleep(Duration::from_secs(2)).await;
            }
            Err(e) => {
                println!("❌ Connect failed: {} — retrying in 2s", e);
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}
