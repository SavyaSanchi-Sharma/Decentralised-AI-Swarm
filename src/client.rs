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

pub async fn run_client(id: &str, tcp_addr: &str, http_addr: &str) -> Result<()> {
    println!("🤖 Worker `{}` starting. TCP={} HTTP={}", id, tcp_addr, http_addr);

    let api_key = std::env::var("API_KEY").expect("API_KEY missing");
    let http = Client::new();

    // Optional: UDP discovery if tcp_addr = ""
    if tcp_addr.is_empty() {
        if let Ok(sock) = UdpSocket::bind("0.0.0.0:0").await {
            let _ = sock.send_to(b"DISCOVER_SWARM", "255.255.255.255:9999").await;
        }
    }

    // ===============================
    // OUTER LOOP — reconnect forever
    // ===============================
    loop {
        println!("🔌 Connecting to TCP {} ...", tcp_addr);

        match TcpStream::connect(tcp_addr).await {
            Ok(stream) => {
                println!("🟢 Connected!");

                let (reader_half, writer_half) = stream.into_split();

                let reader = BufReader::new(reader_half);

                // 📌 IMPORTANT: wrap writer in Arc<Mutex<...>>
                let writer = Arc::new(Mutex::new(writer_half));

                // ---------------------------
                // REGISTER WORKER
                // ---------------------------
                {
                    let reg = TcpMessage::RequestModel { worker_id: id.to_string() };
                    let msg = serde_json::to_string(&reg)? + "\n";
                    let mut w = writer.lock().await;
                    w.write_all(msg.as_bytes()).await?;
                }

                // ---------------------------
                // SPAWN HEARTBEAT TASK
                // ---------------------------
                {
                    let hb_writer = writer.clone();
                    let wid = id.to_string();
                    tokio::spawn(async move {
                        loop {
                            let hb = TcpMessage::Heartbeat { worker_id: wid.clone() };
                            let msg = serde_json::to_string(&hb).unwrap() + "\n";

                            let mut w = hb_writer.lock().await;
                            if w.write_all(msg.as_bytes()).await.is_err() {
                                println!("💔 Heartbeat failed; TCP closed.");
                                break;
                            }
                            drop(w);

                            sleep(Duration::from_secs(2)).await;
                        }
                    });
                }

                // Local model storage
                let mut local_model: Option<DynamicModel> = None;

                // Clone reader into mutable variable
                let mut reader = reader;

                // ==================================================
                // MAIN CLIENT LOOP (never exits until TCP closes)
                // ==================================================
                loop {
                    // ---------------------------
                    // 1. Wait until dataset available
                    // ---------------------------
                    println!("⏳ Waiting for dataset for {} ...", id);

                    let dataset_url = format!("{}/get_dataset?worker_id={}", http_addr, id);
                    let (xs, ys) = loop {
                        match http.get(&dataset_url)
                            .header("x-api-key", api_key.clone())
                            .send()
                            .await 
                        {
                            Ok(resp) if resp.status().is_success() => {
                                let val: Value = resp.json().await.unwrap_or(Value::Null);
                                if let Some(arr) = val.as_array() {
                                    if !arr.is_empty() {
                                        let mut xs = vec![];
                                        let mut ys = vec![];
                                        for s in arr {
                                            xs.push(s["x"]
                                                .as_array().unwrap()
                                                .iter().map(|v| v.as_f64().unwrap() as f32)
                                                .collect::<Vec<f32>>()
                                            );

                                            ys.push(s["y"]
                                                .as_array().unwrap()
                                                .iter().map(|v| v.as_f64().unwrap() as f32)
                                                .collect::<Vec<f32>>()
                                            );
                                        }
                                        println!("📥 Loaded {} samples.", xs.len());
                                        break (xs, ys);
                                    }
                                }
                            }
                            _ => {}
                        }
                        sleep(Duration::from_secs(1)).await;
                    };

                    // ---------------------------
                    // 2. WAIT FOR START OR MODEL PUSH
                    // ---------------------------
                    println!("⏳ Waiting for Start...");

                    loop {
                        let mut raw = String::new();
                        let n = reader.read_line(&mut raw).await?;

                        if n == 0 {
                            println!("❌ TCP connection closed by server.");
                            break;
                        }

                        let trimmed = raw.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<TcpMessage>(trimmed) {
                            Ok(TcpMessage::Start) => {
                                println!("🚀 Start received!");
                                break;
                            }
                            Ok(TcpMessage::Model { model }) => {
                                println!("📦 Received model push");
                                local_model = Some(model);
                            }
                            Ok(TcpMessage::Heartbeat { .. }) => {}
                            Ok(_) => {}
                            Err(e) => {
                                println!("⚠ TCP parse error: {} | raw={}", e, trimmed);
                            }
                        }
                    }

                    // ---------------------------
                    // 3. Ensure we have local model
                    // ---------------------------
                    if local_model.is_none() {
                        println!("⬇ Fetching model from HTTP...");
                        match http.get(format!("{}/get_model", http_addr))
                            .header("x-api-key", api_key.clone())
                            .send()
                            .await
                        {
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

                    // ---------------------------
                    // 4. Fetch training parameters
                    // ---------------------------
                    let params = http.get(format!("{}/get_training_params", http_addr))
                        .header("x-api-key", api_key.clone())
                        .send()
                        .await?
                        .json::<Value>()
                        .await?;

                    let stop_loss = params["stop_loss"].as_f64().unwrap_or(0.01) as f32;
                    let max_epochs = params["max_epochs"].as_u64().unwrap_or(20) as usize;

                    // ---------------------------
                    // 5. TRAINING LOOP
                    // ---------------------------
                    println!("🎯 Training started!");

                    for epoch in 1..=max_epochs {
                        let loss = model.train_step(&xs, &ys, 0.01);
                        println!("📈 [{}] Epoch {}/{}  loss={:.6}", id, epoch, max_epochs, loss);

                        // send model update
                        let update = TcpMessage::ModelUpdate {
                            worker_id: id.to_string(),
                            round: epoch as u64,
                            model: model.clone(),
                        };

                        let msg = serde_json::to_string(&update)? + "\n";
                        let mut w = writer.lock().await;
                        if w.write_all(msg.as_bytes()).await.is_err() {
                            println!("❌ Failed to send update. Reconnecting...");
                            break;
                        }
                        drop(w);

                        if loss <= stop_loss {
                            println!("🏁 Early stop: loss {:.6} <= {}", loss, stop_loss);
                            break;
                        }

                        sleep(Duration::from_millis(300)).await;
                    }

                    println!("🔄 Training round completed.");
                }

                // If TCP broke, reconnect
                println!("🔁 Reconnecting...");
                sleep(Duration::from_secs(2)).await;
            }

            Err(e) => {
                println!("❌ Could not connect to server: {}. Retrying...", e);
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}
