use crate::messages::Message;
use crate::model::DynamicModel;
use anyhow::Result;

use rand::Rng;
use serde_json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use std::time::Duration;

use reqwest::Client as HttpClient;

// ===============================================================
// Worker Client Node
// ===============================================================
pub async fn run_client(id: &str, server_tcp: &str, server_http: &str) -> Result<()> {
    println!("🤖 Starting client `{}` connecting to {}", id, server_tcp);

    // ---- TCP connection to server for federated learning ----
    let stream = TcpStream::connect(server_tcp).await?;
    let (r, mut w) = stream.into_split();
    let mut reader = BufReader::new(r);
    let mut line = String::new();

    let mut model_opt: Option<DynamicModel> = None;
    let http = HttpClient::new();

    // Initially request a model
    send_request_model(&mut w).await?;

    loop {
        // ============================================================
        // 1) POLL TRAINING STATUS (HTTP)
        // ============================================================
        let status = http.get(format!("{}/training_status", server_http))
            .send().await;

        let mut training_active = false;
        let mut epochs_run = 0usize;

        if let Ok(resp) = status {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                training_active = json.get("active")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                epochs_run = json.get("epochs_run")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
            }
        }

        // ============================================================
        // 2) TRAINING HAS NOT STARTED YET — WAIT (DON'T EXIT)
        // ============================================================
        if !training_active && epochs_run == 0 {
            println!("[{}] Waiting for training to start...", id);

            // stay alive — request model to sync whenever ready
            send_request_model(&mut w).await?;

            tokio::time::sleep(Duration::from_secs(1)).await;
            continue;
        }

        // ============================================================
        // 3) TRAINING COMPLETED — CLEAN EXIT
        // ============================================================
        if !training_active && epochs_run > 0 {
            println!("[{}] Training finished — exiting worker", id);
            break;
        }

        // ============================================================
        // 4) TRY READING ANY TCP MESSAGE FROM SERVER (MODEL ETC.)
        // ============================================================
        line.clear();
        if let Ok(Ok(n)) = tokio::time::timeout(
            Duration::from_millis(200),
            reader.read_line(&mut line)
        ).await {
            if n > 0 {
                if let Ok(msg) = serde_json::from_str::<Message>(&line) {
                    if let Message::Model(m) = msg {
                        println!("[{}] received global model from server", id);
                        model_opt = Some(m);
                    }
                }
            }
        }

        // ============================================================
        // 5) FETCH DATASET IF AVAILABLE
        // ============================================================
        let dataset = if model_opt.is_some() {
            match http.get(format!("{}/dataset", server_http)).send().await {
                Ok(resp) => match resp.json::<Vec<(Vec<f32>, Vec<f32>)>>().await {
                    Ok(ds) => ds,
                    Err(_) => Vec::new(),
                },
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };

        // ============================================================
        // 6) TRAIN ONE LOCAL STEP
        // ============================================================
        if let Some(mut local_model) = model_opt.clone() {
            if dataset.is_empty() {
                // fallback to synthetic batch
                let (xs, ys) = make_synthetic_batch(16, local_model.layer_sizes[0]);
                let loss = local_model.train_step(&xs, &ys, 0.01);
                println!("[{}] local loss (synthetic): {:.4}", id, loss);
            } else {
                // real dataset training
                let xs: Vec<_> = dataset.iter().map(|p| p.0.clone()).collect();
                let ys: Vec<_> = dataset.iter().map(|p| p.1.clone()).collect();
                let loss = local_model.train_step(&xs, &ys, 0.01);
                println!("[{}] local loss: {:.4}", id, loss);
            }

            // ========================================================
            // 7) SEND UPDATE
            // ========================================================
            let out = serde_json::to_string(&Message::ModelUpdate(local_model.clone()))?;
            w.write_all(format!("{}\n", out).as_bytes()).await?;

            // wait for new global model
            line.clear();
            if let Ok(Ok(n)) = tokio::time::timeout(Duration::from_millis(1000), reader.read_line(&mut line)).await {
                if n > 0 {
                    if let Ok(Message::Model(m)) = serde_json::from_str::<Message>(&line) {
                        println!("[{}] received updated global model", id);
                        model_opt = Some(m);
                    }
                }
            }
        } else {
            // Still don't have a model — ask again
            send_request_model(&mut w).await?;
        }

        // small wait to not overload network
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    Ok(())
}

// ===============================================================
// Helpers
// ===============================================================
async fn send_request_model(w: &mut tokio::net::tcp::OwnedWriteHalf) -> Result<()> {
    let req = serde_json::to_string(&Message::RequestModel)?;
    w.write_all(format!("{}\n", req).as_bytes()).await?;
    Ok(())
}

fn make_synthetic_batch(batch: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::rng();
    let mut xs = Vec::with_capacity(batch);
    let mut ys = Vec::with_capacity(batch);

    for _ in 0..batch {
        let x: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let sum: f32 = x.iter().sum();
        ys.push(vec![(sum > 0.0) as i32 as f32]);
        xs.push(x);
    }

    (xs, ys)
}
