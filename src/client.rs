// src/client.rs
use anyhow::Result;
use reqwest::Client;
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    time::{sleep, Duration},
};

use crate::{messages::TcpMessage, model::DynamicModel};

pub async fn run_client(id: &str, tcp_addr: &str, http_addr: &str) -> Result<()> {
    println!("🤖 Worker `{}` starting. TCP={} HTTP={}", id, tcp_addr, http_addr);

    let api_key = std::env::var("API_KEY").expect("API_KEY missing");
    let http_client = Client::new();

    // connect tcp
    let mut stream = TcpStream::connect(tcp_addr).await?;
    let (r, mut w) = stream.split();
    let mut reader = BufReader::new(r);

    // register via TCP for model pushes
    let reg = TcpMessage::RequestModel { worker_id: id.to_string() };
    let reg_s = serde_json::to_string(&reg)? + "\n";
    w.write_all(reg_s.as_bytes()).await?;
    println!("🔌 TCP registered as {}", id);

    // wait until dataset uploaded on server
    loop {
        let url = format!("{}/get_dataset?worker_id={}", http_addr.trim_end_matches('/'), id);
        match http_client.get(&url).header("x-api-key", api_key.clone()).send().await {
            Ok(resp) if resp.status().is_success() => {
                let val: Value = resp.json().await.unwrap_or(Value::Null);
                if let Some(arr) = val.as_array() {
                    if !arr.is_empty() {
                        println!("📥 Loaded {} samples for worker {}", arr.len(), id);
                        break;
                    }
                }
            }
            _ => {}
        }
        println!("⏳ Worker {} waiting for dataset...", id);
        sleep(Duration::from_secs(2)).await;
    }

    // Wait for Start command over TCP
    println!("⏳ Worker {} waiting for Start...", id);
    loop {
        let mut ln = String::new();
        let n = reader.read_line(&mut ln).await?;
        if n == 0 { println!("TCP closed"); return Ok(()); }
        let trimmed = ln.trim();
        if trimmed.is_empty() { continue; }
        match serde_json::from_str::<TcpMessage>(trimmed) {
            Ok(TcpMessage::Start) => {
                println!("🚀 Worker {} received Start", id);
                break;
            }
            Ok(TcpMessage::Model { model: _ }) => {
                // model pushed early — ignore until start
            }
            Err(e) => {
                println!("TCP parse error on client: {} | raw: {}", e, trimmed);
            }
            _ => {}
        }
    }

    // fetch training params
    let params = http_client.get(format!("{}/get_training_params", http_addr))
        .header("x-api-key", api_key.clone())
        .send().await?.json::<Value>().await?;
    let stop_loss = params["stop_loss"].as_f64().unwrap_or(0.01) as f32;
    let max_epochs = params["max_epochs"].as_u64().unwrap_or(20) as usize;

    // fetch dataset (again)
    let ds_resp = http_client.get(format!("{}/get_dataset?worker_id={}", http_addr, id))
        .header("x-api-key", api_key.clone()).send().await?;
    let ds_json: Value = ds_resp.json().await?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for s in ds_json.as_array().unwrap() {
        xs.push(s["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>());
        ys.push(s["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>());
    }

    // fetch current model
    let model: DynamicModel = loop {
        let resp = http_client.get(format!("{}/get_model", http_addr))
            .header("x-api-key", api_key.clone()).send().await?;
        if resp.status().is_success() {
            let txt = resp.text().await?;
            match serde_json::from_str(&txt) {
                Ok(m) => break m,
                Err(_) => { sleep(Duration::from_secs(1)).await; continue; }
            }
        }
        sleep(Duration::from_secs(1)).await;
    };

    let mut model = model;

    // training loop
    for epoch in 1..=max_epochs {
        let loss = model.train_step(&xs, &ys, 0.01);
        println!("[{}] epoch {}/{} loss={:.6}", id, epoch, max_epochs, loss);

        // send update via TCP
        let upd = TcpMessage::ModelUpdate { worker_id: id.to_string(), round: epoch as u64, model: model.clone() };
        let upd_s = serde_json::to_string(&upd)? + "\n";
        w.write_all(upd_s.as_bytes()).await?;

        // small sleep
        sleep(Duration::from_millis(500)).await;

        if loss <= stop_loss {
            println!("🏁 Worker {} stopping early, loss {:.6} <= stop_loss {:.6}", id, loss, stop_loss);
            break;
        }
    }

    println!("🏁 Worker {} finished training", id);
    Ok(())
}
