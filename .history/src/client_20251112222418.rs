use crate::messages::Message;
use crate::model::Model;
use anyhow::Result;
use rand::Rng;
use serde_json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use std::time::Duration;

pub async fn run_client(id: &str, server_addr: &str) -> Result<()> {
    let mut model = Model::new(4, 8, 1);
    let stream = TcpStream::connect(server_addr).await?;
    let (r, mut w) = stream.into_split();
    let mut reader = BufReader::new(r);
    let mut line = String::new();

    println!("✅ Client {id} connected to server at {server_addr}");

    loop {
        // 1️⃣ Train locally on synthetic data
        let (xs, ys) = make_synthetic_batch(16, 4);
        let loss = model.train_step(&xs, &ys, 0.01);
        println!("[{id}] local loss: {loss:.4}");

        // 2️⃣ Send current weights to server
        let msg = Message::Weights(model.to_flat_vec());
        let json = serde_json::to_string(&msg)?;
        w.write_all(format!("{}\n", json).as_bytes()).await?;
        w.flush().await?;

        // 3️⃣ Wait for global update
        line.clear();
        if reader.read_line(&mut line).await? > 0 {
            if let Ok(Message::Weights(flat)) = serde_json::from_str::<Message>(line.trim()) {
                if let Ok(global) = Model::from_flat_vec(&flat) {
                    model = global;
                    println!("[{id}] ✅ synced global weights");
                }
            }
            line.clear();
        }

        // 4️⃣ Small delay before next local step
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

fn make_synthetic_batch(batch: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let mut xs = Vec::with_capacity(batch);
    let mut ys = Vec::with_capacity(batch);
    for _ in 0..batch {
        let x: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let s: f32 = x.iter().sum();
        ys.push(vec![(s > 0.0) as i32 as f32]);
        xs.push(x);
    }
    (xs, ys)
}
