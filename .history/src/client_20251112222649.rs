use crate::messages::Message;
use crate::model::Model;
use anyhow::Result;
use rand::distributions::Standard;
use rand::Rng;

use serde_json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use std::time::Duration;

pub async fn run_client(id: &str, server_addr: &str) -> Result<()> {
    let mut model = Model::new(4, 8, 1);
    let stream = TcpStream::connect(server_addr).await?;
    // Use `into_split` instead of `split`
    let (r, w) = stream.into_split();
    let mut reader = BufReader::new(r);
    let mut writer = w;
    let mut line = String::new();

    println!("✅ Client {} connected to server at {}", id, server_addr);

    loop {
        let (xs, ys) = make_synthetic_batch(16, 4);
        let loss = model.train_step(&xs, &ys, 0.01);
        println!("[{}] local loss: {:.4}", id, loss);
        let json = serde_json::to_string(&msg)?;
        writer.write_all(format!("{}\n", json).as_bytes()).await?;
        writer.flush().await?;

        line.clear();
        if reader.read_line(&mut line).await? > 0 {
            if let Ok(Message::Weights(flat)) = serde_json::from_str::<Message>(line.trim()) {
                if let Ok(global) = Model::from_flat_vec(&flat) {
                    model = global;
                    println!("[{}] ✅ synced global weights", id);
                }
            }
            line.clear();
        }

        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

fn make_synthetic_batch(batch: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::rng();
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
