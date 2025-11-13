use crate::messages::Message;
use crate::model::Model;
use anyhow::Result;
use serde_json;
use std::sync::{Arc};
use tokio::sync::Mutex;
use tokio::{io::{AsyncBufReadExt, AsyncWriteExt, BufReader}, net::{TcpListener}, sync::broadcast};

pub struct Server {
    model: Arc<Mutex<Model>>,
    tx: broadcast::Sender<Vec<f32>>,
}

impl Server {
    pub async fn run(addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        let model = Arc::new(Mutex::new(Model::new(4, 8, 1)));
        let (tx, _rx) = broadcast::channel(16);

        loop {
            let (stream, _) = listener.accept().await?;
            let m = Arc::clone(&model);
            let tx_clone = tx.clone();
            let mut rx = tx_clone.subscribe();

            tokio::spawn(async move {
                let (r, mut w) = stream.into_split();
                let mut reader = BufReader::new(r);
                let mut line = String::new();

                // Task for receiving updates
                loop {
                    line.clear();
                    if reader.read_line(&mut line).await.unwrap() == 0 {
                        break;
                    }
                    if let Ok(msg) = serde_json::from_str::<Message>(&line) {
                        match msg {
                            Message::Weights(flat) => {
                                if let Ok(peer_model) = Model::from_flat_vec(&flat) {
                                    let mut global = m.lock().await;
                                    tokio::task::spawn_blocking(move || {
                                        global.merge_inplace(&peer_model, 0.5);
                                    }).await.unwrap();
                                    let updated = global.to_flat_vec();
                                    let _ = tx_clone.send(updated);
                                }
                            }
                            Message::RequestSync => {
                                let g = m.lock().await;
                                let data = serde_json::to_string(&Message::Weights(g.to_flat_vec())).unwrap();
                                let _ = w.write_all(format!("{}\n", data).as_bytes()).await;
                            }
                            _ => {}
                        }
                    }
                }
            });

            // Task for broadcasting updates
            let (stream, _) = listener.accept().await?;
            let (_, mut w) = stream.into_split();
            tokio::spawn(async move {
                while let Ok(flat) = rx.recv().await {
                    let msg = serde_json::to_string(&Message::Weights(flat)).unwrap();
                    let _ = w.write_all(format!("{}\n", msg).as_bytes()).await;
                }
            });
        }
    }
}
