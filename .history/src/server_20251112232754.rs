use crate::messages::Message;
use crate::model::Model;
use anyhow::Result;
use serde_json;
use std::sync::Arc;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpListener,
    sync::{broadcast, Mutex},
};

pub struct Server {
    model: Arc<Mutex<Model>>,
    tx: broadcast::Sender<Vec<f32>>,
}

impl Server {
    /// Create a new Server instance
    pub fn new() -> Self {
        let model = Arc::new(Mutex::new(Model::new(4, 8, 1)));
        let (tx, _rx) = broadcast::channel(16);
        Self { model, tx }
    }

    /// Run the server
    pub async fn run(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        println!("✅ Server listening on {}", addr);

        loop {
            let (stream, addr) = listener.accept().await?;
            println!("🔗 New client connected: {}", addr);

            // Clone shared state
            let m = Arc::clone(&self.model);
            let tx_clone = self.tx.clone();
            let mut rx = tx_clone.subscribe();

            let (r, w) = stream.into_split();
            let mut reader = BufReader::new(r);
            let writer_lock = Arc::new(Mutex::new(w));

            // ================= Reader task =================
            let m_reader = Arc::clone(&m);
            let tx_reader = tx_clone.clone();
            let writer_clone = Arc::clone(&writer_lock);

            tokio::spawn(async move {
                let mut line = String::new();

                loop {
                    line.clear();
                    let bytes_read = match reader.read_line(&mut line).await {
                        Ok(n) => n,
                        Err(e) => {
                            eprintln!("❌ Error reading from {}: {}", addr, e);
                            break;
                        }
                    };

                    if bytes_read == 0 {
                        println!("❌ Client {} disconnected.", addr);
                        break;
                    }

                    if let Ok(msg) = serde_json::from_str::<Message>(line.trim()) {
                        match msg {
                            Message::Weights(flat) => {
                                if let Ok(peer_model) = Model::from_flat_vec(&flat) {
                                    // merge
                                    {
                                        let mut global = m_reader.lock().await;
                                        global.merge_inplace(&peer_model, 0.5);
                                    }

                                    // prepare updated weights
                                    let updated = {
                                        let global = m_reader.lock().await;
                                        global.to_flat_vec()
                                    };

                                    let _ = tx_reader.send(updated);
                                }
                            }

                            Message::RequestSync => {
                                let data = {
                                    let g = m_reader.lock().await;
                                    serde_json::to_string(&Message::Weights(g.to_flat_vec())).unwrap()
                                };

                                let mut w = writer_clone.lock().await;
                                if let Err(e) = w.write_all(format!("{}\n", data).as_bytes()).await {
                                    eprintln!("⚠️ Failed to send sync to {}: {}", addr, e);
                                }
                            }

                            _ => {}
                        }
                    }
                }
            });

            // ================= Writer task =================
            let writer_clone = Arc::clone(&writer_lock);
            tokio::spawn(async move {
                while let Ok(flat) = rx.recv().await {
                    let msg = serde_json::to_string(&Message::Weights(flat)).unwrap();
                    let mut w = writer_clone.lock().await;
                    if let Err(e) = w.write_all(format!("{}\n", msg).as_bytes()).await {
                        eprintln!("⚠️ Broadcast send to {} failed: {}", addr, e);
                        break;
                    }
                }
            });
        }
    }
}
