// src/main.rs
mod server;
mod client;
mod model;
mod messages;

use anyhow::Result;
use dotenv::from_filename;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Load project.env
    from_filename("project.env").ok();

    let args = env::args().skip(1).collect::<Vec<_>>();

    if args.is_empty() {
        eprintln!("Usage:");
        eprintln!("  cargo run -- server");
        eprintln!("  cargo run -- client <worker_id>");
        return Ok(());
    }

    // Required env vars
    let http = env::var("HTTP_ADDR").expect("HTTP_ADDR not set in project.env");
    let tcp = env::var("TCP_ADDR").expect("TCP_ADDR not set in project.env");
    let api_key = env::var("API_KEY").expect("API_KEY not set in project.env");

    // Normalize HTTP URI
    let http_base = if http.starts_with("http://") || http.starts_with("https://") {
        http.clone()
    } else {
        format!("http://{}", http)
    };

    match args[0].as_str() {
        "server" => {
            println!("🚀 Starting server...");
            println!("HTTP = {}", http);
            println!("TCP  = {}", tcp);
            server::Server::run(&tcp, &http, api_key).await?;
        }

        "client" => {
            if args.len() < 2 {
                eprintln!("Usage: cargo run -- client <worker_id>");
                return Ok(());
            }
            let id = args[1].clone();
            println!("🤖 Starting worker '{}'", id);
            client::run_client(&id, &tcp, &http_base).await?;
        }

        _ => {
            eprintln!("Unknown command: {}", args[0]);
            eprintln!("Usage:");
            eprintln!("  cargo run -- server");
            eprintln!("  cargo run -- client <worker_id>");
        }
    }

    Ok(())
}
