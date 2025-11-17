// src/main.rs
mod server;
mod client;
mod model;
mod messages;

use anyhow::Result;
use clap::{Parser, Subcommand};
use dotenv::from_filename;
use std::env;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Server,
    Client {
        #[arg(long)]
        id: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // load project.env (fail fast)
    from_filename("project.env").ok();

    let cli = Cli::parse();

    let http = env::var("HTTP_ADDR").expect("HTTP_ADDR not set in project.env");
    let tcp = env::var("TCP_ADDR").expect("TCP_ADDR not set in project.env");
    let api_key = env::var("API_KEY").expect("API_KEY not set in project.env");

    match cli.cmd {
        Commands::Server => {
            server::Server::run(&tcp, &http, api_key).await?;
        }
        Commands::Client { id } => {
            // pass http as full base URI (no leading http:// in env, so add)
            let http_base = format!("http://{}", http);
            client::run_client(&id, &tcp, &http_base).await?;
        }
    }

    Ok(())
}
