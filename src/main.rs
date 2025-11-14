use anyhow::Result;
use clap::{Parser, Subcommand};

mod server;
mod client;
mod model;
mod messages;

#[derive(Parser)]
#[command(name = "swarm")]
#[command(about = "Federated training server + workers (dynamic, advanced)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the federated server (TCP for clients + HTTP REST API)
    Server {
        #[arg(long, default_value_t = String::from("0.0.0.0:50000"))]
        tcp_addr: String,
        #[arg(long, default_value_t = String::from("0.0.0.0:7000"))]
        http_addr: String,
        #[arg(long, default_value_t = 5000)]
        sync_interval_ms: u64,
        #[arg(long, default_value_t = String::from("training.log"))]
        log_file: String,
    },

    /// Start a worker client node
    Client {
        #[arg(long)]
        id: String,
        #[arg(long)]
        server: String,
        #[arg(long)]
        http: String, // http base like http://127.0.0.1:7000
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Server { tcp_addr, http_addr, sync_interval_ms, log_file } => {
            println!("🚀 Starting server: tcp={} http={}", tcp_addr, http_addr);
            server::Server::run(&tcp_addr, &http_addr, sync_interval_ms, log_file).await?;
        }
        Commands::Client { id, server, http } => {
            println!("🤖 Starting client: id={} server={} http={}", id, server, http);
            client::run_client(&id, &server, &http).await?;
        }
    }

    Ok(())
}
