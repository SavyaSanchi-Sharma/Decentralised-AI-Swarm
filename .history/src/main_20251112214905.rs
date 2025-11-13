use anyhow::Result;
use clap::{Parser, Subcommand};

mod server;
mod client;
mod model;
mod messages;

#[derive(Parser)]
#[command(name = "rusthive-federated")]
#[command(about = "Federated neural training system in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run as server
    Server {
        #[arg(long, default_value_t = String::from("0.0.0.0:50000"))]
        addr: String,
    },
    /// Run as client
    Client {
        #[arg(long)]
        id: String,
        #[arg(long)]
        server: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Server { addr } => {
            println!("🚀 Starting RustHive Server on {addr}");
            server::Server::run(&addr).await?;
        }
        Commands::Client { id, server } => {
            println!("🤖 Starting Client `{id}`, connecting to {server}");
            client::run_client(&id, &server).await?;
        }
    }
    Ok(())
}
