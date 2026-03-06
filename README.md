# Decentralised AI Swarm

A fully **data-parallel, decentralised neural network training system** built in Rust with a Streamlit control panel (`front.py`).

Multiple worker processes each train on a local data shard, compute raw gradients, and ship them to the server every round. The server aggregates gradients from **all** connected workers in a sample-weighted average, applies one global optimiser step, and broadcasts the updated model — true parallel training across any number of machines.

---

## System Overview

```
 ┌─────────────────────────────────────────────────────┐
 │               front.py  (Streamlit UI)               │
 │  Configure server · Upload datasets · Start training │
 │  Monitor workers live · Download trained model       │
 └──────────────────────┬──────────────────────────────┘
                        │  REST + WebSocket
 ┌──────────────────────▼──────────────────────────────┐
 │                   Server (Rust)                      │
 │   HTTP  · TCP  · UDP discovery  · WebSocket /ws      │
 │                                                      │
 │   ┌──────────────────────────────────────────────┐   │
 │   │  Gradient Barrier                            │   │
 │   │  waits for all expected workers to submit    │   │
 │   │  → weighted average → LR step → broadcast   │   │
 │   └──────────────────────────────────────────────┘   │
 └────────┬────────────────┬────────────────┬───────────┘
          │  TCP           │                │
   ┌──────▼───┐     ┌──────▼───┐     ┌──────▼───┐
   │ Worker A  │     │ Worker B  │     │ Worker C  │
   │ shard A   │     │ shard B   │     │ shard C   │
   │ compute dw│     │ compute dw│     │ compute dw│
   └───────────┘     └───────────┘     └───────────┘
```

### How a training round works

1. **Setup** — in `front.py` (or curl): create model architecture, upload per-worker dataset shards, set training params.
2. **`POST /start_training`** — server snapshots which workers are currently connected (`round_expected_workers`) and broadcasts `Start` over TCP. This snapshot makes the barrier race-free: late-joiners or mid-round disconnects never stall a round.
3. **Workers in parallel** — each worker computes gradients over its local shard (no local weight update), sends `GradUpdate { dw, db, n_samples }` to the server.
4. **Barrier closes** — once every expected worker has submitted, the server computes a sample-weighted average of `dw`/`db`, applies the global LR step, saves a versioned snapshot, and broadcasts the updated `Model` to all workers.
5. **Next round** starts immediately.
6. **Early stop** — if any worker's re-evaluated loss falls below `stop_loss` it waits for the next `Start` signal.

---

## Files

| File | Role |
|---|---|
| `src/main.rs` | Entry point — reads `project.env`, dispatches `server` / `client` mode |
| `src/server.rs` | Central coordinator: HTTP (axum 0.7), TCP, UDP discovery, WebSocket, gradient barrier |
| `src/client.rs` | Worker: UDP auto-discovery, TCP connection, heartbeat, dataset fetch, gradient compute + send, model receive |
| `src/model.rs` | `DynamicModel` — configurable MLP: forward, backprop (gradient-only), `apply_gradients`, `predict`, `merge_inplace` |
| `src/messages.rs` | `TcpMessage` wire protocol: `RequestModel`, `Start`, `GradUpdate`, `ModelUpdate` (legacy), `Model`, `Heartbeat` |
| `front.py` | Streamlit control panel — REST + WebSocket to the Rust server |
| `project.env` | Addresses, API key, training defaults |

---

## `front.py` — Streamlit Control Panel

`front.py` is fully connected to the Rust server via its HTTP and WebSocket endpoints. Workers are **not** contacted directly.

Enter the server host/port in the **sidebar** — it defaults to `127.0.0.1:7000` for local development.

| UI section | API call |
|---|---|
| Create model | `POST /create_model` |
| Workers & Dataset | `GET /sync_status` + `POST /upload_dataset` |
| Training Parameters | `POST /set_training_params` |
| Start Training | `GET /get_model` (check exists) → `POST /start_training` |
| Download Model | `GET /download_model` |
| Workers (Live) | `GET /sync_status` |

A background WebSocket thread subscribes to `/ws` and triggers a Streamlit `rerun()` on every `model_updated` or `training_started` event — so the dashboard stays live without any manual refresh.

### Running the frontend

```bash
pip install streamlit websocket-client python-dotenv requests
streamlit run front.py
```

---

## Running on Multiple Machines (True Parallel Training)

Start **one server** and **one worker per machine**. Workers auto-discover the server via UDP broadcast (LAN) or connect directly with the server's TCP address.

```bash
# Machine 1 (server)
cargo run --release -- server

# Machine 2, 3, 4 … (each a separate machine or process)
cargo run --release -- client worker_a
cargo run --release -- client worker_b
cargo run --release -- client worker_c
```

Set `TCP_ADDR`, `HTTP_ADDR` in `project.env` to the server's actual LAN IP before starting workers. Workers with an empty `TCP_ADDR` will use UDP broadcast to find the server automatically.

---

## TCP Protocol

Newline-delimited JSON. Tagged with `"cmd"`.

| Message | Direction | Description |
|---|---|---|
| `RequestModel` | worker → server | Register and request current model |
| `Start` | server → worker | Begin a training round |
| `GradUpdate` | worker → server | Per-layer gradients + sample count *(primary path)* |
| `ModelUpdate` | worker → server | Full weights — legacy federated averaging mode |
| `Model` | server → worker | Broadcast fused/updated model |
| `Heartbeat` | worker ↔ server | Keep-alive |

---

## HTTP API

Endpoints require `x-api-key` header (set in `project.env`).

| Endpoint | Method | Description |
|---|---|---|
| `/create_model` | POST | `{ input_dim, hidden_layers, output_dim }` |
| `/upload_dataset` | POST | `{ worker_id, data: [{x, y}] }` |
| `/register_worker` | POST | Pre-register a worker |
| `/set_training_params` | POST | `{ stop_loss, max_epochs, learning_rate }` |
| `/start_training` | POST | Snapshot workers, broadcast `Start`, clear grad buffer |
| `/get_model` | GET | Current global model JSON |
| `/get_dataset` | GET | `?worker_id=<id>` — worker fetches its shard |
| `/get_training_params` | GET | Current hyperparameters |
| `/sync_status` | GET | Per-worker state, grad submitted, in-current-round |
| `/download_model` | GET | Model JSON (latest or `?version=N`) |
| `/ws` | WebSocket | Live event stream |

---

## Configuration — `project.env`

```env
HTTP_ADDR=127.0.0.1:7000
TCP_ADDR=127.0.0.1:9000
UDP_ADDR=0.0.0.0:9999
BROADCAST_ADDR=255.255.255.255:9999
API_KEY=your_api_key_here

STOP_LOSS=0.01
MAX_EPOCHS=20
```

For LAN multi-machine use, set `HTTP_ADDR` and `TCP_ADDR` to the server machine's LAN IP.

---

## Build

```bash
# Requires Rust stable (edition 2024)
cargo build --release
```

---

## Dependencies

| Crate | Purpose |
|---|---|
| `tokio` | Async runtime |
| `axum 0.7` | HTTP + WebSocket server |
| `tower-http 0.5` | CORS middleware |
| `reqwest` | HTTP client (workers) |
| `serde` / `serde_json` | Serialisation |
| `rand` | Weight initialisation |
| `anyhow` | Error handling |
| `dotenv` | `project.env` loading |
| `chrono` | Timestamps |
