# front.py
import streamlit as st
import requests, json, os, time, threading
from datetime import datetime
import websocket

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Decentralised AI Swarm", layout="wide")
st.title("🧠 Decentralised AI Swarm – Control Panel")

# ---------------------------------------------------------------------------
# Sidebar — server configuration (user-editable, persisted in session_state)
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Server Config")

if "server_host" not in st.session_state:
    st.session_state.server_host = "127.0.0.1"
if "server_port" not in st.session_state:
    st.session_state.server_port = 7000
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY", "")

server_host = st.sidebar.text_input("Server Host", value=st.session_state.server_host)
server_port = st.sidebar.number_input("HTTP Port", value=st.session_state.server_port, min_value=1, max_value=65535)
api_key     = st.sidebar.text_input("API Key", value=st.session_state.api_key, type="password")

# Save back so values survive reruns
st.session_state.server_host = server_host
st.session_state.server_port = int(server_port)
st.session_state.api_key     = api_key

BASE    = f"http://{server_host}:{server_port}"
WS_URL  = f"ws://{server_host}:{server_port}/ws"
HEADERS = {"x-api-key": api_key, "Content-Type": "application/json"}

st.sidebar.markdown("---")
st.sidebar.write(f"📡 Endpoint: `{BASE}`")

# ---------------------------------------------------------------------------
# Auto-refresh controls
# ---------------------------------------------------------------------------
st.sidebar.subheader("Auto Refresh")
refresh_rate = st.sidebar.slider("Refresh interval (sec)", 1, 10, 3)
enable_auto  = st.sidebar.checkbox("Enable")
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if enable_auto:
    now = time.time()
    if now - st.session_state.last_refresh >= refresh_rate:
        st.session_state.last_refresh = now
        st.rerun()
if st.sidebar.button("🔄 Manual Refresh"):
    st.rerun()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def safe_post(url, payload=None):
    try:
        return requests.post(url, headers=HEADERS, json=payload, timeout=5)
    except Exception as e:
        st.error(f"POST error: {e}")
        return None

def human_ts(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "-"

# ---------------------------------------------------------------------------
# WebSocket listener — background thread, triggers UI refresh on events
# ---------------------------------------------------------------------------
def ws_listener(ws_url: str):
    """Reconnecting WebSocket thread. Sets trigger_refresh on relevant events."""
    while True:
        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url)
            while True:
                msg = ws.recv()
                try:
                    data = json.loads(msg)
                except:
                    continue
                if data.get("event") in [
                    "training_started", "model_updated",
                    "training_complete", "snapshot",
                    "worker_connect", "worker_disconnect",
                ]:
                    st.session_state["trigger_refresh"] = True
        except Exception:
            time.sleep(1)

# Only start one thread per session; re-start if the WS URL changed.
current_ws_key = f"ws_thread_{WS_URL}"
if current_ws_key not in st.session_state:
    # Mark the old thread's URL as stale (it will exit on next recv error)
    st.session_state[current_ws_key] = True
    threading.Thread(target=ws_listener, args=(WS_URL,), daemon=True).start()

if st.session_state.get("trigger_refresh"):
    st.session_state["trigger_refresh"] = False
    st.rerun()

# ---------------------------------------------------------------------------
# ── 1. Create Model
# ---------------------------------------------------------------------------
st.subheader("🧩 Create New Model")
with st.form("create_model_form"):
    col1, col2, col3 = st.columns(3)
    input_dim  = col1.number_input("Input Dim",  value=4, min_value=1)
    hidden     = col2.text_input("Hidden Layers (comma-sep)", "8,8")
    output_dim = col3.number_input("Output Dim", value=1, min_value=1)
    if st.form_submit_button("Create Model"):
        try:
            hlist = [int(x.strip()) for x in hidden.split(",") if x.strip()]
            res = safe_post(f"{BASE}/create_model",
                            {"input_dim": int(input_dim), "hidden_layers": hlist, "output_dim": int(output_dim)})
            if res and res.status_code == 200:
                st.success(f"✅ Model created — layers: [{int(input_dim)}, {', '.join(str(h) for h in hlist)}, {int(output_dim)}]")
            else:
                st.error(f"Failed: {res.status_code if res else 'no response'}")
        except Exception as e:
            st.error(f"Invalid input: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── 2. Workers & Dataset Upload
# ---------------------------------------------------------------------------
st.subheader("🧑‍💻 Workers & Dataset")

status  = safe_get(f"{BASE}/sync_status") or {"workers": {}}
workers = list(status.get("workers", {}).keys())

col_sel, col_hint = st.columns([2, 3])
if workers:
    selected = col_sel.selectbox("Select Worker", workers)
    col_hint.info(f"{len(workers)} worker(s) connected")
else:
    col_sel.warning("No workers connected yet")
    selected = col_sel.text_input("Worker ID (manual)", value="worker_a")

ds_text = st.text_area(
    "Dataset JSON (list of `{x, y}` objects)",
    height=180,
    placeholder='[{"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0]}, ...]',
)
if st.button("⬆️ Upload Dataset"):
    try:
        data = json.loads(ds_text)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array")
        for item in data:
            if "x" not in item or "y" not in item:
                raise ValueError("Each sample must have 'x' and 'y' keys")
        res = safe_post(f"{BASE}/upload_dataset", {"worker_id": selected, "data": data})
        if res and res.status_code == 200:
            st.success(f"Dataset uploaded for **{selected}** ({len(data)} samples)")
        else:
            st.error(f"Failed: {res.status_code if res else 'no response'}")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── 3. Training Parameters
# ---------------------------------------------------------------------------
st.subheader("⚙️ Training Parameters")
c1, c2, c3 = st.columns(3)
stop_loss  = c1.number_input("Stop Loss",       value=0.01,  format="%.4f")
max_epochs = c2.number_input("Max Epochs",      value=20,    min_value=1)
lr         = c3.number_input("Learning Rate",   value=0.01,  format="%.4f")
if st.button("Update Params"):
    res = safe_post(f"{BASE}/set_training_params",
                    {"stop_loss": float(stop_loss),
                     "max_epochs": int(max_epochs),
                     "learning_rate": float(lr)})
    if res and res.status_code == 200:
        st.success("Params updated")
    else:
        st.error(f"Failed: {res.status_code if res else 'no response'}")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── 4. Start Training
# ---------------------------------------------------------------------------
st.subheader("🚀 Start Training")

sync = safe_get(f"{BASE}/sync_status") or {"workers": {}}
n_workers = len(sync.get("workers", {}))

if n_workers == 0:
    st.warning("⚠️ No workers connected. Start workers before training.")
else:
    st.info(f"🟢 **{n_workers}** worker(s) ready — all will train in parallel this round.")

if st.button("▶️ Start Training Now", disabled=(n_workers == 0)):
    model_check = requests.get(f"{BASE}/get_model", headers={"x-api-key": api_key})
    if model_check.status_code != 200:
        st.error("Create a model first (step 1)")
    else:
        res = safe_post(f"{BASE}/start_training")
        if res and res.status_code == 200:
            data = res.json()
            st.success(f"✅ Training started across **{data.get('workers', n_workers)}** worker(s)")
        else:
            st.error(f"Failed: {res.status_code if res else 'no response'}")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── 5. Download Model
# ---------------------------------------------------------------------------
st.subheader("📥 Download Trained Model")
col_dl, col_ver = st.columns([1, 2])
dl_version = col_ver.text_input("Version (leave blank for latest)", value="")
if col_dl.button("Download model.json"):
    url = f"{BASE}/download_model"
    if dl_version.strip():
        url += f"?version={dl_version.strip()}"
    try:
        res = requests.get(url, headers={"x-api-key": api_key}, stream=True)
        if res.status_code == 200:
            st.download_button("⬇️ Save model.json", res.content, "model.json", "application/json")
        else:
            st.error(f"Failed: {res.status_code} — {res.text}")
    except Exception as e:
        st.error(f"Download failed: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── 6. Live Worker Monitor
# ---------------------------------------------------------------------------
st.subheader("📡 Live Worker Monitor")

status = safe_get(f"{BASE}/sync_status") or {"workers": {}}
workers_map = status.get("workers", {})
pending = status.get("pending_grad_count", 0)
expected = status.get("expected_this_round", 0)

if workers_map:
    st.write(f"Pending gradient submissions this round: **{pending} / {expected}**")
    for wid, meta in workers_map.items():
        state       = meta.get("state", "?")
        updates     = meta.get("updates", 0)
        last_seen   = human_ts(meta.get("last_seen", 0))
        grad_done   = "✅" if meta.get("grad_submitted") else "⏳"
        in_round    = "🔵" if meta.get("in_current_round") else "⚪"
        addr        = meta.get("addr", "")
        state_badge = "🟡 training" if state == "training" else "⚪ idle"

        st.write(
            f"{in_round} **{wid}** ({addr}) — {state_badge} | "
            f"updates: {updates} | grad: {grad_done} | last seen: {last_seen}"
        )
else:
    st.info("No workers connected. Run: `cargo run --release -- client <worker_id>`")
