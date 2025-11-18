# front.py
import streamlit as st
import requests, json, os, time, threading
from dotenv import load_dotenv
from datetime import datetime
import websocket

load_dotenv("project.env")
HTTP_ADDR = os.getenv("HTTP_ADDR")
API_KEY   = os.getenv("API_KEY")
BASE      = f"http://{HTTP_ADDR}"
WS_URL    = f"ws://{HTTP_ADDR}/ws"
HEADERS   = {"x-api-key": API_KEY, "Content-Type": "application/json"}

st.set_page_config(page_title="Decentralised AI Swarm", layout="wide")
st.title("🧠 Decentralised AI Swarm – Control Panel")
st.sidebar.header("Server Info")
st.sidebar.write(f"📡 HTTP: **{BASE}**")
st.sidebar.write(f"🔑 API KEY: `{API_KEY}`")

def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=2)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

def safe_post(url, payload=None):
    try:
        return requests.post(url, headers=HEADERS, json=payload, timeout=3)
    except Exception as e:
        st.error(f"POST error: {e}")
        return None

def human(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "-"

# WebSocket listener (keeps listening)
def ws_listener():
    while True:
        try:
            ws = websocket.WebSocket()
            ws.connect(WS_URL)
            while True:
                msg = ws.recv()
                try:
                    data = json.loads(msg)
                except:
                    continue
                event = data.get("event")
                if event in ["training_started", "model_updated", "training_complete", "snapshot"]:
                    st.session_state["trigger_refresh"] = True
                # continue listening (do not break)
        except Exception:
            # wait a bit before reconnecting
            time.sleep(1)
            continue

if "ws_thread_started" not in st.session_state:
    threading.Thread(target=ws_listener, daemon=True).start()
    st.session_state.ws_thread_started = True

if st.session_state.get("trigger_refresh"):
    st.session_state["trigger_refresh"] = False
    st.rerun()

# Auto refresh UI
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
if st.button("🔄 Manual Refresh"):
    st.rerun()

st.markdown("---")

# Create model
st.subheader("🧩 Create New Model")
with st.form("create_model_form"):
    input_dim   = st.number_input("Input Dim", value=4, min_value=1)
    hidden      = st.text_input("Hidden Layers (comma sep)", "8,8")
    output_dim  = st.number_input("Output Dim", value=1, min_value=1)
    if st.form_submit_button("Create Model"):
        try:
            hlist = [int(x.strip()) for x in hidden.split(",") if x.strip()]
            res = safe_post(f"{BASE}/create_model", {"input_dim": input_dim, "hidden_layers": hlist, "output_dim": output_dim})
            if res and res.status_code == 200:
                st.success("🎉 Model created")
            else:
                st.error(f"Failed: {res.status_code} {res.text}")
        except Exception as e:
            st.error(f"Invalid input: {e}")

st.markdown("---")

# Workers & dataset
st.subheader("🧑‍💻 Workers & Dataset")
status = safe_get(f"{BASE}/sync_status") or {"workers": {}}
workers = list(status.get("workers", {}).keys())
if workers:
    selected = st.selectbox("Select Worker", workers)
else:
    st.warning("No workers connected")
    selected = "worker1"

ds_text = st.text_area("Dataset JSON", height=200, placeholder='[{"x":[1,2,3,4],"y":[0]}]')
if st.button("Upload Dataset"):
    try:
        data = json.loads(ds_text)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list")
        for item in data:
            if not isinstance(item, dict) or "x" not in item or "y" not in item:
                raise ValueError("Each item must contain x and y")
        res = safe_post(f"{BASE}/upload_dataset", {"worker_id": selected, "data": data})
        if res and res.status_code == 200:
            st.success("Dataset uploaded")
        else:
            st.error(f"Failed: {res.status_code} {res.text}")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

st.markdown("---")

# Training params
st.subheader("⚙️ Training Parameters")
c1, c2 = st.columns(2)
stop_loss  = c1.number_input("Stop Loss", value=0.01)
max_epochs = c2.number_input("Max Epochs", value=20)
if st.button("Update Params"):
    res = safe_post(f"{BASE}/set_training_params", {"stop_loss": float(stop_loss), "max_epochs": int(max_epochs)})
    if res and res.status_code == 200:
        st.success("Params updated")
    else:
        st.error(f"Failed: {res.status_code} {res.text}")

st.markdown("---")

# Start training
st.subheader("🚀 Start Training")
if st.button("Start Training Now"):
    exists = requests.get(f"{BASE}/get_model", headers={"x-api-key": API_KEY})
    if exists.status_code != 200:
        st.error("Create a model first")
    else:
        res = safe_post(f"{BASE}/start_training")
        if res and res.status_code == 200:
            st.success("Training started")
        else:
            st.error(f"Failed: {res.status_code} {res.text}")

st.markdown("---")

# Download model
st.subheader("📥 Download Latest Model")
if st.button("Download model.json"):
    try:
        res = requests.get(f"{BASE}/download_model", headers={"x-api-key": API_KEY}, stream=True)
        if res.status_code == 200:
            st.download_button("⬇️ Save model.json", res.content, "model.json", "application/json")
        else:
            st.error(f"Failed: {res.status_code} {res.text}")
    except Exception as e:
        st.error(f"Download failed: {e}")

st.markdown("---")

# Live workers
st.subheader("📡 Workers (Live)")
status = safe_get(f"{BASE}/sync_status") or {"workers": {}}
if status["workers"]:
    for wid, meta in status["workers"].items():
        st.write(f"🔹 **{wid}** — updates: {meta.get('updates')} | state: {meta.get('state')} | last seen: {human(meta.get('last_seen', 0))}")
else:
    st.info("No workers connected.")
