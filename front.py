# front.py
import streamlit as st
import requests, json, os, time
from dotenv import load_dotenv
from datetime import datetime

# ---------------------------------------------------------
# Load project.env
# ---------------------------------------------------------
load_dotenv("project.env")

HTTP_ADDR = os.getenv("HTTP_ADDR")
API_KEY  = os.getenv("API_KEY")

BASE     = f"http://{HTTP_ADDR}"
HEADERS  = {"x-api-key": API_KEY, "Content-Type": "application/json"}

# ---------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Decentralised AI Swarm", layout="wide")
st.title("🧠 Decentralised AI Swarm – Control Panel")

st.sidebar.header("Server Info")
st.sidebar.write(f"HTTP: **{BASE}**")
st.sidebar.write(f"API KEY: `{API_KEY}`")

# ---------------------------------------------------------
# Utility wrappers
# ---------------------------------------------------------
def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=2)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def safe_post(url, payload=None):
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=3)
        return r
    except Exception as e:
        st.error(f"POST error: {e}")
        return None

def human(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "-"


# ---------------------------------------------------------
# Auto-refresh (CORRECT, WORKING)
# ---------------------------------------------------------
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

# Manual refresh button
if st.button("🔄 Manual Refresh"):
    st.rerun()


st.markdown("---")


# ==================================================================
# 🧩 CREATE MODEL
# ==================================================================
st.subheader("🧩 Create New Model")

with st.form("create_model_form"):
    input_dim   = st.number_input("Input Dim", value=4, min_value=1)
    hidden      = st.text_input("Hidden Layers (comma sep)", "8,8")
    output_dim  = st.number_input("Output Dim", value=1, min_value=1)

    create_btn = st.form_submit_button("Create Model")

    if create_btn:
        try:
            hlist = [int(x.strip()) for x in hidden.split(",") if x.strip()]
            payload = {
                "input_dim": input_dim,
                "hidden_layers": hlist,
                "output_dim": output_dim
            }

            r = safe_post(f"{BASE}/create_model", payload)
            if r and r.status_code == 200:
                st.success("Model created successfully")
            else:
                st.error(f"Failed: {r.status_code} {r.text}")

        except Exception as e:
            st.error(f"Invalid input: {e}")

st.markdown("---")


# ==================================================================
# 🧑‍💻 WORKERS + DATASET UPLOAD
# ==================================================================
st.subheader("🧑‍💻 Workers & Dataset")

status = safe_get(f"{BASE}/sync_status") or {"workers": {}}
workers = list(status.get("workers", {}).keys())

if not workers:
    st.warning("No workers connected yet")
    selected_worker = "worker1"
else:
    selected_worker = st.selectbox("Select Worker", workers)

ds_text = st.text_area("Dataset JSON", height=200,
                       placeholder='[{"x":[1,2,3,4], "y":[0]}]')

if st.button("Upload Dataset to Worker"):
    try:
        ds = json.loads(ds_text)

        if not isinstance(ds, list):
            raise ValueError("Dataset must be a list of objects")

        for item in ds:
            if not isinstance(item, dict) or "x" not in item or "y" not in item:
                raise ValueError("Each item must contain 'x' and 'y' arrays")

        payload = {"worker_id": selected_worker, "data": ds}
        r = safe_post(f"{BASE}/upload_dataset", payload)

        if r and r.status_code == 200:
            st.success(f"Dataset uploaded to {selected_worker}")
        else:
            st.error(f"Upload failed: {r.status_code} {r.text}")

    except Exception as e:
        st.error(f"Invalid dataset JSON: {e}")

st.markdown("---")


# ==================================================================
# ⚙️ TRAINING PARAMETERS
# ==================================================================
st.subheader("⚙️ Training Parameters")

p1, p2 = st.columns(2)
with p1:
    stop_loss  = st.number_input("Stop Loss", value=0.01)
with p2:
    max_epochs = st.number_input("Max Epochs", value=20)

if st.button("Update Training Params"):
    payload = {
        "stop_loss": float(stop_loss),
        "max_epochs": int(max_epochs)
    }
    r = safe_post(f"{BASE}/set_training_params", payload)
    if r and r.status_code == 200:
        st.success("Params updated")
    else:
        st.error(f"Failed: {r.status_code} {r.text}")

st.markdown("---")


# ==================================================================
# 🚀 START TRAINING
# ==================================================================
st.subheader("🚀 Start Training")

if st.button("Start Training Now"):
    # First check if model exists
    model_check = requests.get(f"{BASE}/get_model", headers=HEADERS)
    if model_check.status_code != 200:
        st.error("❌ Create model before starting training!")
    else:
        r = safe_post(f"{BASE}/start_training")
        if r and r.status_code == 200:
            st.success("Training started!")
        else:
            st.error(f"Failed: {r.status_code} {r.text}")

st.markdown("---")


# ==================================================================
# 📥 DOWNLOAD MODEL
# ==================================================================
st.subheader("📥 Download Current Model")

if st.button("Download model.json"):
    r = requests.get(f"{BASE}/download_model", headers=HEADERS)
    if r.status_code == 200:
        st.download_button("Save model.json", r.content,
                           "model.json", "application/json")
    else:
        st.error("No model available")

st.markdown("---")


# ==================================================================
# 📡 LIVE WORKER STATUS
# ==================================================================
st.subheader("📡 Workers (Live)")

status = safe_get(f"{BASE}/sync_status") or {"workers": {}}
workers = status.get("workers", {})

if workers:
    for wid, meta in workers.items():
        st.write(
            f"🔹 **{wid}** — "
            f"updates: {meta.get('updates')} | "
            f"state: {meta.get('state')} | "
            f"last seen: {human(meta.get('last_seen',0))}"
        )
else:
    st.write("❌ No workers registered")

