# front.py
import streamlit as st
import requests, json, os, time
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("project.env")

HTTP_ADDR = os.getenv("HTTP_ADDR")
API_KEY = os.getenv("API_KEY")
BASE = f"http://{HTTP_ADDR}"
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

st.set_page_config(page_title="Swarm UI", layout="wide")
st.title("🧠 Decentralised AI Swarm — UI (Safe)")

st.sidebar.header("Server")
st.sidebar.write(BASE)

def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def human(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts)

# Top controls
st.subheader("Controls")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Refresh Status"):
        st.rerun()
with col2:
    if st.button("Fetch Model Versions"):
        r = requests.get(f"{BASE}/sync_status", headers=HEADERS)
        if r.status_code == 200:
            st.json(r.json())
        else:
            st.error("Failed to fetch status")
with col3:
    if st.button("Start Training"):
        # Validate that model exists first
        r_model = requests.get(f"{BASE}/get_model", headers=HEADERS)
        if r_model.status_code != 200:
            st.error("❌ No model exists. Create a model before starting training.")
        else:
            r = requests.post(f"{BASE}/start_training", headers=HEADERS)
            if r.status_code == 200:
                st.success("Started training")
            else:
                st.error(f"Failed to start training: {r.status_code} {r.text}")

st.markdown("---")

# Create model
st.subheader("Create Model")
with st.form("create_model_form"):
    input_dim = st.number_input("input_dim", value=4, min_value=1)
    hidden = st.text_input("hidden layers (comma separated)", value="8")
    output_dim = st.number_input("output_dim", value=1, min_value=1)
    submitted = st.form_submit_button("Create model")
    if submitted:
        try:
            hlist = [int(x.strip()) for x in hidden.split(",") if x.strip()]
            r = requests.post(f"{BASE}/create_model", headers=HEADERS, json={"input_dim": input_dim, "hidden_layers": hlist, "output_dim": output_dim})
            if r.status_code == 200:
                st.success("Model created")
            else:
                st.error(f"Create model failed: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"Invalid hidden layers: {e}")

st.markdown("---")

# Workers & dataset
st.subheader("Workers & Dataset")
status = safe_get(f"{BASE}/sync_status") or {"workers": {}}
workers = list(status.get("workers", {}).keys())
selected = st.selectbox("Select Worker", workers if workers else ["worker1"])

ds_text = st.text_area("Dataset JSON (list of {\"x\":[],\"y\":[]})", height=200, placeholder='[{"x":[1,2,3,4],"y":[0]}]')
if st.button("Upload dataset to selected worker"):
    try:
        ds = json.loads(ds_text)
        # client-side validation
        if not isinstance(ds, list) or any(not isinstance(s, dict) or "x" not in s or "y" not in s for s in ds):
            st.error("Dataset must be a list of objects with 'x' and 'y' arrays")
        else:
            r = requests.post(f"{BASE}/upload_dataset", headers=HEADERS, json={"worker_id": selected, "data": ds})
            if r.status_code == 200:
                st.success("Dataset uploaded")
            else:
                st.error(f"Upload failed: {r.status_code} {r.text}")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

st.markdown("---")

# Training params
st.subheader("Training Params")
col1, col2 = st.columns(2)
with col1:
    stop_loss = st.number_input("stop_loss", value=float(os.getenv("STOP_LOSS", "0.01")))
with col2:
    max_epochs = st.number_input("max_epochs", value=int(os.getenv("MAX_EPOCHS", "20")))
if st.button("Set params"):
    r = requests.post(f"{BASE}/set_training_params", headers=HEADERS, json={"stop_loss": float(stop_loss), "max_epochs": int(max_epochs)})
    if r.status_code == 200:
        st.success("Params set")
    else:
        st.error(f"Failed to set params: {r.status_code} {r.text}")

st.markdown("---")

# Inspect area
st.subheader("Inspect & Download")
if st.button("Show sync_status"):
    r = requests.get(f"{BASE}/sync_status", headers=HEADERS)
    if r.status_code == 200:
        st.json(r.json())
    else:
        st.error("Failed to fetch sync status")
if st.button("Show current model"):
    r = requests.get(f"{BASE}/get_model", headers=HEADERS)
    if r.status_code == 200:
        st.json(r.json())
    elif r.status_code == 204:
        st.warning("No model created yet")
    else:
        st.error(f"Error: {r.status_code} {r.text}")

# Download model (latest)
if st.button("Download current model"):
    r = requests.get(f"{BASE}/download_model", headers=HEADERS)
    if r.status_code == 200:
        st.download_button("Download model.json", r.content, "model.json", "application/json")
    else:
        st.error("No model available to download")

st.markdown("---")
st.write("Workers (live):")
status = safe_get(f"{BASE}/sync_status") or {"workers": {}}
workers = status.get("workers", {})
if workers:
    for wid, meta in workers.items():
        st.write(f"- **{wid}** updates={meta.get('updates')} state={meta.get('state')} last_seen={human(meta.get('last_seen',0))}")
else:
    st.write("No workers registered")
