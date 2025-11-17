# front.py
import streamlit as st
import requests, json, os
from dotenv import load_dotenv
load_dotenv("project.env")

HTTP_ADDR = os.getenv("HTTP_ADDR")
API_KEY = os.getenv("API_KEY")
BASE = f"http://{HTTP_ADDR}"
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

st.set_page_config(page_title="Swarm UI", layout="wide")
st.title("Decentralised AI Swarm — UI")

st.sidebar.header("Server")
st.sidebar.write(BASE)

# Create model
st.header("Create model")
input_dim = st.number_input("input_dim", value=4, min_value=1)
hidden = st.text_input("hidden layers (comma separated)", value="8")
output_dim = st.number_input("output_dim", value=1, min_value=1)
if st.button("Create model"):
    hlist = [int(x.strip()) for x in hidden.split(",") if x.strip()]
    r = requests.post(f"{BASE}/create_model", headers=HEADERS, json={"input_dim": input_dim, "hidden_layers": hlist, "output_dim": output_dim})
    st.write(r.text)

# Upload dataset
st.header("Upload dataset for worker")
wid = st.text_input("worker id", value="worker1")
ds_text = st.text_area("dataset JSON (list of {\"x\":[],\"y\":[]})", height=200)
if st.button("Upload dataset"):
    try:
        ds = json.loads(ds_text)
    except Exception as e:
        st.error(f"invalid json: {e}")
        ds = None
    if ds is not None:
        r = requests.post(f"{BASE}/upload_dataset", headers=HEADERS, json={"worker_id": wid, "data": ds})
        st.write(r.text)

# Set params
st.header("Training params")
stop_loss = st.number_input("stop_loss", value=float(os.getenv("STOP_LOSS", 0.01)))
max_epochs = st.number_input("max_epochs", value=int(os.getenv("MAX_EPOCHS", 20)))
if st.button("Set params"):
    r = requests.post(f"{BASE}/set_training_params", headers=HEADERS, json={"stop_loss": float(stop_loss), "max_epochs": int(max_epochs)})
    st.write(r.text)

# Start training
st.header("Start training")
if st.button("Start"):
    r = requests.post(f"{BASE}/start_training", headers=HEADERS)
    st.write(r.text)

# Inspect
st.header("Inspect")
if st.button("Show sync_status"):
    r = requests.get(f"{BASE}/sync_status", headers=HEADERS)
    st.json(r.json())
if st.button("Show get_model"):
    r = requests.get(f"{BASE}/get_model", headers=HEADERS)
    if r.status_code == 200:
        st.json(r.json())
    else:
        st.write("No model yet")
