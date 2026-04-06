import json
import streamlit as st
from huggingface_hub import HfApi
from utils.logger import get_logger

log = get_logger("chat_store")

REPO_ID = st.secrets.get("REPO_ID")
FILE    = st.secrets.get("FILE")

def _get_write_api() -> HfApi:
    return HfApi(token=st.secrets["HF_TOKEN_WRITE"])

def load_chat_history() -> dict:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILE,
            repo_type="dataset",
            token=st.secrets["HF_TOKEN"],
            force_download=True,  # always get latest
        )
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load chat history: {e}")
        return {}

def save_chat_history(sessions: dict) -> None:
    try:
        api = _get_write_api()
        data = json.dumps(sessions, indent=2)
        api.upload_file(
            path_or_fileobj=data.encode(),
            path_in_repo=FILE,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    except Exception as e:
        log.error(f"Could not save chat history: {e}")