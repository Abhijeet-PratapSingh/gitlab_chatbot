import os
import streamlit as st
from huggingface_hub import snapshot_download
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBED_MODEL

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    local_dir = "/tmp/chroma_db"
    chroma_file = os.path.join(local_dir, "chroma.sqlite3")

    if not os.path.exists(chroma_file):
        with st.spinner("Downloading vectors, please wait..."):
            snapshot_download(
                repo_id="really-stupid/gitlab-chroma-db",
                repo_type="dataset",
                local_dir=local_dir,
                token=st.secrets.get("HF_TOKEN"),
            )

    vs = Chroma(
        persist_directory=local_dir,
        collection_name="gitlab_handbook",
        collection_metadata={"hnsw:space": "cosine"},
        embedding_function=get_embeddings(),
    )
    return vs