"""UI for the GitLab Handbook Chatbot."""

import sys
import os
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.chain import RAGChain
from rag.retriever import Retriever
from vectorstore.loader import load_vectorstore
from config.settings import APP_TITLE, APP_ICON, APP_SUBTITLE
from utils.chat_store import load_chat_history, save_chat_history
from utils.logger import get_logger
from datetime import datetime
import uuid
import json
import base64
import zlib

log = get_logger("ui")

def create_new_session() -> str:
    session_id = str(uuid.uuid4())
    return session_id

def encode_chat(chat_data: dict) -> str:
    payload = json.dumps(
        {"name": chat_data.get("name", "Shared Chat"),
         "messages": chat_data.get("messages", [])},
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(
        zlib.compress(payload.encode("utf-8"), level=9)
    ).decode("ascii")

def decode_chat(token: str) -> dict | None:
    try:
        return json.loads(zlib.decompress(base64.urlsafe_b64decode(token)))
    except Exception as e:
        log.warning(f"Could not decode share token: {e}")
        return None
    
def build_share_url(chat_data: dict) -> str:
    base = ""
    try:
        base = st.secrets.get("APP_BASE_URL", "").rstrip("/")
    except Exception:
        pass
    if not base:
        base = os.environ.get("APP_BASE_URL", "").rstrip("/")
    if not base:
        try:
            host = st.context.headers.get("host", "")
            if host:
                scheme = "https" if "streamlit.app" in host else "http"
                base = f"{scheme}://{host}"
        except Exception:
            pass
    if not base:
        log.warning("APP_BASE_URL not set, share links may not work correctly")
 
    return f"{base}?share={encode_chat(chat_data)}"


def extract_chat_name(message: str, max_length: int = 15) -> str:
    # Remove extra whitespace
    name = message.strip()
    
    # Pull out the first sentence
    if '?' in name:
        name = name.split('?')[0].strip()
    elif '.' in name:
        name = name.split('.')[0].strip()
    else:
        # Just take the first few words
        words = name.split()[:3]
        name = ' '.join(words).strip()
    
    # Make sure it's not too long
    if len(name) > max_length:
        name = name[:max_length-3] + '...'
    
    return name.strip()

# Page setup
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
st.markdown(
    """
    <style>
    .stChatMessage {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .error-box {
        background-color: #ffe6e6;
        border-left: 4px solid #d62728;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_share_token = st.query_params.get("share", None)
 
if _share_token and "shared_view_loaded" not in st.session_state:
    decoded = decode_chat(_share_token)
    if decoded:
        st.session_state.shared_view = decoded
        st.session_state.shared_view_loaded = True
    else:
        st.warning("This share link is invalid or corrupted.")
 
if st.session_state.get("shared_view"):
    chat = st.session_state.shared_view
    st.title(f"{chat.get('name', 'Shared Chat')}")
    st.caption("Read-only shared view")
    st.divider()
    for msg in chat.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.divider()
    if st.button("Open your own chat"):
        del st.session_state["shared_view"]
        del st.session_state["shared_view_loaded"]
        st.query_params.clear()
        st.rerun()
    st.stop()


# Set up state for tracking messages, chats, and the AI model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sessions" not in st.session_state:
    st.session_state.sessions = load_chat_history()

if "current_chat_id" not in st.session_state:
    if st.session_state.sessions:
        latest_id = max(
            st.session_state.sessions.keys(),
            key=lambda x: st.session_state.sessions[x].get("updated_at", "")
        )
        st.session_state.current_chat_id = latest_id
        st.session_state.messages = st.session_state.sessions[latest_id].get("messages", [])
    else:
        st.session_state.current_chat_id = create_new_session()
        st.session_state.sessions[st.session_state.current_chat_id] = {
            "name": "New Chat",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
if "chain" not in st.session_state:
    try:
        vectorstore = load_vectorstore()  
        retriever = Retriever(vectorstore)
        st.session_state.chain = RAGChain(retriever)
        log.info("RAG Chain initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize RAGChain: {e}")
        st.session_state.chain = None

# Set up search options
top_k = 3
search_type = "mmr"

with st.sidebar:
    st.title("Chat")

    if st.button("+ New Chat", use_container_width=True):
        chat_id = create_new_session()
        st.session_state.sessions[chat_id] = {
            'name': 'New Chat',
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        }
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = []
        save_chat_history(st.session_state.sessions)
        st.rerun()

    st.divider()
    st.subheader("History")

    if not st.session_state.sessions:
        st.caption("No chats yet")
    else:
        # Show all their chats sorted by most recent
        for chat_id, chat_data in sorted(
            st.session_state.sessions.items(),
            key=lambda x: x[1].get('updated_at', ''),
            reverse=True
        ):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                chat_name = chat_data.get('name', 'Unnamed')
                msg_count = len(chat_data.get('messages', []))
                is_active = chat_id == st.session_state.current_chat_id
                
                btn_label = f"◉ {chat_name}" if is_active else chat_name
                if st.button(btn_label, key=f"chat_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = chat_data.get('messages', [])
                    st.rerun()

            with col2:
                if st.button("✎", key=f"rename_{chat_id}", help="Rename", use_container_width=True):
                    st.session_state.rename_chat_id = chat_id
                    st.rerun()

            with col3:
                if st.button("X", key=f"delete_{chat_id}", help="Delete", use_container_width=True):
                    del st.session_state.sessions[chat_id]
                    save_chat_history(st.session_state.sessions)
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = create_new_session()
                        st.session_state.sessions[st.session_state.current_chat_id] = {
                            'name': 'New Chat',
                            'messages': [],
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat(),
                        }
                        st.session_state.messages = []
                        save_chat_history(st.session_state.sessions)
                    st.rerun()

        # Show rename form if user clicked rename
        if st.session_state.get("rename_chat_id"):
            chat_id = st.session_state.rename_chat_id
            current_name = st.session_state.sessions[chat_id].get('name', 'Unnamed')
            st.divider()
            new_name = st.text_input(
                "New chat name:",
                value=current_name,
                key="rename_input"
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", use_container_width=True):
                    if new_name.strip():
                        st.session_state.sessions[chat_id]['name'] = new_name
                        st.session_state.rename_chat_id = None
                        save_chat_history(st.session_state.sessions)
                        st.rerun()
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.rename_chat_id = None
                    st.rerun()

        st.divider()
        st.subheader("Share Chat")
        
        if st.session_state.messages:
            if st.button("Share Chat", use_container_width=True):
                current_chat_data = st.session_state.sessions[st.session_state.current_chat_id]
                st.session_state.show_share_url = build_share_url(current_chat_data)
                st.rerun()
            
            if st.session_state.get("show_share_url"):
                st.success("Chat shared!")
                st.code(st.session_state.show_share_url, language="text")
                st.caption("Share this URL with others")
                
                if st.button("Close", use_container_width=True, key="close_share"):
                    st.session_state.show_share_url = None
                    st.rerun()
        else:
            st.caption("No messages to share yet")

# Display the main chat area
st.title(APP_TITLE)

st.markdown(APP_SUBTITLE)

# Display all the messages in the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input and generate responses
if st.session_state.chain is None:
    st.error(
        "**Initialization Error**\n\n"
        "Failed to load the RAG chain. Please check:\n"
        "1. ChromaDB is populated (run `python ingest.py`)\n"
        "2. HF_TOKEN is set in `.env`\n"
        "3. The model is accessible on HuggingFace"
    )
else:
    prompt = st.chat_input("What would you like to know about GitLab?")

    if prompt:
        # Add their message to the conversation
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Use the first message as the chat name
        if len(st.session_state.messages) == 1:
            chat_name = extract_chat_name(prompt)
            st.session_state.sessions[st.session_state.current_chat_id]['name'] = chat_name
        
        # Save the updated chat
        st.session_state.sessions[st.session_state.current_chat_id]['messages'] = st.session_state.messages
        st.session_state.sessions[st.session_state.current_chat_id]['updated_at'] = datetime.now().isoformat()
        save_chat_history(st.session_state.sessions)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            chunks = None

            try:
                # Stream the response from the AI as it generates
                for token, chunk_list in st.session_state.chain.stream(
                    query=prompt,
                    top_k=top_k,
                    search_type=search_type,
                ):
                    if token is not None:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")

                    if chunk_list is not None:
                        chunks = chunk_list

                # Show the complete response
                message_placeholder.markdown(full_response)

            except Exception as e:
                log.error(f"Error during response generation: {e}")
                error_msg = f"Error: {str(e)}"
                message_placeholder.markdown(error_msg)
                full_response = error_msg

        # Save the response to conversation history
        message_to_save = {
            "role": "assistant",
            "content": full_response.replace("▌", ""),
        }
        st.session_state.messages.append(message_to_save)
        st.session_state.sessions[st.session_state.current_chat_id]['messages'] = st.session_state.messages
        st.session_state.sessions[st.session_state.current_chat_id]['updated_at'] = datetime.now().isoformat()
        save_chat_history(st.session_state.sessions)
