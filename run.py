import argparse
import subprocess
import sys
import os

from config.settings import CHROMA_PATH, APP_TITLE
from utils.logger import get_logger

log = get_logger("run")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Launch the {APP_TITLE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                 
  python run.py --port 8080    
  python run.py --check         
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run health check only and exit without launching UI",
    )
    return parser.parse_args()

def _resolve_hf_token() -> str:
    try:
        import streamlit as st
        tok = st.secrets.get("HF_TOKEN", "")
        if tok:
            return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "")

def _check_chroma() -> bool:
    if not os.path.exists(CHROMA_PATH):
        log.error(
            f"ChromaDB not found at: {CHROMA_PATH}\n"
            "Run 'python ingest.py' first to index the handbook."
        )
        return False

    try:
        from vectorstore.store import health_check
        from vectorstore.loader import load_vectorstore
        vectorstore = load_vectorstore()
        health      = health_check(vectorstore)

        if health["status"] == "ok":
            log.info(f"ChromaDB OK - {health['chunk_count']:,} chunks ready")
            return True
        elif health["status"] == "empty":
            log.warning(
                "ChromaDB exists but is empty.\n"
                "Run 'python ingest.py' to index the handbook."
            )
            return False
        else:
            log.error(f"ChromaDB error: {health.get('error', 'unknown')}")
            return False

    except Exception as e:
        log.error(f"Health check failed: {e}")
        return False


def _check_env() -> bool:
    from config.settings import LLM_PROVIDER

    if LLM_PROVIDER == "huggingface":
        if not _resolve_hf_token():
            log.error(
                "HF_TOKEN is not set in .env\n"
                "Get a free token at https://huggingface.co/settings/tokens"
            )
            return False
        log.info(f"LLM: HuggingFace Inference API")
    else:
        log.error(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'\n"
            "Set LLM_PROVIDER to 'huggingface' in .env"
        )
        return False

    return True

def _launch(port: int) -> None:
    """Launch the Streamlit app"""
    ui_path = os.path.join(os.path.dirname(__file__), "app", "ui.py")

    if not os.path.exists(ui_path):
        log.error(f"UI file not found: {ui_path}")
        sys.exit(1)

    log.info("=" * 55)
    log.info(f"  {APP_TITLE}")
    log.info(f"  http://localhost:{port}")
    log.info("=" * 55)

    cmd = [
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    except subprocess.CalledProcessError as e:
        log.error(f"Streamlit exited with error: {e}")
        sys.exit(1)

def main():
    args = _parse_args()

    log.info("=" * 55)
    log.info(f"  {APP_TITLE} - Pre-flight checks")
    log.info("=" * 55)

    if not _check_env():
        sys.exit(1)

    chroma_ok = _check_chroma()

    if args.check:
        if chroma_ok:
            log.info("All checks passed")
            sys.exit(0)
        else:
            sys.exit(1)

    if not chroma_ok:
        log.warning(
            "Launching UI anyway - you will see an empty knowledge base warning.\n"
            "Run 'python ingest.py' to populate it."
        )

    # Launch
    _launch(args.port)


if __name__ == "__main__":
    main()