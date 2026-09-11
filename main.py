# main.py
# Entry point for local development and HuggingFace Spaces

import uvicorn
from src.app import app  # noqa: F401 — imported for Spaces auto-detection

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=7860,        # 7860 is required for HuggingFace Spaces
        reload=True,      # auto-reload on code changes during development
        reload_dirs=["src", "templates"]
    )