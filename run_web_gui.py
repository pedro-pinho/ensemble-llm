#!/usr/bin/env python3
"""Launch the Ensemble LLM Web GUI"""

import uvicorn
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ensemble_llm.web_server import app, create_html_file

if __name__ == "__main__":
    # Ensure HTML file exists
    create_html_file()

    # Run the server
    print("\n" + "=" * 60)
    print("🚀 Starting Ensemble LLM Web GUI")
    print("=" * 60)
    print("📍 URL: http://localhost:8000")
    print("📝 Logs: http://localhost:8000 (sidebar)")
    print("🔄 WebSocket: ws://localhost:8000/ws")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=False)
