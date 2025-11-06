"""FastAPI web server for Ensemble LLM GUI"""

import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from collections import defaultdict

from .main import EnsembleLLM
from .config import DEFAULT_MODELS
from .learning_system import CacheManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnsembleLLM.WebServer")

# Get the static directory path
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Ensure subdirectories exist
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)

# Session storage (in production, use Redis or similar)
sessions: Dict[str, "SessionData"] = {}
active_websockets: Dict[str, List[WebSocket]] = defaultdict(list)


class QueryRequest(BaseModel):
    query: str
    web_search: bool = False
    speed_mode: str = "balanced"


class SessionData:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.chat_history = []
        self.logs_history = []
        self.settings = {"web_search": False, "speed_mode": "balanced"}
        self.ensemble = None
        self.is_processing = False

    def add_to_history(self, query: str, response: str, metadata: Dict):
        self.chat_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "metadata": metadata,
            }
        )
        self.last_activity = datetime.now()

    def add_log_entry(self, entry: Dict):
        self.logs_history.append({"timestamp": datetime.now().isoformat(), **entry})
        # Keep only last 100 log entries per session
        if len(self.logs_history) > 100:
            self.logs_history = self.logs_history[-100:]

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "chat_history": self.chat_history,
            "settings": self.settings,
        }


class LogStreamer:
    """Stream logs to WebSocket clients"""

    def __init__(self, websocket: WebSocket, session: SessionData):
        self.websocket = websocket
        self.session = session
        self.model_colors = {
            "llama3.2:3b": "#3B82F6",  # Blue
            "llama3.2:1b": "#60A5FA",  # Light Blue
            "phi3.5:latest": "#10B981",  # Emerald
            "phi3.5": "#10B981",  # Emerald
            "qwen2.5:7b": "#F59E0B",  # Amber
            "mistral:7b-instruct-q4_K_M": "#EF4444",  # Red
            "mistral:7b": "#EF4444",  # Red
            "gemma2:2b": "#8B5CF6",  # Violet
            "gemma2:9b": "#7C3AED",  # Purple
            "tinyllama:1b": "#EC4899",  # Pink
            "codellama:7b": "#14B8A6",  # Teal
            "default": "#6B7280",  # Gray
        }

    async def send_log(self, log_type: str, data: Dict):
        """Send log entry to WebSocket"""

        # Add color for model-specific logs
        if "model" in data:
            data["color"] = self.model_colors.get(
                data["model"], self.model_colors["default"]
            )

        message = {
            "type": log_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        try:
            await self.websocket.send_json(message)
            # Also store in session history
            self.session.add_log_entry(message)
        except Exception as e:
            logger.error(f"Failed to send log: {str(e)}")

    async def send_query_start(self, query: str, models: List[str]):
        await self.send_log("query_start", {"query": query, "models": models})

    async def send_model_response(
        self, model: str, response: str, success: bool, response_time: float
    ):
        await self.send_log(
            "model_response",
            {
                "model": model,
                "response": response,  # Full response, no truncation
                "success": success,
                "response_time": response_time,
            },
        )

    async def send_voting_details(self, voting_data: Dict):
        await self.send_log(
            "voting",
            {
                "total_models": voting_data.get("total_models", 0),
                "successful_models": voting_data.get("successful_models", 0),
                "selected_model": voting_data.get("selected_model"),
                "scores": voting_data.get("all_scores", {}),
            },
        )

    async def send_final_response(self, response: str, metadata: Dict):
        await self.send_log(
            "final_response",
            {
                "response": response,
                "selected_model": metadata.get("selected_model"),
                "total_time": metadata.get("total_ensemble_time", 0),
            },
        )


# Initialize FastAPI app
app = FastAPI(title="Ensemble LLM Council")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_or_create_session(session_id: Optional[str]) -> SessionData:
    """Get existing session or create new one"""

    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = SessionData(session_id)
        logger.info(f"Created new session: {session_id}")

    return sessions[session_id]


def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    now = datetime.now()
    expired = []

    for sid, session in sessions.items():
        if now - session.last_activity > timedelta(hours=24):
            expired.append(sid)

    for sid in expired:
        if sid in sessions:
            del sessions[sid]
            logger.info(f"Cleaned up expired session: {sid}")


@app.on_event("startup")
async def startup_event():
    """Initialize the server"""
    logger.info("Starting Ensemble LLM Web Server...")

    # Periodic session cleanup
    async def cleanup_task():
        while True:
            await asyncio.sleep(3600)  # Every hour
            cleanup_old_sessions()

    asyncio.create_task(cleanup_task())


@app.get("/")
async def root():
    """Serve the main HTML page"""
    html_path = STATIC_DIR / "index.html"

    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Error: index.html not found. Please check the static directory.</h1>",
            status_code=500,
        )

    return FileResponse(html_path)


@app.get("/api/session")
async def get_session(request: Request):
    """Get or create session"""
    session_id = request.cookies.get("session_id")
    session = get_or_create_session(session_id)

    response = JSONResponse(
        {
            "session_id": session.session_id,
            "settings": session.settings,
            "chat_history": session.chat_history,
        }
    )

    if not session_id:
        response.set_cookie(
            "session_id",
            session.session_id,
            max_age=86400,
            httponly=True,
            samesite="lax",
        )

    return response


@app.post("/api/reset")
async def reset_session(request: Request):
    """Reset session and clear cache"""
    session_id = request.cookies.get("session_id")

    if session_id and session_id in sessions:
        # Clear the session
        sessions[session_id] = SessionData(session_id)

        # Clear cache
        try:
            CacheManager.clear_all_cache()
            logger.info(f"Reset session and cleared cache for: {session_id}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")

    return {"status": "success", "message": "Session reset and cache cleared"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""

    await websocket.accept()

    # Get or create session
    session = get_or_create_session(session_id)

    # Add websocket to active connections
    active_websockets[session_id].append(websocket)

    # Create log streamer
    log_streamer = LogStreamer(websocket, session)

    # Send initial session data
    await websocket.send_json(
        {
            "type": "session_init",
            "data": {
                "session_id": session.session_id,
                "chat_history": session.chat_history,
                "settings": session.settings,
            },
        }
    )

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data["type"] == "query":
                # Process query
                await process_query(
                    session=session,
                    query=data["query"],
                    web_search=data.get("web_search", False),
                    speed_mode=data.get("speed_mode", "balanced"),
                    log_streamer=log_streamer,
                )

            elif data["type"] == "update_settings":
                # Update session settings
                session.settings.update(data.get("settings", {}))
                await websocket.send_json(
                    {"type": "settings_updated", "data": session.settings}
                )

            elif data["type"] == "ping":
                # Keep connection alive
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        # Remove from active websockets
        if websocket in active_websockets[session_id]:
            active_websockets[session_id].remove(websocket)
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_websockets[session_id]:
            active_websockets[session_id].remove(websocket)


async def process_query(
    session: SessionData,
    query: str,
    web_search: bool,
    speed_mode: str,
    log_streamer: LogStreamer,
):
    """Process a query and stream results"""

    if session.is_processing:
        await log_streamer.send_log(
            "error", {"message": "Already processing a query. Please wait..."}
        )
        return

    session.is_processing = True

    try:
        # Initialize ensemble if needed
        if not session.ensemble:
            session.ensemble = EnsembleLLM(
                models=DEFAULT_MODELS[:3],  # Use first 3 models
                enable_web_search=web_search,
                speed_mode=speed_mode,
                smart_learning=True,
                verbose_logging=False,  # We handle logging ourselves
            )
            await session.ensemble.initialize()

        # Update ensemble settings
        session.ensemble.enable_web_search = web_search
        session.ensemble.speed_mode = speed_mode

        # Send query start
        await log_streamer.send_query_start(query, session.ensemble.models)

        # Create a custom response handler to stream logs
        original_query_method = session.ensemble.query_all_models_optimized

        async def streaming_query_method(prompt: str) -> List[Dict]:
            """Wrapper to stream individual model responses"""

            responses = await original_query_method(prompt)

            # Stream each model's response
            for response in responses:
                await log_streamer.send_model_response(
                    model=response.get("model", "unknown"),
                    response=response.get("response", ""),
                    success=response.get("success", False),
                    response_time=response.get("response_time", 0),
                )

            return responses

        # Temporarily replace the query method
        session.ensemble.query_all_models_optimized = streaming_query_method

        # Execute the query
        response, metadata = await session.ensemble.ensemble_query(query, verbose=False)

        # Restore original method
        session.ensemble.query_all_models_optimized = original_query_method

        # Send voting details
        await log_streamer.send_voting_details(metadata)

        # Send final response
        await log_streamer.send_final_response(response, metadata)

        # Add to chat history
        session.add_to_history(query, response, metadata)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        await log_streamer.send_log("error", {"message": f"Error: {str(e)}"})

    finally:
        session.is_processing = False


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "sessions": len(sessions)}


@app.get("/api/memory/stats")
async def get_memory_stats(request: Request):
    """Get memory statistics"""
    session_id = request.cookies.get("session_id")
    session = get_or_create_session(session_id)

    if session.ensemble and session.ensemble.memory_manager:
        stats = session.ensemble.memory_manager.get_memory_stats()
        return JSONResponse(stats)

    return JSONResponse({"error": "Memory not initialized"})


@app.post("/api/memory/forget")
async def forget_memory(request: Request, category: str = None, key: str = None):
    """Forget specific memories"""
    session_id = request.cookies.get("session_id")
    session = get_or_create_session(session_id)

    if session.ensemble and session.ensemble.memory_manager:
        session.ensemble.memory_manager.semantic_memory.forget(category, key)
        return JSONResponse({"status": "success"})

    return JSONResponse({"error": "Memory not initialized"})
