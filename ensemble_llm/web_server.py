"""FastAPI web server for Ensemble LLM GUI"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
)
from fastapi.responses import HTMLResponse, JSONResponse
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

# Session storage (in production, use Redis or similar)
sessions: Dict[str, Dict] = {}
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
            "phi3.5:latest": "#10B981",  # Emerald
            "qwen2.5:7b": "#F59E0B",  # Amber
            "mistral:7b-instruct-q4_K_M": "#EF4444",  # Red
            "gemma2:2b": "#8B5CF6",  # Violet
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

    # Create static directory if it doesn't exist
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    # Periodic session cleanup
    async def cleanup_task():
        while True:
            await asyncio.sleep(3600)  # Every hour
            cleanup_old_sessions()

    asyncio.create_task(cleanup_task())


@app.get("/")
async def root():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "static" / "index.html"

    if not html_path.exists():
        # Create the HTML file if it doesn't exist
        create_html_file()

    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


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
        response.set_cookie("session_id", session.session_id, max_age=86400)

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
        active_websockets[session_id].remove(websocket)
        logger.info(f"WebSocket disconnected for session: {session_id}")


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
        logger.error(f"Error processing query: {str(e)}")
        await log_streamer.send_log("error", {"message": f"Error: {str(e)}"})

    finally:
        session.is_processing = False


def create_html_file():
    """Create the HTML file with embedded CSS and JavaScript"""

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ensemble LLM Council</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <style>
        /* Custom scrollbar for logs */
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #1f2937;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #4b5563;
            border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #6b7280;
        }
        
        /* Log entry animations */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .log-entry {
            animation: slideIn 0.3s ease-out;
        }
        
        /* Model colors */
        .model-llama { border-left: 4px solid #3B82F6; }
        .model-phi { border-left: 4px solid #10B981; }
        .model-qwen { border-left: 4px solid #F59E0B; }
        .model-mistral { border-left: 4px solid #EF4444; }
        .model-gemma { border-left: 4px solid #8B5CF6; }
        .model-tinyllama { border-left: 4px solid #EC4899; }
        .model-codellama { border-left: 4px solid #14B8A6; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100" x-data="ensembleApp()">
    <div class="flex h-screen">
        <!-- Sidebar for Logs -->
        <div class="w-1/3 bg-gray-800 border-r border-gray-700 flex flex-col">
            <div class="p-4 border-b border-gray-700">
                <h2 class="text-xl font-bold text-gray-100">Voting Logs</h2>
                <p class="text-xs text-gray-400 mt-1">Full model responses & voting details</p>
            </div>
            
            <div class="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-2" x-ref="logsContainer">
                <template x-for="log in logs" :key="log.timestamp">
                    <div class="log-entry">
                        <!-- Query Start -->
                        <template x-if="log.type === 'query_start'">
                            <div class="bg-blue-900/20 rounded p-3 mb-2">
                                <div class="text-xs text-blue-400 mb-1" x-text="formatTime(log.timestamp)"></div>
                                <div class="font-semibold text-blue-300">New Query:</div>
                                <div class="text-sm mt-1" x-text="log.data.query"></div>
                                <div class="text-xs text-gray-400 mt-1">
                                    Models: <span x-text="log.data.models.join(', ')"></span>
                                </div>
                            </div>
                        </template>
                        
                        <!-- Model Response -->
                        <template x-if="log.type === 'model_response'">
                            <div :class="getModelClass(log.data.model)" 
                                 class="bg-gray-800/50 rounded p-3 mb-2">
                                <div class="flex justify-between items-start mb-2">
                                    <div>
                                        <span class="font-semibold" :style="`color: ${log.data.color}`" 
                                              x-text="log.data.model"></span>
                                        <span x-show="log.data.success" class="text-green-400 text-xs ml-2">✓</span>
                                        <span x-show="!log.data.success" class="text-red-400 text-xs ml-2">✗</span>
                                    </div>
                                    <span class="text-xs text-gray-400" 
                                          x-text="`${log.data.response_time.toFixed(2)}s`"></span>
                                </div>
                                <div class="text-sm text-gray-200 whitespace-pre-wrap" 
                                     x-text="log.data.response"></div>
                            </div>
                        </template>
                        
                        <!-- Voting Details -->
                        <template x-if="log.type === 'voting'">
                            <div class="bg-purple-900/20 rounded p-3 mb-2">
                                <div class="font-semibold text-purple-300 mb-2">Voting Results:</div>
                                <div class="text-xs space-y-1">
                                    <div>Total Models: <span x-text="log.data.total_models"></span></div>
                                    <div>Successful: <span x-text="log.data.successful_models"></span></div>
                                    <div class="mt-2 font-semibold">Selected: 
                                        <span class="text-green-400" x-text="log.data.selected_model"></span>
                                    </div>
                                    <div class="mt-2">Scores:</div>
                                    <template x-for="(scores, model) in log.data.scores" :key="model">
                                        <div class="ml-2 text-gray-300">
                                            <span x-text="model"></span>:
                                            <span class="text-yellow-400" x-text="scores.final.toFixed(3)"></span>
                                            (C: <span x-text="scores.consensus.toFixed(2)"></span>,
                                             Q: <span x-text="scores.quality.toFixed(2)"></span>)
                                        </div>
                                    </template>
                                </div>
                            </div>
                        </template>
                        
                        <!-- Final Response -->
                        <template x-if="log.type === 'final_response'">
                            <div class="bg-green-900/20 rounded p-3 mb-2 border border-green-700">
                                <div class="font-semibold text-green-300">Final Answer 
                                    <span class="text-xs">
                                        (via <span x-text="log.data.selected_model"></span>)
                                    </span>
                                </div>
                                <div class="text-xs text-gray-400 mb-1">
                                    Time: <span x-text="`${log.data.total_time.toFixed(2)}s`"></span>
                                </div>
                            </div>
                        </template>
                        
                        <!-- Error -->
                        <template x-if="log.type === 'error'">
                            <div class="bg-red-900/20 rounded p-3 mb-2 border border-red-700">
                                <div class="text-red-400" x-text="log.data.message"></div>
                            </div>
                        </template>
                    </div>
                </template>
            </div>
        </div>
        
        <!-- Main Chat Area -->
        <div class="flex-1 flex flex-col bg-gray-900">
            <!-- Header -->
            <div class="bg-gray-800 border-b border-gray-700 p-4">
                <div class="flex justify-between items-center">
                    <div>
                        <h1 class="text-2xl font-bold text-gray-100">Council of LLMs</h1>
                        <p class="text-sm text-gray-400">Ensemble Intelligence System</p>
                    </div>
                    
                    <!-- Settings -->
                    <div class="flex items-center space-x-4">
                        <!-- Speed Mode -->
                        <div>
                            <label class="text-xs text-gray-400">Speed</label>
                            <select x-model="speedMode" 
                                    class="ml-2 bg-gray-700 text-gray-200 text-sm rounded px-2 py-1">
                                <option value="turbo">Turbo</option>
                                <option value="fast">Fast</option>
                                <option value="balanced">Balanced</option>
                                <option value="quality">Quality</option>
                            </select>
                        </div>
                        
                        <!-- Web Search Toggle -->
                        <label class="flex items-center cursor-pointer">
                            <span class="text-sm mr-2">Web Search</span>
                            <div class="relative">
                                <input type="checkbox" x-model="webSearch" class="sr-only">
                                <div class="w-10 h-5 bg-gray-600 rounded-full shadow-inner"></div>
                                <div class="dot absolute w-4 h-4 bg-gray-300 rounded-full shadow 
                                           -left-1 top-0.5 transition"
                                     :class="{'translate-x-6 bg-green-400': webSearch}"></div>
                            </div>
                        </label>
                        
                        <!-- Reset Button -->
                        <button @click="resetSession()" 
                                class="bg-red-600 hover:bg-red-700 text-white text-sm px-3 py-1 rounded">
                            Reset Session
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Chat Messages -->
            <div class="flex-1 overflow-y-auto p-6" x-ref="chatContainer">
                <template x-for="message in chatHistory" :key="message.timestamp">
                    <div class="mb-6">
                        <!-- User Query -->
                        <div class="flex justify-end mb-4">
                            <div class="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-2xl">
                                <div class="text-sm" x-text="message.query"></div>
                                <div class="text-xs opacity-70 mt-1" x-text="formatTime(message.timestamp)"></div>
                            </div>
                        </div>
                        
                        <!-- AI Response -->
                        <div class="flex justify-start">
                            <div class="bg-gray-700 rounded-lg px-4 py-2 max-w-2xl">
                                <div class="text-sm text-gray-200 whitespace-pre-wrap" 
                                     x-text="message.response"></div>
                                <div class="text-xs text-gray-400 mt-2">
                                    <span x-text="message.metadata.selected_model"></span> •
                                    <span x-text="`${message.metadata.total_ensemble_time?.toFixed(1)}s`"></span>
                                </div>
                            </div>
                        </div>
                    </div>
                </template>
                
                <!-- Loading Indicator -->
                <div x-show="isProcessing" class="flex justify-start mb-4">
                    <div class="bg-gray-700 rounded-lg px-4 py-2">
                        <div class="flex items-center">
                            <svg class="animate-spin h-4 w-4 mr-2" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" 
                                        stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" 
                                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                            </svg>
                            <span class="text-sm">Council is deliberating...</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Input Area -->
            <div class="border-t border-gray-700 p-4 bg-gray-800">
                <form @submit.prevent="sendQuery" class="flex space-x-2">
                    <input type="text" 
                           x-model="currentQuery"
                           :disabled="isProcessing"
                           placeholder="Ask your question to the council..."
                           class="flex-1 bg-gray-700 text-gray-200 rounded-lg px-4 py-2 
                                  focus:outline-none focus:ring-2 focus:ring-blue-500">
                    
                    <button type="submit"
                            :disabled="isProcessing || !currentQuery.trim()"
                            class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 
                                   text-white px-6 py-2 rounded-lg transition">
                        <span x-show="!isProcessing">Send</span>
                        <span x-show="isProcessing">Processing...</span>
                    </button>
                </form>
            </div>
        </div>
    </div>
    
    <script>
        function ensembleApp() {
            return {
                // WebSocket connection
                ws: null,
                sessionId: null,
                
                // UI State
                currentQuery: '',
                isProcessing: false,
                webSearch: false,
                speedMode: 'balanced',
                
                // Data
                chatHistory: [],
                logs: [],
                
                // Initialize
                async init() {
                    // Get or create session
                    const response = await fetch('/api/session', {
                        credentials: 'include'
                    });
                    const data = await response.json();
                    
                    this.sessionId = data.session_id;
                    this.chatHistory = data.chat_history || [];
                    this.webSearch = data.settings?.web_search || false;
                    this.speedMode = data.settings?.speed_mode || 'balanced';
                    
                    // Connect WebSocket
                    this.connectWebSocket();
                    
                    // Auto-scroll to bottom
                    this.$nextTick(() => {
                        this.scrollToBottom();
                    });
                },
                
                connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
                    
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        console.log('WebSocket connected');
                        // Send ping every 30 seconds to keep connection alive
                        setInterval(() => {
                            if (this.ws.readyState === WebSocket.OPEN) {
                                this.ws.send(JSON.stringify({type: 'ping'}));
                            }
                        }, 30000);
                    };
                    
                    this.ws.onmessage = (event) => {
                        const message = JSON.parse(event.data);
                        this.handleWebSocketMessage(message);
                    };
                    
                    this.ws.onclose = () => {
                        console.log('WebSocket disconnected');
                        // Reconnect after 3 seconds
                        setTimeout(() => this.connectWebSocket(), 3000);
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                    };
                },
                
                handleWebSocketMessage(message) {
                    switch (message.type) {
                        case 'session_init':
                            // Session initialized
                            break;
                            
                        case 'query_start':
                        case 'model_response':
                        case 'voting':
                        case 'error':
                            // Add to logs
                            this.logs.push(message);
                            this.$nextTick(() => {
                                this.$refs.logsContainer.scrollTop = this.$refs.logsContainer.scrollHeight;
                            });
                            break;
                            
                        case 'final_response':
                            // Add to logs
                            this.logs.push(message);
                            
                            // Add to chat history
                            this.chatHistory.push({
                                timestamp: new Date().toISOString(),
                                query: this.currentQuery,
                                response: message.data.response,
                                metadata: message.data
                            });
                            
                            // Clear input and stop processing
                            this.currentQuery = '';
                            this.isProcessing = false;
                            
                            // Scroll to bottom
                            this.$nextTick(() => {
                                this.scrollToBottom();
                            });
                            break;
                    }
                },
                
                async sendQuery() {
                    if (!this.currentQuery.trim() || this.isProcessing) return;
                    
                    this.isProcessing = true;
                    
                    // Send query via WebSocket
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({
                            type: 'query',
                            query: this.currentQuery,
                            web_search: this.webSearch,
                            speed_mode: this.speedMode
                        }));
                    } else {
                        console.error('WebSocket not connected');
                        this.isProcessing = false;
                    }
                },
                
                async resetSession() {
                    if (!confirm('Reset session and clear cache? This will delete all chat history.')) {
                        return;
                    }
                    
                    const response = await fetch('/api/reset', {
                        method: 'POST',
                        credentials: 'include'
                    });
                    
                    if (response.ok) {
                        // Clear local state
                        this.chatHistory = [];
                        this.logs = [];
                        this.currentQuery = '';
                        
                        // Reload page to get fresh session
                        window.location.reload();
                    }
                },
                
                formatTime(timestamp) {
                    const date = new Date(timestamp);
                    return date.toLocaleTimeString();
                },
                
                getModelClass(model) {
                    if (model.includes('llama')) return 'model-llama';
                    if (model.includes('phi')) return 'model-phi';
                    if (model.includes('qwen')) return 'model-qwen';
                    if (model.includes('mistral')) return 'model-mistral';
                    if (model.includes('gemma')) return 'model-gemma';
                    if (model.includes('tinyllama')) return 'model-tinyllama';
                    if (model.includes('codellama')) return 'model-codellama';
                    return '';
                },
                
                scrollToBottom() {
                    if (this.$refs.chatContainer) {
                        this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
                    }
                }
            }
        }
    </script>
</body>
</html>"""

    # Save HTML file
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    with open(static_dir / "index.html", "w") as f:
        f.write(html_content)

    logger.info("Created index.html file")
