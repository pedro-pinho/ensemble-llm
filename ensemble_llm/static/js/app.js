// Ensemble LLM Web Application
function ensembleApp() {
    return {
        // WebSocket connection
        ws: null,
        sessionId: null,
        reconnectAttempts: 0,
        maxReconnectAttempts: 5,
        reconnectDelay: 3000,
        
        // UI State
        currentQuery: '',
        isProcessing: false,
        webSearch: false,
        speedMode: 'balanced',
        
        // Data
        chatHistory: [],
        logs: [],
        
        // Initialize the application
        async init() {
            console.log('Initializing Ensemble App...');
            
            try {
                // Get or create session
                const response = await fetch('/api/session', {
                    credentials: 'include'
                });
                
                if (!response.ok) {
                    throw new Error('Failed to initialize session');
                }
                
                const data = await response.json();
                
                this.sessionId = data.session_id;
                this.chatHistory = data.chat_history || [];
                this.webSearch = data.settings?.web_search || false;
                this.speedMode = data.settings?.speed_mode || 'balanced';
                
                // Connect WebSocket
                this.connectWebSocket();
                
                // Auto-scroll to bottom after DOM update
                this.$nextTick(() => {
                    this.scrollToBottom();
                });

                // Load memory stats
                this.loadMemoryStats();
                
                // Focus on input field
                document.getElementById('query-input')?.focus();
                
            } catch (error) {
                console.error('Failed to initialize:', error);
                this.showError('Failed to initialize application. Please refresh the page.');
            }
        },
        
        // WebSocket connection management
        connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
            
            console.log('Connecting to WebSocket:', wsUrl);
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                
                // Send ping every 30 seconds to keep connection alive
                this.pingInterval = setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({type: 'ping'}));
                    }
                }, 30000);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                
                // Clear ping interval
                if (this.pingInterval) {
                    clearInterval(this.pingInterval);
                    this.pingInterval = null;
                }
                
                // Attempt reconnection
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        },
        
        // Reconnection logic
        attemptReconnect() {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                
                setTimeout(() => {
                    this.connectWebSocket();
                }, this.reconnectDelay);
            } else {
                this.showError('Lost connection to server. Please refresh the page.');
            }
        },
        
        // Handle incoming WebSocket messages
        handleWebSocketMessage(message) {
            switch (message.type) {
                case 'session_init':
                    console.log('Session initialized:', message.data.session_id);
                    break;
                    
                case 'query_start':
                case 'model_response':
                case 'voting':
                case 'error':
                    // Add to logs
                    this.logs.push(message);
                    
                    // Auto-scroll logs
                    this.$nextTick(() => {
                        const logsContainer = this.$refs.logsContainer;
                        if (logsContainer) {
                            logsContainer.scrollTop = logsContainer.scrollHeight;
                        }
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
                    
                    // Focus back on input
                    document.getElementById('query-input')?.focus();
                    break;
                    
                case 'pong':
                    // Server acknowledged ping
                    break;
                    
                default:
                    console.warn('Unknown message type:', message.type);
            }
        },
        
        // Send query to server
        async sendQuery() {
            if (!this.currentQuery.trim() || this.isProcessing) return;
            
            // Check WebSocket connection
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                this.showError('Not connected to server. Attempting to reconnect...');
                this.connectWebSocket();
                return;
            }
            
            this.isProcessing = true;
            
            // Send query via WebSocket
            const message = {
                type: 'query',
                query: this.currentQuery,
                web_search: this.webSearch,
                speed_mode: this.speedMode
            };
            
            this.ws.send(JSON.stringify(message));
            
            // Scroll to bottom to show loading indicator
            this.$nextTick(() => {
                this.scrollToBottom();
            });
        },
        
        // Reset session and clear cache
        async resetSession() {
            if (!confirm('Reset session and clear cache? This will delete all chat history.')) {
                return;
            }
            
            try {
                const response = await fetch('/api/reset', {
                    method: 'POST',
                    credentials: 'include'
                });
                
                if (response.ok) {
                    // Clear local state
                    this.chatHistory = [];
                    this.logs = [];
                    this.currentQuery = '';
                    
                    // Show success message
                    this.showSuccess('Session reset successfully');
                    
                    // Reload after a short delay
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                } else {
                    throw new Error('Failed to reset session');
                }
            } catch (error) {
                console.error('Failed to reset session:', error);
                this.showError('Failed to reset session. Please try again.');
            }
        },
        
        // Utility functions
        formatTime(timestamp) {
            const date = new Date(timestamp);
            return date.toLocaleTimeString();
        },
        
        getModelClass(model) {
            const modelName = model.toLowerCase();
            
            if (modelName.includes('llama') && !modelName.includes('tiny')) {
                return 'model-llama';
            } else if (modelName.includes('tinyllama')) {
                return 'model-tinyllama';
            } else if (modelName.includes('phi')) {
                return 'model-phi';
            } else if (modelName.includes('qwen')) {
                return 'model-qwen';
            } else if (modelName.includes('mistral')) {
                return 'model-mistral';
            } else if (modelName.includes('gemma')) {
                return 'model-gemma';
            } else if (modelName.includes('codellama')) {
                return 'model-codellama';
            }
            
            return '';
        },
        
        scrollToBottom() {
            const chatContainer = this.$refs.chatContainer;
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        },
        
        showError(message) {
            // Add error to logs
            this.logs.push({
                type: 'error',
                timestamp: new Date().toISOString(),
                data: { message }
            });
        },
        
        showSuccess(message) {
            // Add success message to logs
            this.logs.push({
                type: 'success',
                timestamp: new Date().toISOString(),
                data: { message }
            });
        },
        
        // Settings update
        async updateSettings() {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'update_settings',
                    settings: {
                        web_search: this.webSearch,
                        speed_mode: this.speedMode
                    }
                }));
            }
        },
        
        // Watch for settings changes
        $watch: {
            webSearch() {
                this.updateSettings();
            },
            speedMode() {
                this.updateSettings();
            }
        },

        async loadMemoryStats() {
            try {
                const response = await fetch('/api/memory/stats');
                if (response.ok) {
                    const stats = await response.json();
                    console.log('Memory stats:', stats);
                    // Display stats in UI if desired
                }
            } catch (error) {
                console.error('Failed to load memory stats:', error);
            }
        }
    };
}

// Add keyboard shortcuts
document.addEventListener('DOMContentLoaded', () => {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to send
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const input = document.getElementById('query-input');
            if (input && input.value) {
                input.form?.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to focus input
        if (e.key === 'Escape') {
            document.getElementById('query-input')?.focus();
        }
    });
});