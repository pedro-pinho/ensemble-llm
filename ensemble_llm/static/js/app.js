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

        // Document Management
        documents: [],
        showDocumentsPanel: false,
        uploadProgress: 0,
        uploadingFileName: '',
        uploadMessage: '',
        uploadStatus: '', // 'success', 'error', or ''
        dragOver: false,

        // Toast Notifications
        toasts: [],
        
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

                // Load documents
                await this.loadDocuments();

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
        },

        // ========================================
        // Document Management Methods
        // ========================================

        toggleDocumentsPanel() {
            this.showDocumentsPanel = !this.showDocumentsPanel;
            if (this.showDocumentsPanel) {
                this.loadDocuments();
            }
        },

        async loadDocuments() {
            try {
                const response = await fetch('/api/documents', {
                    credentials: 'include'
                });

                if (!response.ok) {
                    console.warn('Documents endpoint returned error, initializing empty list');
                    this.documents = [];
                    return;
                }

                const data = await response.json();
                this.documents = data.documents || [];
                console.log(`Loaded ${this.documents.length} documents`);

                // Show info if there was an error but it was handled
                if (data.error) {
                    console.warn('Document loading had issues:', data.error);
                }
            } catch (error) {
                console.warn('Error loading documents, using empty list:', error);
                this.documents = [];
                // Don't show error toast on initial load - just log it
            }
        },

        handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                this.uploadFile(file);
            }
            // Reset input so same file can be selected again
            event.target.value = '';
        },

        handleFileDrop(event) {
            this.dragOver = false;

            const files = event.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];

                // Validate file type
                const validTypes = ['.pdf', '.docx'];
                const fileExt = '.' + file.name.split('.').pop().toLowerCase();

                if (!validTypes.includes(fileExt)) {
                    this.showToast(`Invalid file type. Please upload PDF or DOCX files.`, 'error');
                    return;
                }

                this.uploadFile(file);
            }
        },

        async uploadFile(file) {
            // Validate file size (50MB limit)
            const maxSize = 50 * 1024 * 1024; // 50MB in bytes
            if (file.size > maxSize) {
                this.uploadStatus = 'error';
                this.uploadMessage = 'File too large. Maximum size is 50MB.';
                this.showToast('File too large (max 50MB)', 'error');
                return;
            }

            // Validate file type
            const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
            const validExtensions = ['.pdf', '.docx'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();

            if (!validExtensions.includes(fileExt) && !validTypes.includes(file.type)) {
                this.uploadStatus = 'error';
                this.uploadMessage = 'Invalid file type. Please upload PDF or DOCX files only.';
                this.showToast('Invalid file type', 'error');
                return;
            }

            // Reset state
            this.uploadProgress = 0;
            this.uploadingFileName = file.name;
            this.uploadMessage = '';
            this.uploadStatus = '';

            console.log(`Starting upload: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);

            try {
                const formData = new FormData();
                formData.append('file', file);

                // Create XMLHttpRequest for progress tracking
                const xhr = new XMLHttpRequest();

                // Track upload progress
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = Math.round((e.loaded / e.total) * 100);
                        this.uploadProgress = percentComplete;
                        console.log(`Upload progress: ${percentComplete}%`);
                    }
                });

                // Handle completion
                xhr.addEventListener('load', async () => {
                    if (xhr.status === 200) {
                        try {
                            const response = JSON.parse(xhr.responseText);

                            this.uploadProgress = 100;
                            this.uploadStatus = 'success';
                            this.uploadMessage = `Successfully uploaded ${file.name}! (${response.total_chunks} chunks, ${response.total_pages} pages)`;

                            console.log('Upload successful:', response);
                            this.showToast(`${file.name} uploaded successfully!`, 'success');

                            // Reload documents
                            await this.loadDocuments();

                            // Clear status after 3 seconds
                            setTimeout(() => {
                                this.uploadProgress = 0;
                                this.uploadMessage = '';
                                this.uploadStatus = '';
                                this.uploadingFileName = '';
                            }, 3000);

                        } catch (error) {
                            console.error('Error parsing response:', error);
                            this.handleUploadError('Failed to process server response');
                        }
                    } else {
                        let errorMessage = 'Upload failed';
                        try {
                            const errorData = JSON.parse(xhr.responseText);
                            errorMessage = errorData.detail || errorMessage;
                        } catch (e) {
                            // Use default error message
                        }
                        this.handleUploadError(errorMessage);
                    }
                });

                // Handle errors
                xhr.addEventListener('error', () => {
                    this.handleUploadError('Network error during upload');
                });

                xhr.addEventListener('abort', () => {
                    this.handleUploadError('Upload cancelled');
                });

                // Send request
                xhr.open('POST', '/api/documents/upload');
                xhr.send(formData);

            } catch (error) {
                console.error('Upload error:', error);
                this.handleUploadError(error.message || 'Upload failed');
            }
        },

        handleUploadError(message) {
            this.uploadProgress = 0;
            this.uploadStatus = 'error';
            this.uploadMessage = message;
            this.showToast(message, 'error');

            console.error('Upload error:', message);

            // Clear error after 5 seconds
            setTimeout(() => {
                this.uploadMessage = '';
                this.uploadStatus = '';
                this.uploadingFileName = '';
            }, 5000);
        },

        async deleteDocument(documentId) {
            if (!confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
                return;
            }

            try {
                const response = await fetch(`/api/documents/${documentId}`, {
                    method: 'DELETE',
                    credentials: 'include'
                });

                if (!response.ok) {
                    throw new Error('Failed to delete document');
                }

                this.showToast('Document deleted successfully', 'success');

                // Reload documents
                await this.loadDocuments();

            } catch (error) {
                console.error('Error deleting document:', error);
                this.showToast('Failed to delete document', 'error');
            }
        },

        async searchInDocument(doc) {
            this.showDocumentsPanel = false;
            this.currentQuery = `What does ${doc.filename} say about `;

            // Focus on input
            this.$nextTick(() => {
                const input = document.querySelector('input[type="text"]');
                if (input) {
                    input.focus();
                    // Move cursor to end
                    input.setSelectionRange(input.value.length, input.value.length);
                }
            });
        },

        // Toast Notification System
        showToast(message, type = 'info', duration = 4000) {
            const toast = {
                message,
                type, // 'success', 'error', 'info', 'warning'
                show: true
            };

            this.toasts.push(toast);

            // Auto-remove toast after duration
            setTimeout(() => {
                toast.show = false;
                setTimeout(() => {
                    const index = this.toasts.indexOf(toast);
                    if (index > -1) {
                        this.toasts.splice(index, 1);
                    }
                }, 300); // Wait for exit animation
            }, duration);
        },

        // Utility: Format date for display
        formatDate(dateString) {
            if (!dateString) return 'Unknown';

            try {
                const date = new Date(dateString);
                const now = new Date();
                const diffMs = now - date;
                const diffMins = Math.floor(diffMs / 60000);
                const diffHours = Math.floor(diffMs / 3600000);
                const diffDays = Math.floor(diffMs / 86400000);

                if (diffMins < 1) return 'Just now';
                if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
                if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
                if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;

                // For older dates, show formatted date
                return date.toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                });
            } catch (error) {
                return dateString;
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