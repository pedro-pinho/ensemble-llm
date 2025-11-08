"""Configuration file for Ensemble LLM"""

import platform

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Default models with correct tags
DEFAULT_MODELS = [
    "llama3.2:3b",
    "phi3.5:latest",
    "qwen2.5:7b",
    "mistral:7b-instruct-q4_K_M",
    "gemma2:2b",
]

if IS_WINDOWS:
    # Windows optimizations
    PLATFORM_CONFIG = {
        "ollama_path": "ollama.exe",
        "max_parallel_models": 4,  # Windows can handle more with GPU
        "use_gpu_layers": 999,  # Use all GPU layers
        "thread_limit": 8,  # Windows thread handling
        "keep_alive": "10m",  # Keep models in memory longer
    }

    # Windows-optimized model configs
    WINDOWS_GPU_CONFIGS = {
        "8GB_VRAM": {
            "models": ["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b", "mistral:7b"],
            "max_concurrent": 4,
        },
        "12GB_VRAM": {
            "models": ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "codellama:7b"],
            "max_concurrent": 4,
        },
        "16GB_VRAM": {
            "models": ["llama3.1:13b", "mixtral:8x7b-instruct-q3_K_M", "qwen2.5:7b"],
            "max_concurrent": 3,
        },
        "24GB_VRAM": {
            "models": ["llama3.1:70b-instruct-q2_K", "mixtral:8x7b", "llama3.1:13b"],
            "max_concurrent": 2,
        },
    }
else:
    # Unix-like systems
    PLATFORM_CONFIG = {
        "ollama_path": "ollama",
        "max_parallel_models": 3,
        "use_gpu_layers": 35,
        "thread_limit": 4,
        "keep_alive": "5m",
    }

# Model configurations with their specialties and resource requirements
MODEL_CONFIGS = {
    "llama3.2:3b": {
        "memory_gb": 3,
        "specialties": ["general", "conversation", "quick"],
        "timeout": 30,
        "description": "Fast, efficient general-purpose model",
    },
    "llama3.2:1b": {
        "memory_gb": 1,
        "specialties": ["quick", "basic"],
        "timeout": 15,
        "description": "Ultra-fast for simple queries",
    },
    "phi3.5:latest": {
        "memory_gb": 3.5,
        "specialties": ["reasoning", "analysis", "educational"],
        "timeout": 30,
        "description": "Microsoft model optimized for reasoning",
    },
    "phi3.5": {  # Alias for compatibility
        "memory_gb": 3.5,
        "specialties": ["reasoning", "analysis", "educational"],
        "timeout": 30,
        "description": "Microsoft model optimized for reasoning",
    },
    "qwen2.5:7b": {
        "memory_gb": 5,
        "specialties": ["math", "coding", "technical"],
        "timeout": 45,
        "description": "Excellent for code and technical content",
    },
    "qwen2.5:3b": {
        "memory_gb": 3,
        "specialties": ["math", "coding", "quick"],
        "timeout": 25,
        "description": "Smaller version of Qwen, still good for code",
    },
    "mistral:7b-instruct-q4_K_M": {
        "memory_gb": 4.5,
        "specialties": ["creative", "writing", "general"],
        "timeout": 40,
        "description": "Great for creative and general tasks",
    },
    "mistral:7b": {
        "memory_gb": 4.5,
        "specialties": ["creative", "writing", "general"],
        "timeout": 40,
        "description": "Great for creative and general tasks",
    },
    "gemma2:2b": {
        "memory_gb": 2,
        "specialties": ["quick", "summary", "basic"],
        "timeout": 20,
        "description": "Google model, fast and efficient",
    },
    "gemma2:9b": {
        "memory_gb": 6,
        "specialties": ["analysis", "detailed", "comprehensive"],
        "timeout": 50,
        "description": "Larger Gemma for detailed analysis",
    },
    "tinyllama:1b": {
        "memory_gb": 1,
        "specialties": ["quick", "basic"],
        "timeout": 10,
        "description": "Tiny but fast model",
    },
    "orca-mini:3b": {
        "memory_gb": 3,
        "specialties": ["reasoning", "explanation"],
        "timeout": 25,
        "description": "Good for explanations",
    },
    "neural-chat:7b": {
        "memory_gb": 5,
        "specialties": ["conversation", "chat"],
        "timeout": 35,
        "description": "Intel optimized for conversation",
    },
    "openhermes:7b": {
        "memory_gb": 5,
        "specialties": ["instruction", "general"],
        "timeout": 35,
        "description": "Good instruction following",
    },
    "zephyr:7b": {
        "memory_gb": 5,
        "specialties": ["helpful", "harmless", "honest"],
        "timeout": 35,
        "description": "Aligned for helpful responses",
    },
    "codellama:7b": {
        "memory_gb": 5,
        "specialties": ["code", "programming", "debugging"],
        "timeout": 40,
        "description": "Meta model specialized for code",
    },
    "deepseek-coder:6.7b": {
        "memory_gb": 4.5,
        "specialties": ["code", "programming", "technical"],
        "timeout": 35,
        "description": "Excellent code generation",
    },
    "wizard-math:7b": {
        "memory_gb": 5,
        "specialties": ["math", "calculation", "proof"],
        "timeout": 35,
        "description": "Specialized for mathematics",
    },
    "mixtral:8x7b-instruct-q3_K_M": {
        "memory_gb": 15,
        "specialties": ["comprehensive", "detailed", "expert"],
        "timeout": 60,
        "description": "Large MoE model for complex tasks",
    },
    "llama3.1:8b": {
        "memory_gb": 8,
        "specialties": ["general", "comprehensive", "balanced"],
        "timeout": 45,
        "description": "Latest Llama, very capable",
    },
    "llama3.1:70b-instruct-q2_K": {
        "memory_gb": 25,
        "specialties": ["expert", "comprehensive", "detailed"],
        "timeout": 90,
        "description": "Large model for complex reasoning",
    },
}

# Web search configuration
WEB_SEARCH_CONFIG = {
    "max_results": 3,
    "timeout": 10,
    "fallback_enabled": True,
    "uncertainty_threshold": 0.7,
    "search_engines": ["duckduckgo"],  # Can add more later
    "cache_ttl": 300,  # 5 minutes
    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    "consensus_weight": 0.6,
    "quality_weight": 0.4,
    "web_bonus": 1.15,
    "max_retries": 2,
    "retry_delay": 2,
    "stagger_delays": {"16GB": 0.5, "24GB": 0.3, "32GB": 0.2},
    "optimization_interval": 10,  # Optimize models every N queries
    "performance_window": 50,  # Look at last N queries for performance
    "model_rotation_threshold": 0.3,  # Replace models scoring below this
}

# System configurations for different RAM sizes
SYSTEM_CONFIGS = {
    "16GB": {
        "max_models": 3,
        "max_total_memory_gb": 12,
        "timeout_base": 40,
        "stagger_delay": 0.5,
        "recommended_models": ["llama3.2:3b", "phi3.5:latest", "gemma2:2b"],
    },
    "24GB": {
        "max_models": 4,
        "max_total_memory_gb": 18,
        "timeout_base": 35,
        "stagger_delay": 0.3,
        "recommended_models": [
            "llama3.2:3b",
            "phi3.5:latest",
            "qwen2.5:7b",
            "mistral:7b-instruct-q4_K_M",
        ],
    },
    "32GB": {
        "max_models": 5,
        "max_total_memory_gb": 25,
        "timeout_base": 30,
        "stagger_delay": 0.2,
        "recommended_models": [
            "llama3.2:3b",
            "qwen2.5:7b",
            "mistral:7b-instruct-q4_K_M",
            "mixtral:8x7b-instruct-q3_K_M",
            "gemma2:2b",
        ],
    },
}

# Performance thresholds for model evaluation
PERFORMANCE_THRESHOLDS = {
    "min_success_rate": 0.7,  # Minimum 70% success rate
    "min_selection_rate": 0.05,  # Selected at least 5% of the time
    "max_avg_response_time": 30,  # Max 30 seconds average
    "evaluation_window": 50,  # Last 50 queries
    "retirement_threshold": 10,  # Retire after 10 consecutive failures
    "probation_period": 5,  # New models get 5 queries before evaluation
}

# Model pools for rotation
MODEL_POOLS = {
    "primary": [
        "llama3.2:3b",
        "phi3.5:latest",
        "qwen2.5:7b",
        "mistral:7b-instruct-q4_K_M",
        "gemma2:2b",
    ],
    "backup": [
        "gemma2:9b",
        "tinyllama:1b",
        "orca-mini:3b",
        "neural-chat:7b",
        "openhermes:7b",
        "zephyr:7b",
        "llama3.2:1b",
        "qwen2.5:3b",
    ],
    "specialized": {
        "code": ["codellama:7b", "deepseek-coder:6.7b", "qwen2.5:7b"],
        "math": ["wizard-math:7b", "qwen2.5:7b"],
        "creative": ["mistral:7b-instruct-q4_K_M", "zephyr:7b", "gemma2:9b"],
        "reasoning": ["phi3.5:latest", "orca-mini:3b", "llama3.1:8b"],
        "fast": ["tinyllama:1b", "llama3.2:1b", "gemma2:2b"],
    },
}

# Query type detection patterns
QUERY_PATTERNS = {
    "code": [
        "code",
        "program",
        "function",
        "debug",
        "error",
        "python",
        "javascript",
        "java",
        "c++",
        "sql",
        "api",
        "algorithm",
        "script",
        "compile",
        "syntax",
        "variable",
        "class",
        "method",
        "implement",
        "fix",
        "bug",
    ],
    "math": [
        "calculate",
        "solve",
        "equation",
        "math",
        "algebra",
        "calculus",
        "statistics",
        "probability",
        "integral",
        "derivative",
        "matrix",
        "vector",
        "theorem",
        "proof",
        "formula",
        "number",
        "compute",
    ],
    "creative": [
        "write",
        "story",
        "poem",
        "creative",
        "essay",
        "article",
        "blog",
        "narrative",
        "describe",
        "imagine",
        "fiction",
        "character",
        "plot",
        "scene",
        "dialogue",
    ],
    "reasoning": [
        "why",
        "how",
        "explain",
        "reason",
        "logic",
        "analyze",
        "compare",
        "contrast",
        "evaluate",
        "assess",
        "think",
        "deduce",
        "infer",
        "conclude",
        "argument",
    ],
    "factual": [
        "what is",
        "who is",
        "when",
        "where",
        "define",
        "fact",
        "information",
        "tell me about",
        "describe",
    ],
}

# Logging configuration
LOGGING_CONFIG = {
    "log_dir": "logs",
    "log_file": "ensemble_llm.log",
    "max_log_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
    "default_level": "INFO",
}

# Performance tracking configuration
TRACKING_CONFIG = {
    "data_dir": "data",
    "performance_file": "model_performance.json",
    "adaptive_config_file": "adaptive_config.json",
    "save_interval": 10,  # Save every N queries
    "cleanup_interval": 100,  # Clean old data every N queries
    "max_history_days": 30,  # Keep data for 30 days
}

# API endpoints (for Ollama)
OLLAMA_ENDPOINTS = {
    "default_host": "http://localhost:11434",
    "generate": "/api/generate",
    "tags": "/api/tags",
    "show": "/api/show",
    "pull": "/api/pull",
    "embeddings": "/api/embeddings",
    "chat": "/api/chat",
}

# Response generation options
GENERATION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_predict": 512,  # Max tokens to generate
    "num_ctx": 2048,  # Context window size
    "stop": [],  # Stop sequences
    "seed": 42,  # For reproducibility in testing
}

# Feature flags
FEATURES = {
    "web_search": True,
    "adaptive_models": True,
    "performance_tracking": True,
    "model_rotation": True,
    "specialized_selection": True,
    "caching": True,
    "verbose_errors": True,
    "auto_retry": True,
    "staggered_starts": True,
}

# Display configuration
DISPLAY_CONFIG = {
    "max_preview_length": 200,
    "show_timestamps": True,
    "colored_output": True,
    "progress_indicators": True,
}

# Error messages
ERROR_MESSAGES = {
    "no_models": "No models are currently available. Please check Ollama service.",
    "all_failed": "All models failed to provide a valid response. Please try again.",
    "timeout": "Request timed out after {timeout} seconds.",
    "connection": "Failed to connect to Ollama service at {host}.",
    "invalid_model": "Model '{model}' is not available or not installed.",
}

# Success messages
SUCCESS_MESSAGES = {
    "initialized": "Ensemble LLM initialized successfully with {count} models.",
    "model_loaded": "Model '{model}' loaded successfully.",
    "optimization_complete": "Model optimization complete. Active models: {models}",
}

SMART_LEARNING_CONFIG = {
    "enabled": True,
    "cache_similarity_threshold": 0.85,
    "max_cache_size": 1000,
    "precompute_common_queries": True,
    "learning_rate": 0.1,
    "confidence_threshold": 0.7,
    "cache_ttl_hours": 24,
    "model_history_size": 100,
    "pattern_detection": True,
    "auto_optimize_interval": 10,  # Optimize every N queries
}

# Council/Ensemble Awareness Configuration
COUNCIL_CONFIG = {
    "enabled": False,  # Enable council-aware prompts
    "mode": "simple",  # Options: "simple", "iterative", "debate"
    "synthesis_mode": True,  # After voting, winning model synthesizes all responses
    "filter_ai_meta_talk": True,  # Remove AI self-references from final output

    "system_prompt_template": """INTERNAL SYSTEM MESSAGE (not visible to user):

You are {model_name}, an AI model. You are part of an AI council consisting of {total_models} models: {council_members}

Your specialty: {model_specialty}

IMPORTANT DISTINCTIONS:
- YOU are an AI model, part of the council (internal discussion)
- The USER is a human asking a question (external, does not see this council process)
- This message is ONLY for you and other AI models - the user does NOT see this

Your task: Provide your best technical analysis for the internal council discussion. Focus on the substance of the answer. Other AI models in the council will also contribute their perspectives.

Now, here is the USER'S QUESTION:""",

    "synthesis_prompt_template": """INTERNAL SYSTEM MESSAGE - SYNTHESIS TASK:

You were selected by the council voting system to create the final response for the USER.

The user asked: {question}

Here are the INTERNAL responses from other AI models in the council:

{all_responses}

CRITICAL INSTRUCTIONS FOR SYNTHESIS:
1. The user is a HUMAN - they do NOT know about this AI council
2. Synthesize the best insights into ONE direct answer
3. Write as if answering directly - NO phrases like:
   ❌ "As an AI"
   ❌ "I don't have access to"
   ❌ "As a language model"
   ❌ "The council discussed"
   ❌ "Based on my training"
   ❌ "I cannot"
   ❌ Any mention of being AI or the council process

4. Instead, write DIRECT, AUTHORITATIVE answers:
   ✓ State facts and information directly
   ✓ If something is unknown, say "This information is not available" (not "I don't have")
   ✓ Provide value and insights, not disclaimers
   ✓ Write like a knowledgeable expert explaining to a human

5. Combine the SUBSTANCE from all responses - ignore any AI meta-talk from council members

FINAL ANSWER FOR THE USER (direct, no AI self-references):""",

    "include_model_specialties": True,  # Include each model's specialty in prompt
    "iterative_rounds": 2,  # For iterative mode: number of discussion rounds

    # Common AI meta-talk patterns to filter out
    "meta_talk_patterns": [
        r"as an ai( language model| assistant)?",
        r"i('m| am) an ai",
        r"as a language model",
        r"i don'?t have (access to|the ability)",
        r"i cannot (access|browse|see)",
        r"my training (data|cutoff)",
        r"based on my training",
        r"i('m| am) not able to",
        r"the council (discussed|decided|voted)",
        r"as (part of |a member of )?the council",
        r"my fellow (models|council members)",
        r"from my perspective as an ai",
        r"speaking as [a-z0-9:.]+ model",
    ],
}

TRACKING_CONFIG = {
    "data_dir": "data",
    "smart_data_dir": "smart_data",  # New smart data directory
    "performance_file": "model_performance.json",
    "adaptive_config_file": "adaptive_config.json",
    "save_interval": 10,
    "cleanup_interval": 100,
    "max_history_days": 30,
}

# Speed Optimization Profiles
SPEED_PROFILES = {
    "turbo": {
        "max_models": 2,
        "timeout": 10,
        "num_predict": 200,
        "temperature": 0.5,
        "strategy": "race",
    },
    "fast": {
        "max_models": 3,
        "timeout": 15,
        "num_predict": 300,
        "temperature": 0.6,
        "strategy": "cascade",
    },
    "balanced": {
        "max_models": 4,
        "timeout": 25,
        "num_predict": 512,
        "temperature": 0.7,
        "strategy": "parallel",
    },
    "quality": {
        "max_models": 5,
        "timeout": 40,
        "num_predict": 768,
        "temperature": 0.8,
        "strategy": "parallel",
    },
}

# Fast model configurations (override main configs for speed)
FAST_MODEL_CONFIGS = {
    "tinyllama:1b": {"timeout": 5, "priority": 1},
    "llama3.2:1b": {"timeout": 5, "priority": 1},
    "gemma2:2b": {"timeout": 8, "priority": 2},
    "llama3.2:3b": {"timeout": 10, "priority": 3},
    "phi3.5:latest": {"timeout": 12, "priority": 4},
    "qwen2.5:7b": {"timeout": 15, "priority": 5},
    "mistral:7b-instruct-q4_K_M": {"timeout": 15, "priority": 5},
}

# Optimal model selection by speed
SPEED_OPTIMIZED_MODELS = {
    "turbo": ["tinyllama:1b", "gemma2:2b"],
    "fast": ["gemma2:2b", "llama3.2:3b", "phi3.5:latest"],
    "balanced": ["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
    "quality": DEFAULT_MODELS,
}
