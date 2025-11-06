"""Ensemble LLM - Multi-model ensemble with voting and web search"""

from .main import EnsembleLLM
from .web_search import WebSearcher
from .logger import setup_logger

__version__ = "1.0.0"
__all__ = ["EnsembleLLM", "WebSearcher", "setup_logger"]