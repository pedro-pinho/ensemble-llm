"""Logging configuration for Ensemble LLM"""

import logging
import sys
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""

    grey = "\x1b[38;21m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.INFO: green
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.WARNING: yellow
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.ERROR: red
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.CRITICAL: bold_red
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(name: str = "EnsembleLLM", level: str = "INFO") -> logging.Logger:
    """Set up logger with console and file handlers"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "ensemble_llm.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
