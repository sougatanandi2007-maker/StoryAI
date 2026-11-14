"""
StoryWeave AI - Structured Logging Configuration

Sets up structured logging with contextual information
for debugging and monitoring.
"""

import logging
import sys
from datetime import datetime, timezone


class StoryWeaveFormatter(logging.Formatter):
    """Custom formatter that includes structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        level = record.levelname
        logger_name = record.name
        message = record.getMessage()

        log_line = f"{timestamp} | {level:<8} | {logger_name} | {message}"

        if record.exc_info and record.exc_info[0] is not None:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application-wide logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(StoryWeaveFormatter())
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    for noisy in ["uvicorn.access", "pymongo", "chromadb", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("uvicorn.error").setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a module."""
    return logging.getLogger(f"storyweave.{name}")
