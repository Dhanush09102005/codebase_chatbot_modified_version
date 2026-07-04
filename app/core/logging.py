"""
Logging configuration for the application.

Uses Python's built-in logging with:
- Duplicate handler prevention
- Rotating file handler (10MB files, 5 backups)
- Configurable log directory from settings
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.core.config import settings


def setup_logging() -> logging.Logger:
    """
    Configure and return the application logger.
    
    Features:
    - Prevents duplicate handlers on re-initialization
    - Uses RotatingFileHandler to prevent large log files
    - Console and file logging with structured format
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger("code_qa_assistant")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory from config
    log_dir = settings.log_directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Rotating file handler (prevents large log files)
    file_handler = RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Add handlers only if they don't exist
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Global logger instance
logger = setup_logging()
