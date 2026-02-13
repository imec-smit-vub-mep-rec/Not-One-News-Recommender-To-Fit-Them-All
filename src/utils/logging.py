"""
Logging utilities for the RICON analysis pipeline.

Provides consistent logging configuration across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import os

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency at runtime
    psutil = None


# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    console: bool = True,
) -> logging.Logger:
    """Set up logging configuration for the pipeline.
    
    Args:
        level: Logging level (e.g., logging.INFO, 'DEBUG')
        log_file: Optional path to write logs to file
        format_string: Format string for log messages
        date_format: Format string for timestamps
        console: Whether to output to console
        
    Returns:
        Root logger for the pipeline
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger for the pipeline
    logger = logging.getLogger("ricon")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Reduce verbosity of third-party loggers
    logging.getLogger("recpack").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Name of the module/component
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"ricon.{name}")


def log_memory(label: str, logger: Optional[logging.Logger] = None) -> None:
    """Log process/system memory stats for OOM diagnostics."""
    if logger is None:
        logger = get_logger("memory")

    if psutil is None:
        logger.info(f"[MEMORY] {label} | psutil unavailable")
        return

    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    vmem = psutil.virtual_memory()
    avail_gb = vmem.available / (1024 ** 3)
    used_pct = vmem.percent
    logger.info(
        f"[MEMORY] {label} | rss={rss_gb:.2f}GB | available={avail_gb:.2f}GB | used={used_pct:.1f}%"
    )


class ProgressLogger:
    """Helper class for logging progress of long-running operations."""
    
    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        description: str = "Processing",
        log_interval: int = 10
    ):
        """Initialize the progress logger.
        
        Args:
            logger: Logger to use
            total: Total number of items
            description: Description of the operation
            log_interval: Percentage interval for logging updates
        """
        self.logger = logger
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_percent = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """Update progress.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        percent = int(100 * self.current / self.total) if self.total > 0 else 100
        
        if percent >= self.last_logged_percent + self.log_interval or percent == 100:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"{self.description}: {percent}% ({self.current}/{self.total}) "
                f"- {rate:.1f} items/sec - ETA: {eta:.0f}s"
            )
            self.last_logged_percent = percent
    
    def finish(self):
        """Mark the operation as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.description}: Complete. "
            f"Processed {self.current} items in {elapsed:.1f}s"
        )


class LogContext:
    """Context manager for logging entry/exit of operations."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        """Initialize the log context.
        
        Args:
            logger: Logger to use
            operation: Name of the operation
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({elapsed:.1f}s)")
        else:
            self.logger.error(
                f"Failed: {self.operation} ({elapsed:.1f}s) - {exc_type.__name__}: {exc_val}"
            )
        
        return False  # Don't suppress exceptions
