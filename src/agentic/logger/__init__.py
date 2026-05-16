from .logging import setup_logging, get_logger

# Setup runs once when module is imported
#setup_logging()  # Removed to avoid circular import

# Create default logger
logger = get_logger()

__all__ = ["logger", "get_logger"]