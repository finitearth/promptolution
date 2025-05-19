"""Logging configuration for the promptolution library."""
import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with the specified name and level.

    This function provides a standardized way to get loggers throughout the library,
    ensuring consistent formatting and behavior.

    Args:
        name (str): Name of the logger, typically __name__ of the module.
        level (int, optional): Logging level. Defaults to None, which uses the root logger's level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging for the promptolution library.

    This function configures the root logger for the library with appropriate
    formatting and level.

    Args:
        level (int, optional): Logging level. Defaults to logging.INFO.
    """
    # Configure the root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
