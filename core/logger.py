import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Any
import json


class LightLogger:
    """Custom logger class"""

    LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, name="VRPTW"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create log directory
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Generate log file name
        self.log_file = os.path.join(self.log_dir, f"base.log")

        self._setup_file_handler()
        self._setup_console_handler()

        # Add context information
        self.extra = {}

    def _setup_file_handler(self):
        """Configure file handler"""
        # Rotate by file size
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )

        # Set format
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s.py] %(message)s",
            datefmt="%Y%m%d-%H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _setup_console_handler(self):
        """Configure console handler"""
        console_handler = logging.StreamHandler()
        # Console uses simplified format
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_context(self, **kwargs):
        """Set log context information"""
        self.extra.update(kwargs)

    def format_params(
        self,
        params: Dict[str, Any],
        title: str = "Parameter Configuration",
        style: str = "table",
    ) -> None:
        """Format output parameter configuration

        Args:
            params: Parameter dictionary
            style: Output style ('table' or 'json')
        """
        if style == "table":
            # Table style
            self.info(f"{title}:")
            self.info("=" * 50)
            self.info("%-20s | %-25s", "Parameter", "Value")
            self.info("-" * 50)
            for key, value in params.items():
                self.info("%-20s | %-25s", key, str(value))
            self.info("=" * 50)

        elif style == "json":
            # JSON style
            formatted_json = json.dumps(
                params,
                indent=2,
                ensure_ascii=False,
                default=str,  # Handle types that cannot be JSON serialized
            )
            self.info("Parameter Configuration:\n%s", formatted_json)

    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, extra=self.extra, **kwargs)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, extra=self.extra, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, extra=self.extra, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, extra=self.extra, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, extra=self.extra, **kwargs)
