"""
src/utils/logger.py
-------------------
Centralised logging configuration for the Phishing Detection project.

Every module in the project calls get_logger(__name__) to obtain a
logger that writes simultaneously to:
  - The console  (INFO and above)
  - A dated log file under logs/  (DEBUG and above)

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Dataset loaded successfully.")
"""

import logging
import os
from datetime import datetime
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    A logger is created only once per name; subsequent calls return
    the cached instance without adding duplicate handlers.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int
        Minimum severity captured by *both* handlers (default DEBUG).

    Returns
    -------
    logging.Logger
        Fully configured logger instance.
    """
    logger = logging.getLogger(name)

    # Do not add handlers more than once (important in notebooks that
    # re-execute cells)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler (INFO+) ───────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (DEBUG+) ─────────────────────────────────────────────────
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_filename = LOG_DIR / f"phishing_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # If the log directory cannot be created, continue with console only
        logger.warning(f"Could not create file log handler: {exc}")

    # Prevent propagation to the root logger to avoid duplicate output
    logger.propagate = False

    return logger
