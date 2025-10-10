import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Optional


_DEF_LOG_DIR = "logs"
_DEF_LOG_FILE = "app.log"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_logger(name: Optional[str] = None,
               log_dir: str = _DEF_LOG_DIR,
               level: int = logging.INFO) -> logging.Logger:
    """
    Create/retrieve a module logger that logs to console and to a daily rotating file.
    - Logs directory: logs/
    - File name: app.log (rotates at midnight, keeps 14 backups)
    - Format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    """
    _ensure_dir(log_dir)

    logger_name = name or "app"
    logger = logging.getLogger(logger_name)

    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Timed rotating file handler (daily)
    log_path = os.path.join(log_dir, _DEF_LOG_FILE)
    fh = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    return logger
