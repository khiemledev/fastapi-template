import logging
import sys
from pathlib import Path

from loguru import logger

from .config import LoggerConfig


def get_logger():
    return logger


def setup_logger(
    name: str = "app",
    logdir: Path | str = Path(str(LoggerConfig.LogDir)),
    log_level: int = logging.INFO,
    backtrace: bool = bool(LoggerConfig.BackTrace),
):
    """Setup a logger with file and stream handlers.

    Args:
        name (str, optional): name of logger. Defaults to "app".

        logdir (Path | str, optional): folder where log files will
        be stored. Defaults to Path(LoggerConfig.LogDir).

        log_level (int, optional): log level. Defaults to logging.INFO.

        backtrace (bool, optional): enable backtrace. Defaults to LoggerConfig.BackTrace.

    Returns:
        logging.Logger: logger object
    """  # noqa: E501

    # Make log directory
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / name

    # Remove default std.err handler
    logger.remove(0)

    logger.add(
        sys.stdout,
        level=log_level,
        backtrace=backtrace,
        diagnose=True,
    )

    logger.add(
        path.with_suffix(".log"),
        level=log_level,
        rotation=int(LoggerConfig.MaxBytes),
        retention=int(LoggerConfig.MaxBackupCount),
        backtrace=backtrace,
        diagnose=True,
        serialize=False,  # Enable this to log in json format
    )

    return logger
