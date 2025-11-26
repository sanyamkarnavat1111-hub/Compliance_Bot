import os
import logging
import sys
from logging.handlers import RotatingFileHandler

# Anchor logs directory to this module's directory to avoid CWD issues
BASE_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
os.makedirs(BASE_LOG_DIR, exist_ok=True)

GLOBAL_LOG_FILE = os.path.join(BASE_LOG_DIR, 'tendor_proposal_evaluation.log')

def get_logger(module_filename: str) -> logging.Logger:
    """
    Get a logger that logs to both the console and a single global file.
    This is for general application logs not tied to a specific process ID.
    """
    module_name = os.path.splitext(os.path.basename(module_filename))[0]
    # Use a unique name for each logger to avoid conflicts, but all will write to the same file handler
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if called repeatedly on the same logger name
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(GLOBAL_LOG_FILE) for h in logger.handlers):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )

        # Ensure base dir exists (defensive)
        os.makedirs(os.path.dirname(GLOBAL_LOG_FILE), exist_ok=True)

        # Single Rotating file handler for all global logs
        file_handler = RotatingFileHandler(GLOBAL_LOG_FILE, maxBytes=10_000_000, backupCount=5) # Increased maxBytes
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler - ensure it's only added once too
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Avoid duplicate logs from parent/root loggers
        logger.propagate = False

    return logger

def setup_process_logger(process_id: str) -> logging.Logger:
    """
    Sets up and returns a dedicated logger for a specific process ID.
    All logs for this logger instance will go to a unique file.
    """
    process_log_file = os.path.join(BASE_LOG_DIR, f'tendor_proposal_evaluation_{process_id}.log')
    process_logger_name = f'process_logger_{process_id}'
    process_logger = logging.getLogger(process_logger_name)
    process_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to ensure fresh configuration if called multiple times for the same ID
    for handler in list(process_logger.handlers):
        process_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )

    # Ensure directory exists for the process log file
    os.makedirs(os.path.dirname(process_log_file), exist_ok=True)

    # File handler for this specific process log
    file_handler = RotatingFileHandler(process_log_file, maxBytes=10_000_000, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    process_logger.addHandler(file_handler)

    # Console handler (optional, can be removed if only file logging is desired for process logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Keep INFO level for console, or adjust as needed
    console_handler.setFormatter(formatter)
    process_logger.addHandler(console_handler)

    process_logger.propagate = False # Prevent logs from propagating to root logger

    return process_logger
