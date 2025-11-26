import datetime
from config import *
import os

def custom_logger(file_name, message, save_file_name=None):
    """
    Custom logger to log messages with timestamps to files.
    
    Args:
        file_name (str): Module or class name that's logging the message
        message (str): Message to log
        save_file_name (str, optional): Specific log file name to use
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {file_name} - {message} "
    
    try:
        if not os.path.exists(LOG_FILES_DIR_PATH):
            os.makedirs(LOG_FILES_DIR_PATH, exist_ok=True)
    except Exception as e:
        print(f"error in making directory of log file: {str(e)}")

    try:
        if save_file_name is None:
            with open(f"{LOG_FILES_DIR_PATH}/custom_logs.txt", "a") as file:
                file.write(log_message + "\n")
        else:
            with open(f"{LOG_FILES_DIR_PATH}/{save_file_name}.txt", "a") as file:
                file.write(log_message + "\n")
    except Exception as e:
        print(f"{log_message} (Error writing to log: {str(e)})")
