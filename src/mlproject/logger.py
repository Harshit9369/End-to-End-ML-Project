import logging 
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime( '%m_%d_%Y_%H_%M_%S')}.log"
# datetime.now() provides the current time. 
# the format is m-d-Y-H-M-S -> month-days-year-hour-minutes-second with .log file extension
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
'''getcwd() - get current working directory. in the current working directory, it forms a logs folder
and then the LOG_FILE is added to that folder. 
'''
os.makedirs(log_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)