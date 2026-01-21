from src.logger import logging
from src.exception import CustomException
import sys

if __name__=="__main__":
    logging.info("Checking if logging works?")

    try:
        a= 1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)