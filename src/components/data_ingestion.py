import os
import sys
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.config import *
from src.connections.mongo_db_connection import DBConnection



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        self.db = DBConnection()
    
    def initiate_data_ingestion(self):
        try:
            # reading data from DB                                   
            df = self.db.get_data()
            
            logging.info("Data reading completed from database")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_file_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_file_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return self.ingestion_config.raw_data_file_path
                

        except Exception as e:
            raise CustomException(e, sys)






