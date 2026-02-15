from src.logger import logging
from src.exception import CustomException
from src.config import *
import pymongo
import pandas as pd
import numpy as np
import sys


class DBConnection:
    
    _client = None
    
    def __init__(self):

        """Create MongoDB connection only once (Singleton)"""

        if DBConnection._client is None:
            try:
                DBConnection._client = pymongo.MongoClient(
                    CONNECTION_URL, 
                    tls=True,
                    tlsAllowInvalidCertificates=True,
                    serverSelectionTimeoutMS=30000
                )

                logging.info("MongoDB connection established")

            except Exception as e:
                raise CustomException(e, sys)
        
        self.client = DBConnection._client
    

    def get_data(self):
        
        """Fetch data from MongoDB collection"""
        
        try:
            data_base = self.client[DB_NAME]
            collection = data_base[COLLECTION_NAME]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            
            logging.info(f"Fetched {len(df)} records from MongoDB")
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def close_connection(self):
        """Close MongoDB connection"""
        if DBConnection._client is not None:
            DBConnection._client.close()
            DBConnection._client = None
            logging.info("MongoDB connection closed")