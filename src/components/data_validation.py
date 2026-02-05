import json
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.config import *
from src.utils import load_config

class DataValidation:
    def __init__(self, raw_data_file_path):        

        try:            
            self.data_validation_config = DataValidationConfig()
            self._schema_config = load_config(DATA_SCHEMA)
            self.data_ingestion_artifact = raw_data_file_path
        

        except Exception as e:
            raise CustomException(e,sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:

        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Are all required column present: [{status}]")
            return status
        
        except Exception as e:
            raise CustomException(e, sys)

    def is_column_exist(self, df: pd.DataFrame) -> bool:
 
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_cols"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_cols"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        
        except Exception as e:
            raise CustomException(e, sys) 

        

    def initiate_data_validation(self):

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            raw_data_df = pd.read_csv("self.data_ingestion_artifact")

            # Checking col len of dataframe for train/test df
            status = self.validate_number_of_columns(dataframe=raw_data_df)
            
            if not status:
                validation_error_msg += f"Columns are missing in raw data. "
            else:
                logging.info(f"All required columns present in raw data: {status}")

            
            # Validating col dtype for train/test df
            status = self.is_column_exist(df=raw_data_df)
            if not status:
                validation_error_msg += f"Columns are missing in raw data. "
            else:
                logging.info(f"All categorical/int columns present in raw data: {status}")


            validation_status = len(validation_error_msg) == 0

            validation_error_msg

            validation_report_file_path = self.data_validation_config.data_validation_report_path
            

            # Ensure the directory for validation_report_file_path exists
            os.makedirs(validation_report_file_path, exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            
            return validation_status, validation_error_msg, validation_report_file_path
        
        except Exception as e:
            raise CustomException(e, sys) 