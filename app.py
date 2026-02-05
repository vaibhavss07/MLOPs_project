from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config import *
import sys

if __name__=="__main__":

    try:
        # Fetching data from DB and writing in raw_data.csv
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()

        # Feature engg and transformation pipeline creation
        data_transformation = DataTransformation()

        # # Hyperparameter tuning and finding best model best pipeline
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(raw_data_path)


    except Exception as e:
        raise CustomException(e, sys)