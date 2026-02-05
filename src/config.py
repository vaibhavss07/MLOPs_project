import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# DB connection constants
CONNECTION_URL =  os.getenv("connection_url")
DB_NAME =  os.getenv("db_name")
COLLECTION_NAME =  os.getenv("collection_name")

DATA_SCHEMA = "schema.yaml"

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:

    artifact_dir: str = os.path.join("artifacts", TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    raw_data_file_path: str = os.path.join(training_pipeline_config.artifact_dir, "raw_data.csv")
    train_test_split_ratio: float = 0.2

@dataclass
class DataValidationConfig:
    data_validation_report_path: str = os.path.join(training_pipeline_config.artifact_dir, "data_validation_report.yaml")

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")


@dataclass
class ModelTrainerConfig:
    model_comparison_results_dir = os.path.join(training_pipeline_config.artifact_dir, "model_comparison_results.csv")
    best_pipeline_dir = os.path.join(training_pipeline_config.artifact_dir, "best_pipeline_{}.pkl")
    label_encoder_dir = os.path.join(training_pipeline_config.artifact_dir, "label_encoder.pkl")
    