import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# DB connection constants
CONNECTION_URL =  os.getenv("connection_url")
DB_NAME =  os.getenv("db_name")
COLLECTION_NAME =  os.getenv("collection_name")

# raw data schema
DATA_SCHEMA = "schema.yaml"

# dagshub constants
REPO_OWNER=os.getenv("repo_owner")
REPO_NAME=os.getenv("repo_name")
REMOTE_SERVER = os.getenv("remote_server")

# AWS constants
AWS_ACCESS_KEY = os.getenv("aws_access_key")
AWS_SECRET_ACCESS_KEY = os.getenv("aws_secret_access_key")
REGION_NAME = "us-east-1"

# MODEL Evaluation related constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "vs-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"


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
    X_train_transformed_dir: str = os.path.join(data_transformation_dir, "X_train_transformed.npy")
    X_test_transformed_dir: str = os.path.join(data_transformation_dir, "X_test_transformed.npy")
    feature_names_dir: str = os.path.join(data_transformation_dir, "Feature_names.json")
    preprocessing_pipeline_dir: str = os.path.join(data_transformation_dir, "preprocessing_pipeline.pkl")


@dataclass
class ModelTrainerConfig:
    model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
    model_comparison_results_dir = os.path.join(model_trainer_dir, "model_comparison_results.csv")
    best_pipeline_dir = os.path.join(model_trainer_dir, "{}_model.pkl")
    label_encoder_dir = os.path.join(model_trainer_dir, "label_encoder.pkl")

    shap_dir = os.path.join(model_trainer_dir, "shap")
    global_shap_dir = os.path.join(shap_dir, "global_shap")
    local_shap_dir = os.path.join(shap_dir, "local_shap")
    shap_explainer_dir = os.path.join(shap_dir, "shap_explainer.pkl")
    