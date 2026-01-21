import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


list_of_files=[
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/components/model_pusher.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "logs/.gitkeep",
    "artifacts/.gitkeep",
    ".env"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory only if it doesn't exist
    if filedir != "":
        if not os.path.exists(filedir):
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for the file {filename}")
        else:
            logging.info(f"Directory already exists: {filedir}")

    # Create file only if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filename}")