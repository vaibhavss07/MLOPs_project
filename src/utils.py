# ===================== Core Libraries =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import date
import yaml
import sys
from src.exception import CustomException
from src.logger import logging

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)



# ===================== Metrics =====================
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)



# ==================== read_yaml_file FUNCTION ====================

def load_config(config_path='config.yaml'):

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # after FE
        numerical_cols = config['numerical_cols']
        categorical_cols = config['categorical_cols']
        power_transform_cols = config['power_transform_cols']
        ordinal_columns = config['ordinal_columns']
        nominal_columns = config['nominal_columns']
        
        # Convert ordinal_categories from dict to list format
        # This matches the original format: [['High School', "Master's", "Bachelor's", 'Doctorate']]
        ordinal_categories = [config['ordinal_categories'][col] for col in ordinal_columns]
        
        return {
            'columns': config['columns'],
            'og_numerical_cols': config['og_numerical_cols'],
            'og_categorical_cols': config['og_categorical_cols'],
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'power_transform_cols': power_transform_cols,
            'ordinal_columns': ordinal_columns,
            'ordinal_categories': ordinal_categories,
            'nominal_columns': nominal_columns
        }

    except Exception as e:
        raise CustomException(e, sys) 

# ==================== EVALUATION FUNCTION ====================

def evaluate_clf(true, predicted, predicted_proba=None):

    try:
        acc = accuracy_score(true, predicted)
        f1 = f1_score(true, predicted, average='binary')
        precision = precision_score(true, predicted, average='binary')
        recall = recall_score(true, predicted, average='binary')
        
        # Use predicted probabilities if available, otherwise use predicted labels
        if predicted_proba is not None:
            roc_auc = roc_auc_score(true, predicted_proba)
        else:
            roc_auc = roc_auc_score(true, predicted)
        
        gini = 2 * roc_auc - 1  # Gini = 2 * AUC - 1
        
        return {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'gini': gini
        }
    
    except Exception as e:
        raise CustomException(e, sys) 