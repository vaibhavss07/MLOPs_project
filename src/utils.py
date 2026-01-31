# ===================== Core Libraries =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import date

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


# ==================== EVALUATION FUNCTION ====================

def evaluate_clf(true, predicted, predicted_proba=None):

    acc = accuracy_score(true, predicted)
    f1 = f1_score(true, predicted, average='binary')
    precision = precision_score(true, predicted, average='binary')
    recall = recall_score(true, predicted, average='binary')
    
    # Use predicted probabilities if available, otherwise use predicted labels
    if predicted_proba is not None:
        roc_auc = roc_auc_score(true, predicted_proba)
    else:
        roc_auc = roc_auc_score(true, predicted)
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }