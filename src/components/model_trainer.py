# ===================== Core Libraries =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import date

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# ===================== Sklearn Utilities =====================
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ===================== Models =====================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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

# ===================== Hyperparameter Optimization =====================
import optuna
from optuna.samplers import TPESampler


# ===================== Model Persistence =====================
import joblib
from src.logger import logging
from src.exception import CustomException
from src.config import *
import os
import sys
from dataclasses import dataclass
from src.components.explainability import SHAPExplainer
from src.utils import evaluate_clf
from src.components.data_transformation import DataTransformation

# ===================== MLflow =====================
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ===================== Dagshub =====================
import dagshub
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

# remote server
mlflow.set_tracking_uri(REMOTE_SERVER)


# ==================== HELPER — imbalance ratio ====================

def _pos_weight(y_train: np.ndarray) -> float:
    """scale_pos_weight for XGBoost: count(neg) / count(pos)."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return float(neg / pos) if pos > 0 else 1.0


# ==================== MODEL TRAINER ====================

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.data_transformation = DataTransformation()

    # ==================== OPTUNA OBJECTIVE FUNCTIONS ====================

    def _cv_score(self, model, X_train, y_train) -> float:
        """Shared cross-validation logic for all objectives."""
        scores = cross_val_score(
            model, X_train, y_train,
            cv=3, scoring='roc_auc', n_jobs=-1
        )
        return scores.mean(), scores.std()
    

    def objective_random_forest(self, trial, X_train, y_train):
        """Optuna objective for Random Forest"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight':     'balanced',   # ← imbalance handled here
            'random_state':     42,
            'n_jobs':           -1,
        }
        
        model = RandomForestClassifier(**params)
        mean, std = self._cv_score(model, X_train, y_train)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
        return mean


    def objective_decision_tree(self, trial, X_train, y_train):
        """Optuna objective for Decision Tree"""
        
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            'class_weight':     'balanced',
            'random_state': 42
        }
        
        model = DecisionTreeClassifier(**params)
        mean, std = self._cv_score(model, X_train, y_train)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
        return mean


    def objective_gradient_boosting(self, trial, X_train, y_train):
        """Optuna objective for Gradient Boosting"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        mean, std = self._cv_score(model, X_train, y_train)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
        return mean


    def objective_logistic_regression(self, trial, X_train, y_train):
        """Optuna objective for Logistic Regression - Corrected"""
        
        # Suggest penalty - use None (not 'none')
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
        
        # Choose compatible solver
        if penalty == 'l1':
            solver = 'saga'
        elif penalty == 'l2':
            solver = trial.suggest_categorical('solver_l2', ['lbfgs', 'saga', 'sag', 'newton-cg'])
        elif penalty == 'elasticnet':
            solver = 'saga'
        else:  # penalty is None
            solver = trial.suggest_categorical('solver_none', ['lbfgs', 'saga', 'sag', 'newton-cg'])
        
        params = {
            'penalty': penalty,
            'C': trial.suggest_float('C', 0.01, 100, log=True),
            'solver': solver,
            'max_iter': trial.suggest_int('max_iter', 1000, 3000),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Add l1_ratio only for elasticnet
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
        try:
            model = LogisticRegression(**params)
            mean, std = self._cv_score(model, X_train, y_train)
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
            return mean
        
        except Exception as e:
            logging.warning(f"LR trial failed: {e}")
            return 0.0


    def objective_knn(self, trial, X_train, y_train):
        """Optuna objective for K-Neighbors"""
        
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        }
        
        model = KNeighborsClassifier(**params)
        mean, std = self._cv_score(model, X_train, y_train)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
        return mean


    def objective_xgboost(self, trial, X_train, y_train):
        """Optuna objective for XGBoost"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 2),
            'scale_pos_weight': _pos_weight(y_train),   # ← imbalance handled here
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**params)
        mean, std = self._cv_score(model, X_train, y_train) 
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params) 
            mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
        return mean


    def objective_catboost(self, trial, X_train, y_train):
        """Optuna objective for CatBoost"""
        
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 7),
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128]),
            'auto_class_weights': 'Balanced',   # ← imbalance handled here
            'random_state': 42,
            'verbose': False
        }
        
        model = CatBoostClassifier(**params)
        mean, std = self._cv_score(model, X_train, y_train)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params) 
            mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
        return mean


    def objective_svc(self, trial, X_train, y_train):
        """Optuna objective for SVC"""
        
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': 'balanced',
            'random_state': 42,
            'probability': True  # Required for predict_proba
        }
        
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 4)
        
        model = SVC(**params)
        mean, std = self._cv_score(model, X_train, y_train)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params) 
            mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
        return mean


    def objective_adaboost(self, trial, X_train, y_train):
        """Optuna objective for AdaBoost - Simplified"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'random_state': 42
            # algorithm defaults to 'SAMME' if not specified
        }
        
        
        try:       
            model = AdaBoostClassifier(**params)
            mean, std = self._cv_score(model, X_train, y_train)  
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params) 
                mlflow.log_metrics({'cv_roc_auc_mean': mean, 'cv_roc_auc_std': std})
            return mean
        
        except Exception as e:
            logging.warning(f"AdaBoost trial failed: {e}")
            return 0.0
    

    # ── Model objective registry ────────────────────────────────────────────────────

    def _get_model_objectives(self):
        return {
            'Random Forest':             self.objective_random_forest,
            'Decision Tree':             self.objective_decision_tree,
            'Gradient Boosting':         self.objective_gradient_boosting,
            'Logistic Regression':       self.objective_logistic_regression,
            'K-Neighbors Classifier':    self.objective_knn,
            'XGBoost':                   self.objective_xgboost,
            'CatBoost':                  self.objective_catboost,
            'Support Vector Classifier': self.objective_svc,
            'AdaBoost':                  self.objective_adaboost,
        }
   
    # ==================== MAIN TRAINING FUNCTION ====================
            

    def _build_final_model(self, model_name: str, best_params: dict, y_train: np.ndarray):
        """Reconstruct final model from Optuna best params with clean param names."""

        if model_name == 'Random Forest':
            return RandomForestClassifier(**best_params)

        elif model_name == 'Decision Tree':
            return DecisionTreeClassifier(**best_params)

        elif model_name == 'Gradient Boosting':
            return GradientBoostingClassifier(**best_params)

        elif model_name == 'Logistic Regression':
            clean = {}
            for k, v in best_params.items():
                clean['solver' if k.startswith('solver_') else k] = v
            penalty = clean.get('penalty')
            if penalty == 'l1' and clean.get('solver') not in ['saga', 'liblinear']:
                clean['solver'] = 'saga'
            elif penalty == 'elasticnet':
                clean['solver'] = 'saga'
            clean['n_jobs'] = -1
            return LogisticRegression(**clean)

        elif model_name == 'K-Neighbors Classifier':
            return KNeighborsClassifier(**best_params)

        elif model_name == 'XGBoost':
            best_params['scale_pos_weight'] = _pos_weight(y_train)
            return XGBClassifier(**best_params)

        elif model_name == 'CatBoost':
            return CatBoostClassifier(**best_params)

        elif model_name == 'Support Vector Classifier':
            best_params['probability'] = True
            return SVC(**best_params)

        elif model_name == 'AdaBoost':
            return AdaBoostClassifier(**best_params)

    # ── Core training loop ────────────────────────────────────────────────

    def train_and_evaluate_models(
        self,
        X_train, X_test, y_train, y_test,
        n_trials=50, timeout=None
    ):
        """
        Runs Optuna search for every model.
        Models are trained directly on pre-transformed numpy arrays.
        No pipeline creation here — preprocessing already done.

        Returns
        -------
        results_df   : DataFrame sorted by Test ROC AUC
        best_models  : dict {model_name: fitted model}
        studies      : dict {model_name: optuna.Study}
        """
        model_objectives = self._get_model_objectives()
        results, best_models, studies = [], {}, {}

        logging.info("=" * 80)
        logging.info("STARTING HYPERPARAMETER OPTIMISATION")
        logging.info("=" * 80)

        for model_name, objective_func in model_objectives.items():
            logging.info(f"\nTraining: {model_name}")

            with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                mlflow.set_tag("model_type", model_name)

                study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(seed=42),
                    study_name=f'{model_name}_optimisation'
                )

                study.optimize(
                    lambda trial: objective_func(trial, X_train, y_train),
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True
                )

                best_params   = study.best_params
                best_cv_score = study.best_value

                logging.info(f"Best CV ROC AUC : {best_cv_score:.4f}")
                logging.info(f"Best params     : {best_params}")

                # Train final model on full training data
                final_model = self._build_final_model(model_name, best_params.copy(), y_train)
                final_model.fit(X_train, y_train)

                # Evaluate
                y_train_pred  = final_model.predict(X_train)
                y_train_proba = final_model.predict_proba(X_train)[:, 1]
                train_metrics = evaluate_clf(y_train, y_train_pred, y_train_proba)

                y_test_pred  = final_model.predict(X_test)
                y_test_proba = final_model.predict_proba(X_test)[:, 1]
                test_metrics = evaluate_clf(y_test, y_test_pred, y_test_proba)

                # Log to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metrics({
                    'best_cv_roc_auc':  best_cv_score,
                    'train_accuracy':   train_metrics['accuracy'],
                    'train_f1':         train_metrics['f1_score'],
                    'train_precision':  train_metrics['precision'],
                    'train_recall':     train_metrics['recall'],
                    'train_roc_auc':    train_metrics['roc_auc'],
                    'test_accuracy':    test_metrics['accuracy'],
                    'test_f1':          test_metrics['f1_score'],
                    'test_precision':   test_metrics['precision'],
                    'test_recall':      test_metrics['recall'],
                    'test_roc_auc':     test_metrics['roc_auc'],
                    'overfitting':      train_metrics['roc_auc'] - test_metrics['roc_auc'],
                })

                # Log model — signature uses raw transformed input
                signature = infer_signature(X_train, final_model.predict(X_train))
                mlflow.sklearn.log_model(
                    final_model,
                    f"{model_name.replace(' ', '_')}_model",
                    signature=signature)
                

                results.append({
                    'Model':           model_name,
                    'Run_ID':          model_run.info.run_id,
                    'Best_CV_ROC_AUC': best_cv_score,
                    'Train_Accuracy':  train_metrics['accuracy'],
                    'Train_F1':        train_metrics['f1_score'],
                    'Train_Precision': train_metrics['precision'],
                    'Train_Recall':    train_metrics['recall'],
                    'Train_ROC_AUC':   train_metrics['roc_auc'],
                    'Test_Accuracy':   test_metrics['accuracy'],
                    'Test_F1':         test_metrics['f1_score'],
                    'Test_Precision':  test_metrics['precision'],
                    'Test_Recall':     test_metrics['recall'],
                    'Test_ROC_AUC':    test_metrics['roc_auc'],
                    'Overfitting':     train_metrics['roc_auc'] - test_metrics['roc_auc'],
                })

                best_models[model_name] = final_model
                studies[model_name]     = study

                logging.info(
                    f"Test → Accuracy: {test_metrics['accuracy']:.4f} | "
                    f"F1: {test_metrics['f1_score']:.4f} | "
                    f"ROC AUC: {test_metrics['roc_auc']:.4f}")

        results_df = (pd.DataFrame(results).sort_values('Test_ROC_AUC', ascending=False).reset_index(drop=True))
        return results_df, best_models, studies

    # ── Entry point ───────────────────────────────────────────────────────

    def initiate_model_trainer(self, raw_data):
        try:
            mlflow.set_experiment("Visa_Approval_Model_Training345")
            mlflow.end_run()   # kill any leftover run from a crash

            # === making data ready ===========================================
            raw_data_df = pd.read_csv(raw_data)

            X = raw_data_df.drop('case_status', axis=1)
            y = raw_data_df['case_status']

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded
            )

            # ── Step 1: Preprocessing ─────────────────────────────────
            # Fit on train, transform both splits.
            # X_train_transformed is the clean original training data —
            # used both for model training and as SHAP background.
            (
                X_train_transformed,
                X_test_transformed,
                feature_names,
                preprocessing_pipeline
            ) = self.data_transformation.initiate_data_transformation(X_train_raw, X_test_raw, y_train)

            with mlflow.start_run(run_name="Full_Training_Session"):

                data = mlflow.data.from_pandas(raw_data_df.copy())
                mlflow.log_input(data, "Input_data")

                # ── Step 2: Train all models ──────────────────────────
                results_df, best_models, studies = self.train_and_evaluate_models(
                    X_train_transformed, X_test_transformed,
                    y_train, y_test,
                    n_trials=1,
                    timeout=None)

                # Save comparison csv
                os.makedirs(os.path.dirname(self.model_trainer_config.model_comparison_results_dir), exist_ok=True)
                results_df.to_csv(self.model_trainer_config.model_comparison_results_dir, index=False)
                logging.info("model_comparison_results.csv saved.")

                # ── Step 3: Identify best model ───────────────────────
                best_model_name  = results_df.iloc[0]['Model']
                best_model       = best_models[best_model_name]
                logging.info(f"BEST MODEL: {best_model_name}")

                # ── Step 4: Save artifacts ────────────────────────────
                # saving best model
                best_model_path = self.model_trainer_config.best_pipeline_dir.format(best_model_name.replace(" ", "_"))
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                joblib.dump(best_model, best_model_path)

                # saving label encoder
                os.makedirs(os.path.dirname(self.model_trainer_config.label_encoder_dir), exist_ok=True)
                joblib.dump(label_encoder, self.model_trainer_config.label_encoder_dir)
                logging.info("Best model and label encoder saved.")

                # ── Step 5: Classification report ─────────────────────
                print("\n" + "=" * 80)
                print(f"CLASSIFICATION REPORT — {best_model_name}")
                print("=" * 80)
                y_test_pred = best_model.predict(X_test_transformed)
                print(classification_report(
                    y_test, y_test_pred,
                    target_names=label_encoder.classes_
                ))

                # ── Step 6: SHAP explainability ───────────────────────
                # Background = original X_train_transformed (no synthetic rows)
                shap_explainer = SHAPExplainer(
                    model=best_model,
                    X_train_transformed=X_train_transformed,
                    feature_names=feature_names
                )

                # Global explanation — overall feature importance
                global_contrib = shap_explainer.global_explanation(
                    X_test_transformed=X_test_transformed,
                    save_dir=self.model_trainer_config.global_shap_dir,
                    max_samples = 500)

                # Local explanation — single prediction breakdown (first test record)
                local_contrib = shap_explainer.local_explanation(
                    single_record=X_test_transformed[0:1],
                    save_path = self.model_trainer_config.local_shap_dir)
                
                logging.info(f"Local SHAP (first test record): {local_contrib}")

                # Save SHAP explainer for inference-time use
                os.makedirs(os.path.dirname(self.model_trainer_config.shap_explainer_dir), exist_ok=True)               
                joblib.dump(shap_explainer, self.model_trainer_config.shap_explainer_dir)
                logging.info("SHAP explainer saved.")

        except Exception as e:
            raise CustomException(e, sys)