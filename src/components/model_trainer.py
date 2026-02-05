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

from src.utils import evaluate_clf
from src.components.data_transformation import DataTransformation

# ===================== MLflow =====================
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ===================== Dagshub =====================
import dagshub
dagshub.init(repo_owner='vaibhavsshinde1998', repo_name='MLOPs_project', mlflow=True)

# remote server
mlflow.set_tracking_uri("https://dagshub.com/vaibhavsshinde1998/MLOPs_project.mlflow")





class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.data_transformation = DataTransformation()

    # ==================== OPTUNA OBJECTIVE FUNCTIONS ====================

    def objective_random_forest(self, trial, X_train, y_train):
        """Optuna objective for Random Forest"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        
        model = RandomForestClassifier(**params)
        pipeline = self.data_transformation.create_pipeline(model)
        
        # Use cross-validation for robust evaluation
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        return cv_scores.mean()


    def objective_decision_tree(self, trial, X_train, y_train):
        """Optuna objective for Decision Tree"""
        
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            'random_state': 42
        }
        
        model = DecisionTreeClassifier(**params)
        pipeline = self.data_transformation.create_pipeline(model)
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        return cv_scores.mean()


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
        pipeline = self.data_transformation.create_pipeline(model)
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        return cv_scores.mean()


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
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Add l1_ratio only for elasticnet
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
        try:
            model = LogisticRegression(**params)
            pipeline = self.data_transformation.create_pipeline(model)
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                        cv=3, scoring='roc_auc', n_jobs=-1)
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
                mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

            return cv_scores.mean()
        
        except Exception as e:
            print(f"⚠️ Trial failed with parameters: {params}")
            print(f"   Error: {e}")
            return 0.0


    def objective_knn(self, trial, X_train, y_train):
        """Optuna objective for K-Neighbors"""
        
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        }
        
        model = KNeighborsClassifier(**params)
        pipeline = self.data_transformation.create_pipeline(model)
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        return cv_scores.mean()


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
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**params)
        pipeline = self.data_transformation.create_pipeline(model)
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        return cv_scores.mean()


    def objective_catboost(self, trial, X_train, y_train):
        """Optuna objective for CatBoost"""
        
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 7),
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128]),
            'random_state': 42,
            'verbose': False
        }
        
        model = CatBoostClassifier(**params)
        pipeline = self.data_transformation.create_pipeline(model)
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        return cv_scores.mean()


    def objective_svc(self, trial, X_train, y_train):
        """Optuna objective for SVC"""
        
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'random_state': 42,
            'probability': True  # Required for predict_proba
        }
        
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 4)
        
        model = SVC(**params)
        pipeline = self.data_transformation.create_pipeline(model)
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        return cv_scores.mean()


    def objective_adaboost(self, trial, X_train, y_train):
        """Optuna objective for AdaBoost - Simplified"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'random_state': 42
            # algorithm defaults to 'SAMME' if not specified
        }
        
        model = AdaBoostClassifier(**params)
        pipeline = self.data_transformation.create_pipeline(model)
        
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                        cv=3, scoring='roc_auc', n_jobs=-1)
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
                mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

            return cv_scores.mean()
        
        except Exception as e:
            print(f"⚠️ AdaBoost trial failed: {e}")
            return 0.0
    
   
    # ==================== MAIN TRAINING FUNCTION ====================
            

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, n_trials=50, timeout=None):
        '''
        Gives every models (best param version) metrics, pipeline, study.
        '''
        # Define models and their objective functions
        model_objectives = {
            'Random Forest': self.objective_random_forest,
            'Decision Tree': self.objective_decision_tree,
            'Gradient Boosting': self.objective_gradient_boosting,
            'Logistic Regression': self.objective_logistic_regression,
            'K-Neighbors Classifier': self.objective_knn,
            'XGBoost': self.objective_xgboost,
            'CatBoost': self.objective_catboost,
            'Support Vector Classifier': self.objective_svc,
            'AdaBoost': self.objective_adaboost
        }
        
        results = []
        best_models = {}
        studies = {}
        
        print("="*80)
        print("STARTING HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("="*80)
        
        for model_name, objective_func in model_objectives.items():
            print(f"\n{'='*80}")
            print(f"Training: {model_name}")
            print(f"{'='*80}")

            # Start parent MLflow run for this model
            with mlflow.start_run(run_name=model_name, nested=True) as model_run:

                # Log model type
                mlflow.set_tag("model_type", model_name)
            
                # Create Optuna study
                study = optuna.create_study(
                    direction='maximize',  # Maximize ROC AUC
                    sampler=TPESampler(seed=42),
                    study_name=f'{model_name}_optimization'
                )
            
                # Optimize
                study.optimize(
                    lambda trial: objective_func(trial, X_train, y_train),
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True
                )
            
                # Get best parameters
                best_params = study.best_params
                best_cv_score = study.best_value
                
                print(f"\n✅ Best CV ROC AUC: {best_cv_score:.4f}")
                print(f"Best Parameters: {best_params}")
            
                # Train final model with best parameters on full training data
                if model_name == 'Random Forest':
                    final_model = RandomForestClassifier(**best_params)
                elif model_name == 'Decision Tree':
                    final_model = DecisionTreeClassifier(**best_params)
                elif model_name == 'Gradient Boosting':
                    final_model = GradientBoostingClassifier(**best_params)

                elif model_name == 'Logistic Regression':
                    # Clean up parameter names (remove optuna suffixes like 'solver_l2')
                    clean_params = {}
                    for key, value in best_params.items():
                        if key.startswith('solver_'):
                            clean_params['solver'] = value
                        else:
                            clean_params[key] = value
                    
                    # Ensure compatibility
                    penalty = clean_params.get('penalty')
                    if penalty == 'l1' and clean_params.get('solver') not in ['saga', 'liblinear']:
                        clean_params['solver'] = 'saga'
                    elif penalty == 'elasticnet':
                        clean_params['solver'] = 'saga'
                    
                    # Add n_jobs
                    clean_params['n_jobs'] = -1
                    
                    final_model = LogisticRegression(**clean_params)

                elif model_name == 'K-Neighbors Classifier':
                    final_model = KNeighborsClassifier(**best_params)
                elif model_name == 'XGBoost':
                    final_model = XGBClassifier(**best_params)
                elif model_name == 'CatBoost':
                    final_model = CatBoostClassifier(**best_params)
                elif model_name == 'Support Vector Classifier':
                    # Ensure probability=True for SVC
                    best_params['probability'] = True
                    final_model = SVC(**best_params)
                elif model_name == 'AdaBoost':
                    final_model = AdaBoostClassifier(**best_params)
                
                # Create pipeline with best model
                best_pipeline = self.data_transformation.create_pipeline(final_model)
                
                # Train on full training set
                best_pipeline.fit(X_train, y_train)
            
                # Evaluate on training set
                y_train_pred = best_pipeline.predict(X_train)
                y_train_proba = best_pipeline.predict_proba(X_train)[:, 1]
                train_metrics = evaluate_clf(y_train, y_train_pred, y_train_proba)
                
                # Evaluate on test set
                y_test_pred = best_pipeline.predict(X_test)
                y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
                test_metrics = evaluate_clf(y_test, y_test_pred, y_test_proba)

                # Log best parameters to parent run
                mlflow.log_params(best_params)
                
                # Log all metrics to parent run
                mlflow.log_metric("best_cv_roc_auc", best_cv_score)
                mlflow.log_metric("train_accuracy", train_metrics['accuracy'])
                mlflow.log_metric("train_f1", train_metrics['f1_score'])
                mlflow.log_metric("train_precision", train_metrics['precision'])
                mlflow.log_metric("train_recall", train_metrics['recall'])
                mlflow.log_metric("train_roc_auc", train_metrics['roc_auc'])
                mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
                mlflow.log_metric("test_f1", test_metrics['f1_score'])
                mlflow.log_metric("test_precision", test_metrics['precision'])
                mlflow.log_metric("test_recall", test_metrics['recall'])
                mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
                mlflow.log_metric("overfitting", train_metrics['roc_auc'] - test_metrics['roc_auc'])
                
            

                # Log the model
                signature = infer_signature(X_train, best_pipeline.predict(X_train))

                mlflow.sklearn.log_model(
                    best_pipeline,
                    f"{model_name.replace(' ', '_')}_pipeline",
                    signature=signature
                )

            
                # Store results
                results.append({
                    'Model': model_name,
                    'Run_ID': model_run.info.run_id,
                    'Best_CV_ROC_AUC': best_cv_score,
                    'Train_Accuracy': train_metrics['accuracy'],
                    'Train_F1': train_metrics['f1_score'],
                    'Train_Precision': train_metrics['precision'],
                    'Train_Recall': train_metrics['recall'],
                    'Train_ROC_AUC': train_metrics['roc_auc'],
                    'Test_Accuracy': test_metrics['accuracy'],
                    'Test_F1': test_metrics['f1_score'],
                    'Test_Precision': test_metrics['precision'],
                    'Test_Recall': test_metrics['recall'],
                    'Test_ROC_AUC': test_metrics['roc_auc'],
                    'Overfitting': train_metrics['roc_auc'] - test_metrics['roc_auc']
                })
            
                # Store best model and study
                best_models[model_name] = best_pipeline
                studies[model_name] = study
                
                print(f"\nTest Performance:")
                print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
                print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test_ROC_AUC', ascending=False).reset_index(drop=True)
        
        return results_df, best_models, studies
    

    def initiate_model_trainer(self, raw_data):
        try:
    
            # ==================== EXECUTION ====================

            # Set MLflow experiment
            mlflow.set_experiment("Visa_Approval_Model_Training_7")

            # Kill any leftover active run from previous crash
            mlflow.end_run()

            raw_data_df = pd.read_csv(raw_data)

            # Prepare data
            X = raw_data_df.drop('case_status', axis=1)
            y = raw_data_df['case_status']

            # Encode target
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=0.2, 
                random_state=42,
                stratify=y_encoded
            )


            # Start one top-level run for the entire session
            with mlflow.start_run(run_name="Full_Training_Session"):

                    
                # logging input data in mlflow
                data = mlflow.data.from_pandas(raw_data_df.copy())
                mlflow.log_input(data,"Input_data")
                

                # Run optimization (adjust n_trials based on your time constraints)
                results_df, best_models, studies = self.train_and_evaluate_models(
                    X_train, X_test, y_train, y_test,
                    n_trials=1,  # Reduce for faster testing, increase for better results
                    timeout=None  # Or set timeout in seconds, e.g., 300 for 5 minutes per model
                )

                # Save model comparison results
                os.makedirs(os.path.dirname(self.model_trainer_config.model_comparison_results_dir), exist_ok=True)
                results_df.to_csv(self.model_trainer_config.model_comparison_results_dir, index=False)
                logging.info(fr"model_comparison_results.csv saved.")

                # Get best model
                best_model_name = results_df.iloc[0]['Model']
                best_model_pipeline = best_models[best_model_name]
                best_run_id = results_df.iloc[0]['Run_ID']


                logging.info(f"\n🏆 BEST MODEL: {best_model_name}")

                
                # Save best model
                best_pipeline_path = self.model_trainer_config.best_pipeline_dir.format(best_model_name.replace(" ", "_"))
                os.makedirs(os.path.dirname(best_pipeline_path), exist_ok=True)
                joblib.dump(best_model_pipeline, best_pipeline_path)

                # Save label encoder
                os.makedirs(os.path.dirname(self.model_trainer_config.label_encoder_dir), exist_ok=True)
                joblib.dump(label_encoder, self.model_trainer_config.label_encoder_dir)

                logging.info(fr"\n✅ Best model and label encoder saved!")  
        
        
                # Detailed classification report for best model
                print("\n" + "="*80)
                print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
                print("="*80)
                y_test_pred = best_model_pipeline.predict(X_test)
                print(classification_report(y_test, y_test_pred, 
                                        target_names=label_encoder.classes_))
            
        except Exception as e:
            raise CustomException(e, sys)