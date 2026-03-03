# ===================== Core Libraries =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import date
from src.utils import load_config

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

# ===================== Imbalanced Learning =====================
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ===================== Model Persistence =====================
import joblib
from src.logger import logging
from src.exception import CustomException
from src.config import *
import os
import sys
from dataclasses import dataclass


# ==================== CUSTOM TRANSFORMERS ====================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""

    # Add this line for dagshub
    __module__ = "src.components.data_transformation"

    def __init__(self):
        pass
            
    def fit(self, X, y=None):
        return self
            
    def transform(self, X):
        X = X.copy()

        # Calculate company age
        from datetime import date
        current_year = date.today().year

        X['company_age'] = current_year - X['yr_of_estab']
                
        # Drop irrelevant columns
        X = X.drop(['yr_of_estab', 'case_id'], axis=1)
                
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Cap outliers using IQR method"""

    # Add this line for dagshub
    __module__ = "src.components.data_transformation"
    
    def __init__(self, method='cap', factor=1.5):
        self.method = method
        self.factor = factor
        self.lower_bounds = {}
        self.upper_bounds = {}
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                self.lower_bounds[col] = Q1 - self.factor * IQR
                self.upper_bounds[col] = Q3 + self.factor * IQR
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col in self.lower_bounds.keys():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(
                    lower=self.lower_bounds[col],
                    upper=self.upper_bounds[col]
                )
        
        return X_df if isinstance(X, pd.DataFrame) else X_df.values


class CorrelationRemoverNumeric(BaseEstimator, TransformerMixin):
    """Remove highly correlated NUMERICAL features"""

    # Add this line for dagshub
    __module__ = "src.components.data_transformation"
    
    def __init__(self, threshold=0.9, numerical_cols=None):
        self.threshold = threshold
        self.numerical_cols = numerical_cols
        self.columns_to_drop = []
        self.columns_to_keep = []
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Select only numerical columns for correlation analysis
        if self.numerical_cols:

            existing_numerical_cols = [col for col in self.numerical_cols 
                                       if col in X_df.columns]
            
            X_numeric = X_df[existing_numerical_cols]
        
        else:
            X_numeric = X_df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = X_numeric.corr().abs()
        
        # Get upper triangle
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find correlated features
        self.columns_to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > self.threshold)
        ]
        
        # Keep track of remaining columns
        self.columns_to_keep = [col for col in X_df.columns 
                                if col not in self.columns_to_drop]
        
        if self.columns_to_drop: 
            logging.info(f"CorrelationRemover — dropping: {self.columns_to_drop}")
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_df = X_df[self.columns_to_keep]
        return X_df if isinstance(X, pd.DataFrame) else X_df.values


class DynamicColumnTransformer(BaseEstimator, TransformerMixin):
    """Dynamically adjust preprocessing based on remaining columns after correlation removal"""

    # Add this line for dagshub
    __module__ = "src.components.data_transformation"
    
    def __init__(self, numerical_cols, power_transform_cols, 
                 ordinal_cols, ordinal_categories, nominal_cols):
        
        self.numerical_cols = numerical_cols
        self.power_transform_cols = power_transform_cols
        self.ordinal_cols = ordinal_cols
        self.ordinal_categories = ordinal_categories  # List of lists for each ordinal column
        self.nominal_cols = nominal_cols
        
        self.remaining_numerical_cols = None
        self.remaining_power_cols = None
        self.remaining_ordinal_cols = None
        self.remaining_ordinal_categories = None
        self.remaining_nominal_cols = None
        self.preprocessor = None
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Get all columns that actually exist in the data
        existing_columns = set(X_df.columns)
        
        # Determine remaining numerical columns after correlation removal
        self.remaining_numerical_cols = [
            col for col in self.numerical_cols 
            if col in existing_columns and col not in self.power_transform_cols
        ]
        
        self.remaining_power_cols = [
            col for col in self.power_transform_cols 
            if col in existing_columns
        ]

        # Filter ordinal columns to only those that exist
        self.remaining_ordinal_cols = [
            col for col in self.ordinal_cols 
            if col in existing_columns
        ]
        
        # Filter ordinal categories to match remaining ordinal columns
        self.remaining_ordinal_categories = [
            self.ordinal_categories[i] 
            for i, col in enumerate(self.ordinal_cols) 
            if col in existing_columns
        ]
        
        # Filter nominal columns to only those that exist
        self.remaining_nominal_cols = [
            col for col in self.nominal_cols 
            if col in existing_columns
        ]
        

        # ── Diagnostics ──────────────────────────────────────────────
        logging.info("DynamicColumnTransformer column status:")
        logging.info(f"  Remaining numerical cols (standard): {self.remaining_numerical_cols}")
        logging.info(f"  Remaining power transform cols: {self.remaining_power_cols}")
        logging.info(f"  Remaining ordinal cols: {self.remaining_ordinal_cols}")
        logging.info(f"  Remaining nominal cols: {self.remaining_nominal_cols}")

        
        # Warning messages for dropped columns
        dropped_numerical = set(self.numerical_cols) - set(self.remaining_numerical_cols) - set(self.remaining_power_cols)
        dropped_ordinal = set(self.ordinal_cols) - set(self.remaining_ordinal_cols)
        dropped_nominal = set(self.nominal_cols) - set(self.remaining_nominal_cols)
        
        if dropped_numerical:
            logging.warning(f"  Dropped numerical cols  : {list(dropped_numerical)}")
        if dropped_ordinal:
            logging.warning(f"  Dropped ordinal cols    : {list(dropped_ordinal)}")
        if dropped_nominal:
            logging.warning(f"  Dropped nominal cols    : {list(dropped_nominal)}")

        # ==================== BUILD TRANSFORMERS ====================

        # Build transformers list
        transformers = []
        
        # Standard numerical features (no power transform)
        if self.remaining_numerical_cols:
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_pipeline, self.remaining_numerical_cols))
        
        # Numerical features with power transformation
        if self.remaining_power_cols:
            power_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('power', PowerTransformer(method='yeo-johnson')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('power', power_pipeline, self.remaining_power_cols))
        
        
        # Ordinal categorical features
        if self.remaining_ordinal_cols:
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(
                    categories=self.remaining_ordinal_categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1  # Assign -1 to unknown categories during inference
                ))
            ])
            transformers.append(('ordinal', ordinal_pipeline, self.remaining_ordinal_cols))
        

        # Nominal categorical features
        if self.remaining_nominal_cols:
            nominal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot_encoder', OneHotEncoder(
                    drop='first',  # Avoid multicollinearity
                    sparse_output=False,
                    handle_unknown='ignore'  # Ignore unknown categories during inference
                ))
            ])
            transformers.append(('nominal', nominal_pipeline, self.remaining_nominal_cols))
        
        # Check if we have any transformers
        if not transformers:
            raise ValueError(
                "No columns remain after filtering! "
                "All specified columns have been dropped by upstream transformers."
            )
        

        # Create ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any columns not specified
        )
        
        self.preprocessor.fit(X_df, y)
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.preprocessor.transform(X_df)

    def get_feature_names_out(self):
        """Return feature names after all encoding — used by SHAP explainer."""
        return self.preprocessor.get_feature_names_out()
    

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        

    def build_preprocessing_pipeline(self) -> Pipeline:
        # ==================== Pipeline function ========================================

        try:
            # ==================== DEFINE YOUR COLUMNS ====================

            # Load configuration
            config = load_config(DATA_SCHEMA)
            
            # Access parameters exactly as in your original code
            #all_columns = config['columns']
            numerical_cols = config['numerical_cols']
            categorical_cols = config['categorical_cols']
            power_transform_cols = config['power_transform_cols']
            ordinal_columns = config['ordinal_columns']
            ordinal_categories = config['ordinal_categories']
            nominal_columns = config['nominal_columns']

            #numerical_cols = ['no_of_employees', 'prevailing_wage', 'company_age']
            #categorical_cols = ['continent', 'education_of_employee', 'has_job_experience', 'requires_job_training', 'region_of_employment', 'unit_of_wage', 'full_time_position', 'case_status']
            
            # Columns that need power transformation to handle skewness.
            #power_transform_cols = ['company_age', 'no_of_employees']

            # Ordinal categorical columns
            #ordinal_columns = ['education_of_employee']

            # Define the order for ordinal categories (VERY IMPORTANT!)
            # ordinal_categories = [
            #     ['High School', "Master's", "Bachelor's", 'Doctorate']   # education_of_employee
            #     # Adjust these based on your actual data values and their logical order
            # ]

            # Nominal categorical columns
            #nominal_columns = ['continent', 'has_job_experience', 'requires_job_training', 'region_of_employment', 'unit_of_wage', 'full_time_position']

            
            
            
            preprocessing_pipeline = Pipeline([
                ('feature_engineer', FeatureEngineer()),
                ('outlier_handler', OutlierHandler(method='cap', factor=1.5)),
                ('correlation_remover', CorrelationRemoverNumeric(
                    threshold=0.9, 
                    numerical_cols=numerical_cols
                )),
                ('preprocessor', DynamicColumnTransformer(
                    numerical_cols=numerical_cols,
                    power_transform_cols=power_transform_cols,
                    ordinal_cols=ordinal_columns,
                    ordinal_categories=ordinal_categories,
                    nominal_cols=nominal_columns
                ))
            ])

            
            return preprocessing_pipeline

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, X_train, X_test, y_train):
        """
        Fit preprocessing pipeline on X_train, transform both splits.
        Saves the fitted pipeline to disk as a reusable artifact.
        """
        try:
            logging.info("Building and fitting preprocessing pipeline.")

            preprocessing_pipeline = self.build_preprocessing_pipeline()

            # Fit ONLY on training data — no leakage
            X_train_transformed = preprocessing_pipeline.fit_transform(X_train, y_train)
            X_test_transformed  = preprocessing_pipeline.transform(X_test)

            # Extract feature names for SHAP and analysis
            feature_names = list(
                preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()
            )

            logging.info(
                f"Transformation complete. "
                f"Train shape: {X_train_transformed.shape} | "
                f"Test shape: {X_test_transformed.shape} | "
                f"Features: {len(feature_names)}"
            )

            # Persist fitted pipeline as artifact
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessing_pipeline_dir), exist_ok=True)


            with open(self.data_transformation_config.X_train_transformed_dir, 'wb') as file_obj:
                np.save(file_obj, X_train_transformed)
            
            with open(self.data_transformation_config.X_test_transformed_dir, 'wb') as file_obj:
                np.save(file_obj, X_test_transformed)

            with open(self.data_transformation_config.feature_names_dir, 'w') as file_obj:
                json.dump(feature_names, file_obj)

            joblib.dump(preprocessing_pipeline, self.data_transformation_config.preprocessing_pipeline_dir)
            logging.info(f"Preprocessing pipeline saved")

            return X_train_transformed, X_test_transformed, feature_names, preprocessing_pipeline

        except Exception as e:
            raise CustomException(e, sys)



        
        