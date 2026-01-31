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

    def __init__(self):
        pass
            
    def fit(self, X, y=None):
        return self
            
    def transform(self, X):
        X = X.copy()

        # Calculate company age
        from datetime import date
        todays_date = date.today()
        current_year = todays_date.year

        X['company_age'] = current_year - X['yr_of_estab']
                
        # Drop irrelevant columns
        X = X.drop(['yr_of_estab', 'case_id'], axis=1)
                
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Cap outliers using IQR method"""
    
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
    
    def __init__(self, threshold=0.9, numerical_cols=None):
        self.threshold = threshold
        self.numerical_cols = numerical_cols
        self.columns_to_drop = []
        self.columns_to_keep = []
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Select only numerical columns for correlation analysis
        if self.numerical_cols:
            X_numeric = X_df[self.numerical_cols]
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
            print(f"Dropping correlated columns: {self.columns_to_drop}")
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_df = X_df[self.columns_to_keep]
        return X_df if isinstance(X, pd.DataFrame) else X_df.values


class DynamicColumnTransformer(BaseEstimator, TransformerMixin):
    """Dynamically adjust preprocessing based on remaining columns after correlation removal"""
    
    def __init__(self, numerical_cols, power_transform_cols, 
                 ordinal_cols, ordinal_categories, nominal_cols):
        
        self.numerical_cols = numerical_cols
        self.power_transform_cols = power_transform_cols
        self.ordinal_cols = ordinal_cols
        self.ordinal_categories = ordinal_categories  # List of lists for each ordinal column
        self.nominal_cols = nominal_cols
        
        self.remaining_numerical_cols = None
        self.remaining_power_cols = None
        self.preprocessor = None
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Determine remaining numerical columns after correlation removal
        self.remaining_numerical_cols = [
            col for col in self.numerical_cols 
            if col in X_df.columns and col not in self.power_transform_cols
        ]
        
        self.remaining_power_cols = [
            col for col in self.power_transform_cols 
            if col in X_df.columns
        ]
        
        print(f"\nRemaining numerical cols (standard): {self.remaining_numerical_cols}")
        print(f"Remaining power transform cols: {self.remaining_power_cols}")
        print(f"Ordinal cols: {self.ordinal_cols}")
        print(f"Nominal cols: {self.nominal_cols}")
        
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
        if self.ordinal_cols:
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(
                    categories=self.ordinal_categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1  # Assign -1 to unknown categories during inference
                ))
            ])
            transformers.append(('ordinal', ordinal_pipeline, self.ordinal_cols))
        
        # Nominal categorical features
        if self.nominal_cols:
            nominal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot_encoder', OneHotEncoder(
                    drop='first',  # Avoid multicollinearity
                    sparse_output=False,
                    handle_unknown='ignore'  # Ignore unknown categories during inference
                ))
            ])
            transformers.append(('nominal', nominal_pipeline, self.nominal_cols))
        
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


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        


    def create_pipeline(self, model):
        # ==================== Pipeline function ========================================

        """Create preprocessing + SMOTE + model pipeline"""

        try:
            # ==================== DEFINE YOUR COLUMNS ====================

            numerical_cols = ['no_of_employees', 'prevailing_wage', 'company_age']
            categorical_cols = ['continent', 'education_of_employee', 'has_job_experience', 'requires_job_training', 'region_of_employment', 'unit_of_wage', 'full_time_position', 'case_status']
            
            # Columns that need power transformation to handle skewness.
            power_transform_cols = ['company_age', 'no_of_employees']

            # Ordinal categorical columns
            ordinal_columns = ['education_of_employee']

            # Define the order for ordinal categories (VERY IMPORTANT!)
            ordinal_categories = [
                ['High School', "Master's", "Bachelor's", 'Doctorate']   # education_of_employee
                # Adjust these based on your actual data values and their logical order
            ]

            # Nominal categorical columns
            nominal_columns = ['continent', 'has_job_experience', 'requires_job_training', 'region_of_employment', 'unit_of_wage', 'full_time_position']

            
            
            
            pipeline = ImbPipeline([
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
                )),
                ('smote', SMOTE(
                    sampling_strategy='auto',
                    random_state=42,
                    k_neighbors=5
                )),
                ('model', model)
            ])

            
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)




        
        