import os                      # For checking if model file exists
import pandas as pd           # Data handling
import numpy as np            # Numerical operations
import joblib                 # Save/load model and pipeline

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# File names for saving trained model and preprocessing pipeline
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
def build_pipeline(num_attribs, cat_attribs):
    """
    Builds a preprocessing pipeline:
    - Numerical: missing values → median, then scaling
    - Categorical: one-hot encoding
    """

    # Numerical pipeline: handle missing values + normalize scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # Fill missing values with median
        ("scaler", StandardScaler())                     # Standardize features
    ])

    # Categorical pipeline: convert text → one-hot vectors
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))  # Ignore unseen categories at test time
    ])

    # Combine both pipelines
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),   # Apply num pipeline to numerical columns
        ("cat", cat_pipeline, cat_attribs)    # Apply cat pipeline to categorical columns
    ])

    return full_pipeline
