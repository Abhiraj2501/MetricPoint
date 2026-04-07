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


# -------------------- TRAIN OR LOAD --------------------

# If model doesn't exist → train
if not os.path.exists(MODEL_FILE):

    # -------- TRAINING PHASE --------

    # Load dataset
    housing = pd.read_csv("housing.csv")

    # Create income category for stratified sampling
    housing['income_cat'] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Stratified split → keeps income distribution same in train/test
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, _ in split.split(housing, housing['income_cat']):
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    # Separate labels (target variable)
    housing_labels = housing["median_house_value"].copy()

    # Drop label from features
    housing_features = housing.drop("median_house_value", axis=1)

    # Identify numerical and categorical columns
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # Build preprocessing pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)

    # Apply transformations (fit + transform)
    housing_prepared = pipeline.fit_transform(housing_features)

    # Train Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save trained model and pipeline for future use
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.")


else:
    # -------- INFERENCE PHASE --------

    # Load saved model and pipeline
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Load new unseen data for prediction
    input_data = pd.read_csv("input.csv")

    # Apply same preprocessing as training
    transformed_input = pipeline.transform(input_data)

    # Predict house prices
    predictions = model.predict(transformed_input)

    # Add predictions to dataframe
    input_data["median_house_value"] = predictions

    # Save results
    input_data.to_csv("output.csv", index=False)

    print("Inference complete. Results saved to output.csv")