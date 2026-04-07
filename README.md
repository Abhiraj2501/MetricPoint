MetricPoint
A data-driven real estate price analysis and prediction system built on Gurgaon's housing market. The goal: give homebuyers and investors a clearer picture of property valuation through structured ML pipelines, not gut instinct.

What This Project Does
California's real estate market moves fast and is notoriously opaque. This project processes raw housing data, cleans it, runs exploratory analysis, and builds a prediction pipeline using scikit-learn to estimate property prices based on key features.
There's also a parallel project-pune module applying the same approach to Pune's housing market.

Project Structure
MetricPoint/
│
├── housing.csv                        # Raw California housing dataset
├── cleaned_data.csv                   # Preprocessed dataset after cleaning
│
├── Analyzingthedata.ipynb             # EDA — distributions, correlations, outliers
├── Visualizingthedata.ipynb           # Visual exploration — heatmaps, scatter plots, geo plots
├── Creating-a-Test-Set.ipynb          # Stratified sampling to create train/test splits
├── HandlingCategoricalAttributes.ipynb # Encoding strategies for categorical features
├── FeatureScaling.ipynb               # Normalization and standardization
├── sklearn-pipelines.ipynb            # End-to-end sklearn Pipeline construction
│
├── main.py                            # Entry point
└── project-pune/                      # Same pipeline applied to Pune housing data

ML Pipeline Overview
The notebooks follow a deliberate, sequential workflow:

Load & Explore — Analyzingthedata.ipynb profiles the dataset: null values, feature distributions, correlation matrix
Visualize — Visualizingthedata.ipynb maps price clusters and surfaces geographic and feature-level patterns
Split — Creating-a-Test-Set.ipynb uses stratified sampling to avoid data snooping bias
Preprocess — HandlingCategoricalAttributes.ipynb and FeatureScaling.ipynb handle encoding and scaling
Pipeline — sklearn-pipelines.ipynb wraps all transformations and the model into a single sklearn Pipeline for clean, reproducible training and inference


Tech Stack

Python 3.x
pandas, NumPy
matplotlib, seaborn
scikit-learn (Pipeline, SimpleImputer, StandardScaler, OrdinalEncoder, OneHotEncoder)
Jupyter Notebook

Run Locally
git clone https://github.com/Abhiraj2501/MetricPoint.git
cd MetricPoint
pip install -r requirements.txt
jupyter notebook


Key Concepts Covered
Stratified train/test splitting (avoiding sampling bias in price brackets)
Handling missing values with imputation strategies
Ordinal vs. one-hot encoding — when to use which
Feature scaling: StandardScaler vs. MinMaxScaler tradeoffs
Building reusable sklearn Pipelines with ColumnTransformer


