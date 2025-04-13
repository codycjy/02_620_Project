"""
Configuration parameters for the cell prediction package.
"""

# Data processing parameters
TARGET_COLUMN = "Cognitive Status"
FEATURE_COLUMN = "mean_expression"
ID_COLUMN = "Donor ID"
CLUSTER_COLUMN = "cluster"
CELL_COUNT_COLUMN = "cell_count"
PERCENT_CELLS_COLUMN = "percent_cells"

# Target mapping
TARGET_MAPPING = {
    "Dementia": 1,
    "No dementia": 0
}

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2  # For train/validation split
CV_FOLDS = 3     # Number of cross-validation folds

# Model hyperparameters - Linear Regression
LR_PARAMS = {
    "fit_intercept": True
}

# Model hyperparameters - Random Forest
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE
}

# Model hyperparameters - XGBoost
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "random_state": RANDOM_STATE
}

# Hyperparameter grids for tuning
LR_GRID = {
    "fit_intercept": [True, False]
}

RF_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

XGB_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# Output parameters
RESULTS_DIR = "results"
MODELS_DIR = "saved_models"
FIGURES_DIR = "figures"