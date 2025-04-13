"""
Cross-validation and model validation functionality.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.base import BaseEstimator

from models.base_model import BaseModel
from evaluation.metrics import calculate_metrics
from config import CV_FOLDS, RANDOM_STATE


class ModelValidator:
    """Class for validating models using cross-validation and hyperparameter tuning."""
    
    def __init__(self, n_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        """
        Initialize the ModelValidator.
        
        Args:
            n_folds: Number of folds for cross-validation.
            random_state: Random seed for reproducibility.
        """
        self.n_folds = n_folds
        self.random_state = random_state
        
    def cross_validate(
        self, 
        model: BaseModel, 
        X: np.ndarray, 
        y: np.ndarray, 
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate.
            X: Feature matrix.
            y: Target vector.
            metric: Metric to use for evaluation.
            
        Returns:
            Dictionary with cross-validation results.
        """
        # Build the model if not already built
        if model.model is None:
            model.model = model.build()
        
        # Set up cross-validation
        cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        scores = cross_val_score(
            model.model, X, y, 
            cv=cv, 
            scoring=metric
        )
        
        # Return results
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'all_scores': scores.tolist()
        }
    
    def tune_hyperparameters(
        self, 
        model: BaseModel, 
        param_grid: Dict[str, List[Any]], 
        X: np.ndarray, 
        y: np.ndarray, 
        metric: str = 'accuracy'
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune model hyperparameters using grid search.
        
        Args:
            model: Model to tune.
            param_grid: Grid of parameters to search.
            X: Feature matrix.
            y: Target vector.
            metric: Metric to optimize.
            
        Returns:
            Tuple of (best_params, best_score).
        """
        # Build the model if not already built
        if model.model is None:
            model.model = model.build()
        
        # Set up grid search
        grid_search = GridSearchCV(
            model.model, 
            param_grid, 
            scoring=metric, 
            cv=self.n_folds, 
            n_jobs=-1,  # Use all available processors
            verbose=1
        )
        
        # Perform grid search
        grid_search.fit(X, y)
        
        # Update model with best parameters
        model.params.update(grid_search.best_params_)
        model.model = model.build()
        model.fit(X, y)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def validate_and_select(
        self, 
        models: List[BaseModel], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray, 
        metric: str = 'accuracy'
    ) -> Tuple[BaseModel, Dict[str, Dict[str, float]]]:
        """
        Validate multiple models and select the best one.
        
        Args:
            models: List of models to validate.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            metric: Metric to use for selection.
            
        Returns:
            Tuple of (best_model, all_results).
        """
        results = {}
        
        # Train and evaluate each model
        for model in models:
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_val)
            
            # Get probabilities if available
            try:
                y_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1
            except (AttributeError, IndexError):
                y_proba = None
            
            # Calculate metrics
            model_metrics = calculate_metrics(y_val, y_pred, y_proba)
            
            # Store results
            results[model.model_name] = model_metrics
        
        # Select best model based on the specified metric
        best_model_name = max(results.items(), key=lambda x: x[1][metric])[0]
        best_model = next(model for model in models if model.model_name == best_model_name)
        
        return best_model, results