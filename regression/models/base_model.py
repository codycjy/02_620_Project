"""
Base model interface for all models.
"""

import os
import joblib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, model_name: str, params: Dict[Any, Any] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model.
            params: Dictionary of model parameters.
        """
        self.model_name = model_name
        self.params = params or {}
        self.model = None
        
    @abstractmethod
    def build(self) -> BaseEstimator:
        """
        Build the model with specified parameters.
        
        Returns:
            Initialized model.
        """
        pass
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the model to training data.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
        """
        if self.model is None:
            self.model = self.build()
        
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted labels.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Check if the model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # For models without built-in probability estimation, use predictions
            preds = self.predict(X).astype(float)
            # Return probabilities for both classes (0 and 1)
            return np.vstack((1-preds, preds)).T
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Generate predictions
        y_pred = self.predict(X_test)
        
        # For AUC, we need probabilities
        try:
            y_proba = self.predict_proba(X_test)[:, 1]  # Probability of class 1
        except (AttributeError, IndexError):
            # If probabilities aren't available, use binary predictions
            y_proba = y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Only calculate AUC if we have multiple classes in the ground truth
        if len(np.unique(y_test)) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_proba)
        
        return metrics
    
    def save(self, model_dir: str) -> str:
        """
        Save the model to disk.
        
        Args:
            model_dir: Directory to save the model.
            
        Returns:
            Path to saved model.
        """
        if self.model is None:
            raise ValueError("No model to save.")
            
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_path)
        
        return model_path
    
    def load(self, model_path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model.
        """
        self.model = joblib.load(model_path)