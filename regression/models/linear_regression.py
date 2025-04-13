"""
Linear Regression model implementation.
"""

from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from models.base_model import BaseModel
from config import LR_PARAMS


class LinearRegressionModel(BaseModel):
    """Linear Regression model for classification."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Linear Regression model.
        
        Args:
            params: Parameters for the model.
        """
        # Merge default parameters with any provided parameters
        model_params = LR_PARAMS.copy()
        if params:
            model_params.update(params)
            
        super().__init__("linear_regression", model_params)
        
    def build(self) -> BaseEstimator:
        """
        Build the Logistic Regression model.
        
        Returns:
            Initialized LogisticRegression model.
        """
        # For classification, use LogisticRegression
        return LogisticRegression(**self.params)