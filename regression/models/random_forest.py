"""
Random Forest model implementation.
"""

from typing import Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

from models.base_model import BaseModel
from config import RF_PARAMS


class RandomForestModel(BaseModel):
    """Random Forest model for classification."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Random Forest model.
        
        Args:
            params: Parameters for the model.
        """
        # Merge default parameters with any provided parameters
        model_params = RF_PARAMS.copy()
        if params:
            model_params.update(params)
            
        super().__init__("random_forest", model_params)
        
    def build(self) -> BaseEstimator:
        """
        Build the Random Forest model.
        
        Returns:
            Initialized RandomForestClassifier model.
        """
        return RandomForestClassifier(**self.params)
        
    def get_feature_importance(self) -> Dict[int, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping feature indices to importance values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        importances = self.model.feature_importances_
        return {i: importance for i, importance in enumerate(importances)}