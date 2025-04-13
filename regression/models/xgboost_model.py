"""
XGBoost model implementation. Should be replaced by our own package.
"""


from typing import Dict, Any

import numpy as np
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator

from models.base_model import BaseModel
from config import XGB_PARAMS


class XGBoostModel(BaseModel):
    """XGBoost model for classification."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the XGBoost model.
        
        Args:
            params: Parameters for the model.
        """
        # Merge default parameters with any provided parameters
        model_params = XGB_PARAMS.copy()
        if params:
            model_params.update(params)
            
        super().__init__("xgboost", model_params)
        
    def build(self) -> BaseEstimator:
        """
        Build the XGBoost model.
        
        Returns:
            Initialized XGBClassifier model.
        """
        return XGBClassifier(**self.params)
        
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