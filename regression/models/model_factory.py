"""
Factory for creating model instances.
"""

from typing import Dict, Any, Optional

from models.base_model import BaseModel
from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel


class ModelFactory:
    """Factory class to create model instances based on model name."""
    
    @staticmethod
    def create_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create and return a model instance.
        
        Args:
            model_name: Name of the model to create.
            params: Optional parameters for the model.
            
        Returns:
            Instantiated model.
            
        Raises:
            ValueError: If model_name is not recognized.
        """
        if model_name.lower() in ["linear", "linearregression", "logistic", "logisticregression"]:
            return LinearRegressionModel(params)
        elif model_name.lower() in ["rf", "randomforest"]:
            return RandomForestModel(params)
        elif model_name.lower() in ["xgb", "xgboost"]:
            return XGBoostModel(params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")