"""
Evaluation metrics for model assessment.
"""

from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class.
        
    Returns:
        Dictionary of metric names and values.
    """
    # Basic classification metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # AUC-ROC if probabilities are provided and there are multiple classes
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate a classification report.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        target_names: Names of the target classes.
        
    Returns:
        Classification report as a string.
    """
    return classification_report(y_true, y_pred, target_names=target_names)


def compare_models(model_results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Compare multiple models based on evaluation metrics.
    
    Args:
        model_results: Dictionary mapping model names to their evaluation metrics.
        
    Returns:
        Dictionary indicating the best model for each metric.
    """
    # Dictionary to store the best model for each metric
    best_models = {}
    
    # Get all metrics from the first model
    first_model_name = list(model_results.keys())[0]
    metrics = list(model_results[first_model_name].keys())
    
    # Find the best model for each metric
    for metric in metrics:
        # Get metric values for all models
        values = {model: results[metric] for model, results in model_results.items() if metric in results}
        
        # Find best model (higher is better for our metrics)
        best_model = max(values.items(), key=lambda x: x[1])
        best_models[metric] = f"{best_model[0]} ({best_model[1]:.4f})"
    
    return best_models