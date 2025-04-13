"""
Utility functions for the cell prediction package.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np


def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging.
    
    Args:
        log_dir: Directory to store log files.
        log_level: Logging level.
        
    Returns:
        Logger instance.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('cell_prediction')
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'run_{timestamp}.log'))
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: Results dictionary.
        output_path: Path to save results.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy arrays and other non-serializable objects to lists
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    # Process the results dictionary
    serializable_results = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)


def timer(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time.
        
    Returns:
        Wrapped function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper


def create_output_dirs(base_dir: str, subdirs: List[str]) -> Dict[str, str]:
    """
    Create output directories.
    
    Args:
        base_dir: Base directory.
        subdirs: List of subdirectories to create.
        
    Returns:
        Dictionary mapping subdirectory names to their paths.
    """
    paths = {}
    
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path
    
    return paths


def format_metrics(metrics: Dict[str, float], decimal_places: int = 4) -> Dict[str, str]:
    """
    Format metric values as strings with specified decimal places.
    
    Args:
        metrics: Dictionary of metric names and values.
        decimal_places: Number of decimal places to round to.
        
    Returns:
        Dictionary of formatted metric strings.
    """
    return {k: f"{v:.{decimal_places}f}" for k, v in metrics.items()}