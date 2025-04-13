"""
Module for splitting data into training and validation sets.
"""

import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from collections import Counter

from config import RANDOM_STATE, TEST_SIZE, CV_FOLDS


class DataSplitter:
    """Class to handle data splitting for model training and validation."""
    
    def __init__(self, random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE):
        """
        Initialize the DataSplitter.
        
        Args:
            random_state: Random seed for reproducibility.
            test_size: Proportion of data to use for validation.
        """
        self.random_state = random_state
        self.test_size = test_size
        self.logger = logging.getLogger('cell_prediction')
        
    def train_val_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        stratify: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            stratify: Optional array for stratified split.
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val).
        """
        # Check if stratification is possible
        if stratify is not None:
            # Count samples per class
            class_counts = Counter(stratify)
            min_count = min(class_counts.values())
            
            # If any class has only one sample, we can't stratify
            if min_count < 2:
                self.logger.warning(
                    f"Cannot perform stratified split because the least populated class has only {min_count} member(s). "
                    f"Falling back to random split. Class distribution: {dict(class_counts)}"
                )
                stratify = None
        
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
    
    def get_cv_folds(self, n_samples: int, n_folds: int = CV_FOLDS) -> KFold:
        """
        Create cross-validation folds.
        
        Args:
            n_samples: Number of samples in dataset.
            n_folds: Number of folds for cross-validation.
            
        Returns:
            KFold object.
        """
        return KFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
    
    def split_donor_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str = 'target_encoded'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index]:
        """
        Split donor-level data into training and validation.
        
        Args:
            df: DataFrame with donor-level features.
            feature_cols: List of feature column names.
            target_col: Target column name.
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val, train_indices, val_indices).
        """
        # Extract features and target
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Check if stratification is possible
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        
        # If any class has only one sample, we can't stratify
        if min_count < 2:
            self.logger.warning(
                f"Cannot perform stratified split because the least populated class has only {min_count} member(s). "
                f"Falling back to random split. Class distribution: {dict(class_counts)}"
            )
            stratify = None
        else:
            stratify = y
            
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Get indices
        train_indices = df.index[:len(X_train)]
        val_indices = df.index[len(X_train):]
        
        return X_train, X_val, y_train, y_val, train_indices, val_indices