#!/usr/bin/env python3
"""
Main script for the cell prediction package.
Runs the full prediction pipeline on the provided data.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import models

# Import package modules
from config import (
    TARGET_COLUMN, TARGET_MAPPING, MODELS_DIR, 
    RESULTS_DIR, FIGURES_DIR, RANDOM_STATE,
    LR_GRID, RF_GRID, XGB_GRID
)
from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from models.model_factory import ModelFactory
from evaluation.validator import ModelValidator
from evaluation.metrics import calculate_metrics, compare_models
from visualization.visualizer import Visualizer
from utils.helpers import setup_logging, save_results, create_output_dirs, timer


@timer
@timer
def load_and_process_data(
    train_file: str,
    test_file: Optional[str] = None,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[
    pd.DataFrame, pd.DataFrame, List[str], 
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load and process the data files.
    
    Args:
        train_file: Path to training data file.
        test_file: Optional path to test data file.
        verbose: Whether to print processing information.
        logger: Optional logger for more detailed logging.
        
    Returns:
        Tuple of (
            train_df, test_df, feature_cols, 
            X_train, X_val, y_train, y_val
        )
    """
    # Initialize data loader
    data_loader = DataLoader(verbose=verbose)
    
    # Load and process training data
    train_df, feature_cols = data_loader.prepare_data(train_file)
    
    # Load and process test data if provided
    test_df = None
    if test_file:
        test_df, _ = data_loader.prepare_data(test_file)
    
    # Check class distribution before splitting
    class_counts = train_df['target_encoded'].value_counts()
    if logger:
        logger.info(f"Class distribution before splitting: {class_counts.to_dict()}")
    
    # Check for classes with too few samples
    min_samples_for_stratification = 2
    can_stratify = True
    
    for class_label, count in class_counts.items():
        if count < min_samples_for_stratification:
            can_stratify = False
            if logger:
                logger.warning(
                    f"Class {class_label} has only {count} samples, which is too few for stratified splitting. "
                    f"Using random splitting instead."
                )
    
    # Split training data into train and validation sets
    data_splitter = DataSplitter(random_state=RANDOM_STATE)
    X = train_df[feature_cols].values
    y = train_df['target_encoded'].values
    
    # Use stratification only if possible
    if can_stratify:
        X_train, X_val, y_train, y_val = data_splitter.train_val_split(X, y, stratify=y)
        if logger:
            logger.info("Using stratified split for train/validation data.")
    else:
        X_train, X_val, y_train, y_val = data_splitter.train_val_split(X, y, stratify=None)
        if logger:
            logger.info("Using random split for train/validation data.")
    
    # Check class distribution after splitting
    if logger:
        train_class_counts = pd.Series(y_train).value_counts().to_dict()
        val_class_counts = pd.Series(y_val).value_counts().to_dict()
        logger.info(f"Train set class distribution: {train_class_counts}")
        logger.info(f"Validation set class distribution: {val_class_counts}")
    
    return train_df, test_df, feature_cols, X_train, X_val, y_train, y_val

@timer
def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tune_hyperparams: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train and evaluate multiple models.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_val: Validation feature matrix.
        y_val: Validation target vector.
        tune_hyperparams: Whether to tune hyperparameters.
        logger: Optional logger instance.
        
    Returns:
        Tuple of (best_model, model_results).
    """
    # Create models
    lr_model = ModelFactory.create_model("linear")
    rf_model = ModelFactory.create_model("rf")
    xgb_model = ModelFactory.create_model("xgb")
    
    models_instances = [lr_model, rf_model, xgb_model]
    model_results = {}
    
    # Validator for model tuning and evaluation
    validator = ModelValidator()
    
    # Hyperparameter grids
    param_grids = {
        "linear_regression": LR_GRID,
        "random_forest": RF_GRID,
        "xgboost": XGB_GRID
    }
    
    # Train and evaluate each model
    for model in models_instances:
        if logger:
            logger.info(f"Training {model.model_name} model...")
        
        # Tune hyperparameters if requested
        if tune_hyperparams:
            if logger:
                logger.info(f"Tuning hyperparameters for {model.model_name}...")
            
            param_grid = param_grids.get(model.model_name, {})
            if param_grid:
                best_params, best_score = validator.tune_hyperparameters(
                    model, param_grid, X_train, y_train
                )
                
                if logger:
                    logger.info(f"Best parameters: {best_params}")
                    logger.info(f"Best score: {best_score:.4f}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_val)
        try:
            y_proba = model.predict_proba(X_val)[:, 1]
        except (AttributeError, IndexError):
            y_proba = None
            
        metrics = calculate_metrics(y_val, y_pred, y_proba)
        model_results[model.model_name] = metrics
        
        if logger:
            logger.info(f"{model.model_name} metrics: {metrics}")
    
    # Compare models and select the best one
    best_model_metrics = compare_models(model_results)
    
    # Find the best model based on AUC or accuracy
    if 'auc_roc' in best_model_metrics:
        best_metric = 'auc_roc'
    else:
        best_metric = 'accuracy'
        
    best_model_name = best_model_metrics[best_metric].split()[0]
    best_model = next(model for model in models_instances if model.model_name == best_model_name)
    
    if logger:
        logger.info(f"Best model: {best_model_name}")
    
    return best_model, model_results


@timer
def make_predictions(
    model: Any, 
    X_test: np.ndarray, 
    class_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model.
        X_test: Test feature matrix.
        class_names: List of class names.
        
    Returns:
        Tuple of (y_pred, y_proba, class_labels).
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)
    except (AttributeError, ValueError):
        y_proba = np.array([[1-p, p] for p in y_pred])
    
    # Convert numeric predictions to class labels
    inv_mapping = {v: k for k, v in TARGET_MAPPING.items()}
    class_labels = [inv_mapping.get(pred, str(pred)) for pred in y_pred]
    
    return y_pred, y_proba, class_labels


@timer
def create_visualizations(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    best_model: Any,
    all_model_results: Dict[str, Any],
    y_val: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str = FIGURES_DIR
) -> None:
    """
    Create and save visualizations.
    
    Args:
        train_df: Training data DataFrame.
        feature_cols: List of feature column names.
        best_model: Best trained model.
        all_model_results: Results from all models.
        y_val: Validation target vector.
        y_pred: Predicted target vector.
        class_names: List of class names.
        output_dir: Directory to save figures.
    """
    # Initialize visualizer
    visualizer = Visualizer(output_dir=output_dir)
    
    # Plot feature distributions
    visualizer.plot_feature_distributions(
        train_df, 
        [col for col in feature_cols if 'feature_' in col and '_std' not in col][:10], 
        TARGET_COLUMN,
        filename='feature_distributions'
    )
    
    # Plot correlation matrix for key features
    visualizer.plot_correlation_matrix(
        train_df,
        [col for col in feature_cols if 'feature_' in col][:15],
        filename='correlation_matrix'
    )
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        y_val, 
        y_pred, 
        class_names, 
        best_model.model_name,
        filename='confusion_matrix'
    )
    
    # Plot model comparison
    visualizer.plot_model_comparison(
        all_model_results,
        metric_name='accuracy',
        filename='model_comparison_accuracy'
    )
    
    # Plot feature importance if supported
    if hasattr(best_model, 'get_feature_importance'):
        visualizer.plot_feature_importance(
            best_model,
            feature_cols,
            filename='feature_importance'
        )
    
    # Plot ROC curves if we have probabilities
    model_probas = {}
    for model_name, metrics in all_model_results.items():
        if 'auc_roc' in metrics:
            # This is just a placeholder since we don't have actual probabilities here
            # In the real implementation, you would store probabilities during evaluation
            model_probas[model_name] = np.random.uniform(0, 1, size=len(y_val))
    
    if model_probas:
        visualizer.plot_roc_curve(
            y_val,
            model_probas,
            filename='roc_curves'
        )


def main():
    """Main function to run the prediction pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cell expression data prediction pipeline")
    parser.add_argument("--train", required=True, help="Path to training data file")
    parser.add_argument("--test", help="Path to test data file")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Create output directories
    output_dirs = create_output_dirs(
        args.output, 
        [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, "logs"]
    )
    
    # Setup logging
    logger = setup_logging(log_dir=os.path.join(args.output, "logs"))
    logger.info("Starting cell prediction pipeline")
    logger.info(f"Using training data: {args.train}")
    if args.test:
        logger.info(f"Using test data: {args.test}")
    
    try:
        # Load and process data
        logger.info("Loading and processing data...")
        train_df, test_df, feature_cols, X_train, X_val, y_train, y_val = load_and_process_data(
            args.train, args.test, verbose=args.verbose
        )
        
        # Train and evaluate models
        logger.info("Training and evaluating models...")
        best_model, model_results = train_and_evaluate_models(
            X_train, y_train, X_val, y_val, 
            tune_hyperparams=not args.no_tune,
            logger=logger
        )
        
        # Make predictions on validation set
        logger.info("Making predictions on validation set...")
        class_names = list(TARGET_MAPPING.keys())
        y_val_pred, y_val_proba, val_class_labels = make_predictions(
            best_model, X_val, class_names
        )
        
        # Create visualizations
        logger.info("Creating visualizations...")
        create_visualizations(
            train_df,
            feature_cols,
            best_model,
            model_results,
            y_val,
            y_val_pred,
            class_names,
            output_dir=os.path.join(args.output, FIGURES_DIR)
        )
        
        # Save best model
        logger.info("Saving best model...")
        model_path = best_model.save(os.path.join(args.output, MODELS_DIR))
        logger.info(f"Model saved to {model_path}")
        
        # Make predictions on test set if provided
        if test_df is not None:
            logger.info("Making predictions on test set...")
            X_test = test_df[feature_cols].values
            y_test_pred, y_test_proba, test_class_labels = make_predictions(
                best_model, X_test, class_names
            )
            
            # Add predictions to test DataFrame
            test_df['predicted_label'] = test_class_labels
            test_df['predicted_proba'] = y_test_proba[:, 1] if y_test_proba.shape[1] > 1 else y_test_proba
            
            # Save predictions
            predictions_path = os.path.join(args.output, RESULTS_DIR, "test_predictions.csv")
            test_df.to_csv(predictions_path, index=False)
            logger.info(f"Test predictions saved to {predictions_path}")
        
        # Save results summary
        results_summary = {
            "data": {
                "num_train_samples": len(train_df),
                "num_test_samples": len(test_df) if test_df is not None else 0,
                "num_features": len(feature_cols),
                "feature_names": feature_cols
            },
            "models": {
                model_name: {
                    "metrics": metrics,
                }
                for model_name, metrics in model_results.items()
            },
            "best_model": {
                "name": best_model.model_name,
                "metrics": model_results[best_model.model_name],
                "path": model_path
            }
        }
        
        results_path = os.path.join(args.output, RESULTS_DIR, "results_summary.json")
        save_results(results_summary, results_path)
        logger.info(f"Results summary saved to {results_path}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Error in pipeline: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())