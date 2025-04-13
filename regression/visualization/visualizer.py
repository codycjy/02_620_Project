
"""
Module for visualizing data and model results.
"""

import os
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from models.base_model import BaseModel


class Visualizer:
    """Class for creating visualizations of data and model results."""
    
    def __init__(self, output_dir: str = 'figures'):
        """
        Initialize the Visualizer.
        
        Args:
            output_dir: Directory to save figures.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        sns.set(style='whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def save_figure(self, fig: plt.Figure, filename: str) -> str:
        """
        Save a figure to disk.
        
        Args:
            fig: Figure to save.
            filename: Filename for the saved figure.
            
        Returns:
            Path to saved figure.
        """
        # Add extension if not present
        if not filename.endswith(('.png', '.jpg', '.pdf')):
            filename += '.png'
            
        # Create path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save figure
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return filepath
        
    def plot_feature_importance(
        self, 
        model: BaseModel, 
        feature_names: List[str], 
        top_n: int = 20, 
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance from a model.
        
        Args:
            model: Trained model.
            feature_names: Names of features.
            top_n: Number of top features to show.
            filename: Optional filename to save the figure.
            
        Returns:
            Figure object.
        """
        # Check if model has feature importance
        if not hasattr(model, 'get_feature_importance'):
            raise ValueError(f"Model {model.model_name} does not support feature importance.")
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Create DataFrame
        imp_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in importance.keys()],
            'Importance': list(importance.values())
        })
        
        # Sort by importance
        imp_df = imp_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax)
        
        # Set labels and title
        ax.set_title(f'Top {top_n} Feature Importance - {model.model_name}', fontsize=16)
        ax.set_xlabel('Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self.save_figure(fig, filename)
            
        return fig
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: List[str], 
        model_name: str,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            class_names: Names of classes.
            model_name: Name of the model.
            filename: Optional filename to save the figure.
            
        Returns:
            Figure object.
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        # Set labels and title
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16)
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('True', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self.save_figure(fig, filename)
            
        return fig
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_scores: Dict[str, np.ndarray], 
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            y_true: True labels.
            y_scores: Dictionary mapping model names to predicted probabilities.
            filename: Optional filename to save the figure.
            
        Returns:
            Figure object.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each model
        for model_name, scores in y_scores.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Set labels and title
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12)
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self.save_figure(fig, filename)
            
        return fig
    
    def plot_model_comparison(
        self, 
        metrics: Dict[str, Dict[str, float]], 
        metric_name: str = 'accuracy',
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of model performance.
        
        Args:
            metrics: Dictionary mapping model names to their metrics.
            metric_name: Metric to use for comparison.
            filename: Optional filename to save the figure.
            
        Returns:
            Figure object.
        """
        # Extract metric values for each model
        models = list(metrics.keys())
        values = [metrics[model][metric_name] for model in models]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot bar chart
        bars = ax.bar(models, values, color=sns.color_palette('viridis', len(models)))
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01, 
                f'{height:.3f}', 
                ha='center', 
                va='bottom',
                fontsize=12
            )
        
        # Set labels and title
        metric_name_capitalized = metric_name.replace('_', ' ').title()
        ax.set_title(f'Model Comparison - {metric_name_capitalized}', fontsize=16)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel(metric_name_capitalized, fontsize=14)
        
        # Set y-axis limits
        ax.set_ylim(0, max(values) * 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self.save_figure(fig, filename)
            
        return fig
    
    def plot_feature_distributions(
    self, 
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str, 
    max_features: int = 10,
    filename: Optional[str] = None
) -> plt.Figure:
        """
        Plot distributions of features by target class.
        
        Args:
            df: DataFrame with features and target.
            feature_cols: List of feature column names.
            target_col: Target column name.
            max_features: Maximum number of features to plot.
            filename: Optional filename to save the figure.
            
        Returns:
            Figure object.
        """
        # Check if feature_cols is empty
        if not feature_cols:
            print("Warning: No features provided for distribution plotting. Skipping.")
            # Create a simple empty figure to return
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No features to plot", ha='center', va='center', fontsize=14)
            plt.axis('off')
            
            # Save if filename provided
            if filename:
                self.save_figure(fig, filename)
                
            return fig
            
        # Sample features if too many
        if len(feature_cols) > max_features:
            sampled_features = np.random.choice(feature_cols, max_features, replace=False)
        else:
            sampled_features = feature_cols
            
        # Check if we still have features after sampling
        if len(sampled_features) == 0:
            print("Warning: No features available after sampling. Skipping distribution plot.")
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No features to plot", ha='center', va='center', fontsize=14)
            plt.axis('off')
            
            # Save if filename provided
            if filename:
                self.save_figure(fig, filename)
                
            return fig
                
        # Get target classes
        classes = df[target_col].unique()
        n_classes = len(classes)
        
        # Create figure
        n_cols = 2
        n_rows = max(1, (len(sampled_features) + n_cols - 1) // n_cols)  # Ensure at least 1 row
        figsize = (16, 5 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes if needed
        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows == 1 and n_cols > 1:
            axes = [axes[0], axes[1]]
        elif n_cols == 1 and n_rows > 1:
            # No action needed, axes is already a 1D array
            pass
        else:  # n_rows == 1 and n_cols == 1
            axes = [axes]
                
        # Plot each feature
        for i, feature in enumerate(sampled_features):
            if i < len(axes):
                ax = axes[i]
                
                # Check if the feature exists in the DataFrame
                if feature not in df.columns:
                    ax.text(0.5, 0.5, f"Feature '{feature}' not found", 
                        ha='center', va='center', fontsize=12)
                    ax.set_title(feature)
                    continue
                    
                # Plot distributions for each class
                for cls in classes:
                    subset = df[df[target_col] == cls]
                    
                    # Skip if no data for this class
                    if len(subset) == 0:
                        continue
                        
                    # Skip if the feature has all NaN values for this class
                    if subset[feature].isna().all():
                        continue
                    
                    try:
                        sns.kdeplot(subset[feature], ax=ax, label=str(cls))
                    except Exception as e:
                        print(f"Warning: Could not plot KDE for feature '{feature}' and class '{cls}': {e}")
                        # Try a histogram instead
                        try:
                            ax.hist(subset[feature], alpha=0.5, label=str(cls))
                        except Exception as e2:
                            print(f"Also could not plot histogram: {e2}")
                            continue
                
                # Set labels
                ax.set_title(feature)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(title=target_col)
                    
        # Remove empty subplots
        for i in range(len(sampled_features), len(axes)):
            fig.delaxes(axes[i])
                
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self.save_figure(fig, filename)
            
        return fig

    def plot_correlation_matrix(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix of features.
        
        Args:
            df: DataFrame with features.
            feature_cols: List of feature column names.
            filename: Optional filename to save the figure.
            
        Returns:
            Figure object.
        """
        # Calculate correlation matrix
        corr = df[feature_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(feature_cols) * 0.5), max(10, len(feature_cols) * 0.5)))
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5, 
            annot=True, 
            fmt='.2f',
            cbar_kws={'shrink': .8},
            ax=ax
        )
        
        # Set title
        ax.set_title('Feature Correlation Matrix', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self.save_figure(fig, filename)
            
        return fig