"""
Data loading and preprocessing module.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from config import (
    TARGET_COLUMN,
    FEATURE_COLUMN,
    ID_COLUMN,
    TARGET_MAPPING
)


class DataLoader:
    """Class to load and preprocess the cell expression data."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the DataLoader.
        
        Args:
            verbose: Whether to print processing information.
        """
        self.verbose = verbose
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            Loaded DataFrame.
        """
        if self.verbose:
            print(f"Loading data from {file_path}...")
            
        try:
            df = pd.read_csv(file_path)
            if self.verbose:
                print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the mean_expression column by converting string arrays to numeric arrays.
        
        Args:
            df: Input DataFrame with string representations of arrays.
            
        Returns:
            DataFrame with processed features.
        """
        if self.verbose:
            print("Processing mean_expression features...")
            
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Convert string representations to actual lists of floats
        def parse_expression(exp_str: str) -> List[float]:
            try:
                # Replace single quotes with double quotes for JSON parsing
                clean_str = exp_str.replace("'", '"')
                return json.loads(clean_str)
            except Exception as e:
                print(f"Error parsing expression: {exp_str}")
                print(f"Error details: {e}")
                return []
        
        # Create new columns for each feature in the expression array
        expressions = df_processed[FEATURE_COLUMN].apply(parse_expression)
        
        # Determine the number of features from the first non-empty array
        for exp in expressions:
            if exp:
                num_features = len(exp)
                break
        else:
            raise ValueError("No valid expression arrays found")
        
        # Add each feature as a separate column
        for i in range(num_features):
            df_processed[f'feature_{i}'] = expressions.apply(lambda x: x[i] if i < len(x) else np.nan)
            
        if self.verbose:
            print(f"Processed {num_features} features from mean_expression.")
            
        return df_processed
    
    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the target column.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with encoded target.
        """
        if self.verbose:
            print("Encoding target variable...")
            
        df_encoded = df.copy()
        
        # Apply mapping to convert categorical target to numeric
        df_encoded['target_encoded'] = df_encoded[TARGET_COLUMN].map(TARGET_MAPPING)
        
        return df_encoded
    
    def aggregate_by_donor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by donor ID since we want to predict at the donor level.
        
        Args:
            df: Processed DataFrame with features.
            
        Returns:
            Aggregated DataFrame at donor level.
        """
        if self.verbose:
            print("Aggregating data by donor ID...")
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Group by donor ID and target (they should be the same for each donor)
        grouped = df.groupby([ID_COLUMN, TARGET_COLUMN])
        
        # Aggregate statistics for each feature
        agg_data = []
        
        for (donor_id, target), group in grouped:
            # Extract just the encoded target
            target_encoded = TARGET_MAPPING[target]
            
            donor_data = {
                ID_COLUMN: donor_id,
                TARGET_COLUMN: target,
                'target_encoded': target_encoded
            }
            
            # Calculate aggregate statistics for each feature
            for feature in feature_cols:
                values = group[feature].values
                donor_data[f"{feature}_mean"] = np.mean(values)
                donor_data[f"{feature}_std"] = np.std(values)
                donor_data[f"{feature}_min"] = np.min(values)
                donor_data[f"{feature}_max"] = np.max(values)
            
            # Add cluster distribution info
            cluster_counts = group['cluster'].value_counts(normalize=True)
            for cluster, proportion in cluster_counts.items():
                donor_data[f'cluster_{cluster}_proportion'] = proportion
                
            # Add cell count stats
            donor_data['total_cell_count'] = group['cell_count'].sum()
            donor_data['mean_percent_cells'] = group['percent_cells'].mean()
            
            agg_data.append(donor_data)
        
        # Convert to DataFrame
        agg_df = pd.DataFrame(agg_data)
        
        if self.verbose:
            print(f"Aggregated data has {len(agg_df)} rows and {len(agg_df.columns)} columns.")
            
        return agg_df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns from DataFrame.
        
        Args:
            df: Processed DataFrame.
            
        Returns:
            List of feature column names.
        """
        # Exclude ID, original target, and encoded target columns
        exclude_cols = [ID_COLUMN, TARGET_COLUMN, 'target_encoded']
        
        # Get all columns except the excluded ones
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def prepare_data(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Full data preparation pipeline.
        
        Args:
            file_path: Path to the data file.
            
        Returns:
            Tuple of (processed DataFrame, list of feature columns)
        """
        # Load raw data
        df_raw = self.load_data(file_path)
        
        # Process features
        df_processed = self.process_features(df_raw)
        
        # Encode target
        df_encoded = self.encode_target(df_processed)
        
        # Aggregate by donor
        df_aggregated = self.aggregate_by_donor(df_encoded)
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df_aggregated)
        
        return df_aggregated, feature_cols
    
    def aggregate_by_donor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by donor ID and combine mean expressions across all clusters.
        For donors missing certain clusters, use mean values as replacements.
        
        Args:
            df: Processed DataFrame with features.
            
        Returns:
            Aggregated DataFrame at donor level with combined features (10 clusters × 6 genes = 60 features).
        """
        if self.verbose:
            print("Aggregating data by donor ID...")
        
        # Get all unique clusters
        all_clusters = sorted(df['cluster'].unique())
        if self.verbose:
            print(f"Found {len(all_clusters)} unique clusters: {all_clusters}")
        
        # Get feature columns (original gene expression values)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        num_genes = len(feature_cols)
        
        if self.verbose:
            print(f"Each cluster has {num_genes} marker genes")
        
        # First, calculate global means for each cluster and feature to use as replacement values
        cluster_means = {}
        
        for cluster in all_clusters:
            # Filter data for this cluster
            cluster_data = df[df['cluster'] == cluster]
            
            # Calculate mean for each feature
            means = {}
            for feature in feature_cols:
                means[feature] = cluster_data[feature].mean()
            
            cluster_means[cluster] = means
        
        # Group by donor ID and target
        grouped = df.groupby([ID_COLUMN, TARGET_COLUMN])
        
        # Prepare aggregated data
        agg_data = []
        
        for (donor_id, target), group in grouped:
            # Extract encoded target
            target_encoded = TARGET_MAPPING[target]
            
            # Start with basic donor info
            donor_data = {
                ID_COLUMN: donor_id,
                TARGET_COLUMN: target,
                'target_encoded': target_encoded
            }
            
            # Get clusters this donor has
            donor_clusters = set(group['cluster'].unique())
            
            missing_clusters = []
            
            # For each cluster, add features
            for cluster in all_clusters:
                # If donor has this cluster, use actual values
                if cluster in donor_clusters:
                    cluster_group = group[group['cluster'] == cluster]
                    
                    for i, feature in enumerate(feature_cols):
                        feature_name = f"cluster{cluster}_gene{i}"
                        # Take the first value (since there should be only one row per cluster for a donor)
                        donor_data[feature_name] = cluster_group[feature].values[0]
                
                # If donor doesn't have this cluster, use global means
                else:
                    missing_clusters.append(cluster)
                    
                    for i, feature in enumerate(feature_cols):
                        feature_name = f"cluster{cluster}_gene{i}"
                        donor_data[feature_name] = cluster_means[cluster][feature]
            
            if missing_clusters and self.verbose:
                print(f"Donor {donor_id} is missing clusters {missing_clusters}, using global mean values")
            
            # Add cluster distribution info
            cluster_counts = group['cluster'].value_counts(normalize=True)
            for cluster, proportion in cluster_counts.items():
                donor_data[f'cluster_{cluster}_proportion'] = proportion
            
            # For clusters that don't exist for this donor, set proportion to 0
            for cluster in all_clusters:
                if cluster not in cluster_counts:
                    donor_data[f'cluster_{cluster}_proportion'] = 0.0
                    
            # Add cell count stats
            donor_data['total_cell_count'] = group['cell_count'].sum()
            donor_data['mean_percent_cells'] = group['percent_cells'].mean()
            
            agg_data.append(donor_data)
        
        # Convert to DataFrame
        agg_df = pd.DataFrame(agg_data)
        
        # Check total number of features
        feature_prefix = 'cluster'
        gene_features = [col for col in agg_df.columns if col.startswith(feature_prefix) and 'gene' in col]
        if self.verbose:
            print(f"Created {len(gene_features)} combined gene features")
            expected = len(all_clusters) * num_genes
            print(f"Expected features ({len(all_clusters)} clusters × {num_genes} genes): {expected}")
            if len(gene_features) != expected:
                print("Warning: Number of features does not match expected count")
        
        # Check class distribution
        class_counts = agg_df[TARGET_COLUMN].value_counts()
        if self.verbose:
            print(f"Class distribution after aggregation: {class_counts.to_dict()}")
            
        # Check for classes with too few samples
        min_samples = 2
        for class_name, count in class_counts.items():
            if count < min_samples:
                print(f"Warning: Class '{class_name}' has only {count} samples, which may cause issues with stratified splitting.")
                print(f"Consider merging classes or using a different validation approach.")
        
        if self.verbose:
            print(f"Aggregated data has {len(agg_df)} rows and {len(agg_df.columns)} columns.")
            
        return agg_df