import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec
import my_kmeans

def load_fold_data(base_path, fold_number, method='kmeans'):
    """Load the kmeans_meta_means.csv file from a specific fold directory"""
    fold_path = os.path.join(base_path, f'fold_{fold_number}', method + '_meta_means.csv')
    return pd.read_csv(fold_path)

def calculate_assignment_matrix(reference_centroids, target_centroids):
    """Calculate cost matrix for Hungarian algorithm (centroid distances)"""
    return pairwise_distances(reference_centroids, target_centroids)

def reassign_clusters(reference_simulation, simulations_data,k):
    """
    Reassign cluster labels to maintain consistency across simulations
    using the Hungarian algorithm (minimum assignment problem)
    """
    # Use the first simulation as reference
    
    
    reassigned_simulations = []
    
    # For each other simulation, find optimal assignment to reference
    
    for sim_idx in range(0, len(simulations_data)):
        target_simulation = simulations_data[sim_idx].copy()
        
        # Calculate assignment cost matrix (distances between centroids)
        cost_matrix = calculate_assignment_matrix(reference_simulation, target_simulation)
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Reorder the target simulation's clusters based on assignment
        reordered_simulation = target_simulation[col_ind, :]
        reassigned_simulations.append(reordered_simulation)
    
    return np.array(reassigned_simulations)

def calculate_pairwise_stability(fold_data):
    """
    Calculate pairwise stability metrics between all simulations in a fold
    Returns a distance matrix
    """
    n_simulations = len(fold_data)
    stability_matrix = np.zeros((n_simulations, n_simulations))
    
    for i in range(n_simulations):
        for j in range(i+1, n_simulations):
            # Calculate average distance between corresponding clusters
            cluster_distances = np.sqrt(np.sum((fold_data[i] - fold_data[j])**2, axis=1))
            avg_distance = np.mean(cluster_distances)
            stability_matrix[i, j] = avg_distance
            stability_matrix[j, i] = avg_distance
    
    return stability_matrix

def calculate_stability_within_fold(fold_data):
    """
    Calculate stability metrics within a fold across different simulations
    """
    n_simulations = len(fold_data)
    n_clusters = fold_data[0].shape[0]
    
    # Calculate pairwise distances between all simulations
    stability_scores = []
    
    for i in range(n_simulations):
        for j in range(i+1, n_simulations):
            # Calculate average distance between corresponding clusters
            cluster_distances = np.sqrt(np.sum((fold_data[i] - fold_data[j])**2, axis=1))
            avg_distance = np.mean(cluster_distances)
            stability_scores.append(avg_distance)
    
    # Lower average distance means higher stability
    mean_stability = np.mean(stability_scores)
    std_stability = np.std(stability_scores)
    
    return {
        'mean_stability': mean_stability,
        'std_stability': std_stability,
        'all_scores': stability_scores
    }

def calculate_stability_across_folds(all_folds_data):
    """
    Calculate stability metrics across all folds
    
    Parameters:
    -----------
    all_folds_data : list of arrays
        List where each item contains cluster centroids from simulations for one fold
    
    Returns:
    --------
    dict
        Dictionary containing stability metrics
    """
    n_folds = len(all_folds_data)
    n_clusters = all_folds_data[0][0].shape[0]
    
    # For each fold, calculate mean centroid position across simulations
    fold_mean_centroids = []
    for fold_data in all_folds_data:
        # Calculate mean across all simulations for this fold
        fold_mean = np.mean(fold_data, axis=0)
        fold_mean_centroids.append(fold_mean)
    
    # Use first fold's mean centroids as reference for reassignment
    reference_centroids = fold_mean_centroids[0]
    
    # Reassign clusters across folds to maintain consistency
    reassigned_centroids = [reference_centroids]  # First fold is already the reference
    
    for i in range(1, n_folds):
        target_centroids = fold_mean_centroids[i]
        
        # Calculate assignment cost matrix (distances between centroids)
        cost_matrix = calculate_assignment_matrix(reference_centroids, target_centroids)
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Reorder the target fold's clusters based on assignment
        reordered_centroids = target_centroids[col_ind, :]
        reassigned_centroids.append(reordered_centroids)
    
    # Now calculate stability metrics across these reassigned mean centroids
    stability_scores = []
    cross_fold_matrix = np.zeros((n_folds, n_folds))
    
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            # Calculate distances between corresponding clusters
            cluster_distances = np.sqrt(np.sum((reassigned_centroids[i] - reassigned_centroids[j])**2, axis=1))
            avg_distance = np.mean(cluster_distances)
            stability_scores.append(avg_distance)
            cross_fold_matrix[i, j] = avg_distance
            cross_fold_matrix[j, i] = avg_distance
    
    # Lower average distance means higher stability
    mean_stability = np.mean(stability_scores)
    std_stability = np.std(stability_scores)
    
    return {
        'mean_stability': mean_stability,
        'std_stability': std_stability,
        'all_scores': stability_scores,
        'cross_fold_matrix': cross_fold_matrix,
        'reassigned_centroids': reassigned_centroids  # Optional: return reassigned centroids for visualization
    }

def visualize_within_fold_stability(fold_stability_matrices, fold_stability_results):
    """Visualize stability within each fold"""
    n_folds = len(fold_stability_matrices)
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
    
    for i, (matrix, result) in enumerate(zip(fold_stability_matrices, fold_stability_results)):
        ax = axes[i] if n_folds > 1 else axes
        sns.heatmap(matrix, annot=False, cmap='viridis_r', ax=ax, fmt='.3f',
                   )
        ax.set_title(f'Fold {i+1} Stability\nMean: {result["mean_stability"]:.4f} ± {result["std_stability"]:.4f}')
        if i == 0:
            ax.set_ylabel('Simulation')
    
    plt.tight_layout()
    return fig

def cluster_on_each_fold(data_folds, n_clusters, n_simulations=10):
    """
    Perform clustering on each fold using the same number of clusters before calculating stability
    
    Parameters:
    -----------
    data_folds : list of arrays
        List containing data for each fold
    n_clusters : int
        Number of clusters to use
    n_simulations : int
        Number of clustering simulations to run per fold
        
    Returns:
    --------
    all_folds_data : list of arrays
        List where each item contains cluster centroids from simulations for one fold
    """
    all_folds_data = []
    for fold_data in data_folds:
        fold_simulations = []
        
        for _ in range(n_simulations):
            # Perform clustering (e.g., K-means)
            kmeans = my_kmeans.KMeans(n_clusters=n_clusters, random_state=None)
            kmeans.fit(fold_data)
            
            # Store the centroids
            fold_simulations.append(kmeans.cluster_centers_)
        
        all_folds_data.append(np.array(fold_simulations))
    
    return all_folds_data

def visualize_within_fold_stability(fold_stability_matrices, fold_stability_results):
    """Visualize stability within each fold"""
    n_folds = len(fold_stability_matrices)
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
    
    for i, (matrix, result) in enumerate(zip(fold_stability_matrices, fold_stability_results)):
        ax = axes[i] if n_folds > 1 else axes
        sns.heatmap(matrix, annot=False, cmap='viridis_r', ax=ax, fmt='.3f',
                   cbar_kws={'label': 'Average Distance (lower is more stable)'})
        ax.set_title(f'Fold {i+1} Stability\nMean: {result["mean_stability"]:.4f} ± {result["std_stability"]:.4f}')
        if i == 0:
            ax.set_ylabel('Simulation')
    
    plt.tight_layout()
    return fig

def visualize_cross_fold_stability(cross_fold_matrix, cross_fold_stability):
    """Visualize stability across folds"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cross_fold_matrix, annot=False, cmap='viridis_r', ax=ax, fmt='.3f',
               cbar_kws={'label': 'Average Distance (lower is more stable)'})
    ax.set_title(f'Cross-Fold Stability\nMean: {cross_fold_stability["mean_stability"]:.4f} ± {cross_fold_stability["std_stability"]:.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Fold')
    
    plt.tight_layout()
    return fig

def visualize_cluster_positions(all_folds_data):
    """Visualize cluster positions across simulations and folds using PCA if needed"""
    n_folds = len(all_folds_data)
    n_clusters = all_folds_data[0][0].shape[0]
    n_features = all_folds_data[0][0].shape[1]
    
    # If we have more than 2 features, we may need dimensionality reduction
    # For now, we'll just select the first 2 features for demonstration
    if n_features >= 2:
        feature_indices = [0, 1]  # Use first two features for plotting
    else:
        raise ValueError("Need at least 2 features for visualization")
    
    # Create a figure with subplots for each cluster
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
    
    for cluster_idx in range(n_clusters):
        ax = axes[cluster_idx]
        
        for fold_idx, fold_data in enumerate(all_folds_data):
            # Extract this cluster's positions across all simulations for this fold
            cluster_positions = fold_data[:, cluster_idx, :]
            
            # Plot points for each simulation
            ax.scatter(
                cluster_positions[:, feature_indices[0]], 
                cluster_positions[:, feature_indices[1]],
                color=colors[fold_idx], 
                alpha=0.5,
                label=f'Fold {fold_idx+1}' if cluster_idx == 0 else None
            )
            
            # Plot fold mean position with larger marker
            fold_mean = np.mean(cluster_positions, axis=0)
            ax.scatter(
                fold_mean[feature_indices[0]], 
                fold_mean[feature_indices[1]],
                color=colors[fold_idx], 
                s=100, 
                edgecolor='black'
            )
        
        ax.set_title(f'Cluster {cluster_idx+1} Positions')
        ax.set_xlabel(f'Feature {feature_indices[0]}')
        ax.set_ylabel(f'Feature {feature_indices[1]}')
        
        if cluster_idx == 0:
            ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def visualize_stability_bars(fold_stability_results, cross_fold_stability):
    """Visualize stability as bar charts"""
    n_folds = len(fold_stability_results)
    
    means = [result['mean_stability'] for result in fold_stability_results] + [cross_fold_stability['mean_stability']]
    stds = [result['std_stability'] for result in fold_stability_results] + [cross_fold_stability['std_stability']]
    labels = [f'Fold {i+1}' for i in range(n_folds)] + ['Cross-Fold']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    bar_width = 0.7
    
    bars = ax.bar(x, means, bar_width, yerr=stds, capsize=5, 
                 color=plt.cm.viridis(np.linspace(0, 0.8, n_folds+1)))
    
    ax.set_title('Clustering Stability Comparison')
    ax.set_ylabel('Average Distance (lower is more stable)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                f'{means[i]:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    return fig
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Clustering method selection")
    parser.add_argument(
            '--method', 
            type=str, 
            choices=['kmeans', 'em'],  # Specifies allowed choices for the method
            default='kmeans',  # Default method if none is provided
            help="Choose the clustering method (e.g., 'kmeans', 'em'). Default is 'kmeans'."
        )
    args = parser.parse_args()
    return args

def main():
    # Configuration
    args = parse_args()
    method = args.method
    base_path = 'cv_splits'  # Change this to the base directory containing cv_splits
    n_folds = 5
    if os.path.exists(base_path):
        print(f"The folder '{base_path}' exists.")
    elif os.path.exists(method + +'_'+ base_path):
        base_path = method + '_' + base_path
        print(f"The folder '{base_path}' does not exist.")
    else:
        print(f"The cv_splits folder does not exist.")
    # Store data from all folds
    all_folds_data = []
    fold_stability_results = []
    fold_stability_matrices = []
    
    # Process each fold
    for fold_num in range(1, n_folds + 1):
        print(f"Processing fold {fold_num}...")
        
        # 1. Load the data
        fold_df = load_fold_data(base_path, fold_num, method).T
        k = 4
        fold_df['simulation_id'] = np.repeat(np.arange(0, len(fold_df) // k ), k)[:len(fold_df)]
        fold_df['cluster_id'] = np.tile([0, 1, 2, 3], len(fold_df) // 4 + 1)[:len(fold_df)]
        # Extract the simulations data - reshape to get simulations, clusters, and features
        # Assuming data has columns for simulation_id, cluster_id, and feature columns
        simulation_ids = fold_df['simulation_id'].unique()
        n_simulations = len(simulation_ids)
        n_clusters = len(fold_df['cluster_id'].unique())

        
        # Extract features (assuming all columns except simulation_id and cluster_id are features)
        feature_cols = [col for col in fold_df.columns if col not in ['simulation_id', 'cluster_id']]
        n_features = len(feature_cols)
        
        # Prepare array to hold all simulations data: shape (n_simulations, n_clusters, n_features)
        simulations_data = np.zeros((n_simulations, n_clusters, n_features))
        
        # Fill the array with data
        for sim_idx, sim_id in enumerate(simulation_ids):
            sim_data = fold_df[fold_df['simulation_id'] == sim_id].sort_values('cluster_id')
            simulations_data[sim_idx] = sim_data[feature_cols].values
        reference_simulation = simulations_data[0]
        # 2. Reassign cluster labels for consistency
        reassigned_data = reassign_clusters(reference_simulation, simulations_data ,k)
        
        # 3. Calculate stability within this fold
        fold_stability = calculate_stability_within_fold(reassigned_data)
        fold_stability_results.append(fold_stability)
        
        # Calculate pairwise stability matrix for visualization
        stability_matrix = calculate_pairwise_stability(reassigned_data)
        fold_stability_matrices.append(stability_matrix)
        
        print(f"Fold {fold_num} stability: {fold_stability['mean_stability']:.4f} ± {fold_stability['std_stability']:.4f}")
        
        # Store for cross-fold analysis
        all_folds_data.append(reassigned_data)
    
    # 4. Calculate stability across all folds
    cross_fold_stability = calculate_stability_across_folds(all_folds_data)
    print("\nCross-fold stability:")
    print(f"Mean: {cross_fold_stability['mean_stability']:.4f}")
    print(f"Std: {cross_fold_stability['std_stability']:.4f}")
    
    # 5. Create visualizations
    # Set the style for the plots
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12})
    
    # Create all visualizations
    fig1 = visualize_within_fold_stability(fold_stability_matrices, fold_stability_results)
    fig2 = visualize_cross_fold_stability(cross_fold_stability['cross_fold_matrix'], cross_fold_stability)
    fig3 = visualize_cluster_positions(all_folds_data)
    fig4 = visualize_stability_bars(fold_stability_results, cross_fold_stability)
    
    # Save the visualizations
    fig1.savefig(os.path.join(base_path,'within_fold_stability.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(base_path,'cross_fold_stability.png'), dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(base_path,'cluster_positions.png'), dpi=300, bbox_inches='tight')
    fig4.savefig(os.path.join(base_path,'stability_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], width_ratios=[1, 1])
    
    # Within-fold stability heatmaps in a single row
    inner_gs = gs[0, :].subgridspec(1, n_folds)
    for i in range(n_folds):
        ax = fig.add_subplot(inner_gs[i])
        sns.heatmap(fold_stability_matrices[i], annot=False, cmap='viridis_r', 
                   ax=ax, fmt='.3f', cbar=(i == n_folds-1),
                   cbar_kws={'label': 'Distance'} if i == n_folds-1 else {})
        ax.set_title(f'Fold {i+1} Stability')
        ax.set_ylabel('Simulation')
    
    # Cross-fold stability heatmap
    ax1 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cross_fold_stability['cross_fold_matrix'], annot=False, cmap='viridis_r', 
               ax=ax1, fmt='.3f', cbar_kws={'label': 'Distance'})
    ax1.set_title('Cross-Fold Stability')
    ax1.set_ylabel('Fold')
    
    # Stability comparison bar chart
    ax2 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(fold_stability_results) + 1)
    means = [r['mean_stability'] for r in fold_stability_results] + [cross_fold_stability['mean_stability']]
    stds = [r['std_stability'] for r in fold_stability_results] + [cross_fold_stability['std_stability']]
    bars = ax2.bar(x, means, yerr=stds, capsize=5, 
                  color=plt.cm.viridis(np.linspace(0, 0.8, n_folds+1)))
    ax2.set_title('Stability Comparison')
    ax2.set_ylabel('Average Distance (lower is more stable)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)] + ['Cross-Fold'])
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                f'{means[i]:.4f}', ha='center', va='bottom')
    
    plt.suptitle('Clustering Stability Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(base_path,'comprehensive_stability_analysis.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Return results as dictionary if needed
    results_summary = {
        'fold_stability': fold_stability_results,
        'cross_fold_stability': cross_fold_stability
    }
    
    return results_summary

if __name__ == "__main__":
    main()