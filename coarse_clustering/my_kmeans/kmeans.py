import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA


# K-means functionality
def distance(point1, point2):
    """Calculate squared Euclidean distance between two points."""
    return np.sum((point1 - point2) ** 2)

def assign_clusters(data, means):
    """Assign data points to nearest cluster."""
    clusters = [[] for _ in range(means.shape[1])]
    assignments = []
    for i in range(data.shape[1]):
        point = data[:, i]
        distances = [distance(point, means[:, j]) for j in range(means.shape[1])]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
        assignments.append(cluster_index)
    return clusters, np.array(assignments)

def update_means(clusters, previous_means):
    """Update cluster means based on assigned points."""
    new_means = previous_means.copy()
    for i, cluster in enumerate(clusters):
        if cluster:
            new_means[:, i] = np.mean(np.vstack(cluster).T, axis=1)
    return new_means

def calculate_wcss(data, means, assignments):
    """Calculate Within-Cluster Sum of Squares (WCSS)."""
    wcss = 0
    for i in range(data.shape[1]):
        point = data[:, i]
        wcss += distance(point, means[:, assignments[i]])
    return wcss

def k_means(data, k, max_iterations=100, random_state=42, verbose=False, visualize=False):
    """
    Perform k-means clustering on data.
    
    Parameters:
    -----------
    data : array-like of shape (n_features, n_samples)
        The input data.
    k : int
        The number of clusters.
    max_iterations : int, default=100
        Maximum number of iterations.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress information.
    visualize : bool, default=False
        Whether to generate visualization during clustering.
        
    Returns:
    --------
    means : array-like of shape (n_features, k)
        The final cluster means.
    wcss : float
        The final within-cluster sum of squares.
    assignments : array-like of shape (n_samples,)
        The cluster assignments for each data point.
    """
    np.random.seed(random_state)

    # Convert to appropriate format
    if isinstance(data, pd.DataFrame):
        data = data.values


    # Initialize centers by selecting random data points
    initial_indices = np.random.choice(data.shape[1], k, replace=False)
    means = data[:, initial_indices].copy()
    
    # Initialize for visualization
    if visualize:
        wcss_history = []
        # Only visualize first 2 dimensions or use PCA to reduce dimensions
        if data.shape[0] > 2:
            pca = PCA(n_components=2)
            vis_data = pca.fit_transform(data.T).T
            vis_means = pca.transform(means.T).T
        else:
            vis_data = data
            vis_means = means
        plt.figure(figsize=(12, 8))
    
    for iteration in range(max_iterations):
        if verbose and iteration % 10 == 0:
            print(f"K-means iteration {iteration}/{max_iterations}")
            
        # Assign points to clusters
        clusters, assignments = assign_clusters(data, means)
        
        # Update means
        new_means = update_means(clusters, means)
        
        # Calculate WCSS for visualization
        if visualize:
            current_wcss = calculate_wcss(data, means, assignments)
            wcss_history.append(current_wcss)
        
        # Check for convergence
        if np.allclose(means, new_means):
            if verbose:
                print(f"K-means converged after {iteration+1} iterations")
            break
        
        means = new_means
        
        # Visualize current state
        if visualize and (iteration % 5 == 0 or iteration == max_iterations - 1):
            plt.clf()
            colors = plt.cm.tab10(np.linspace(0, 1, k))
            
            # Plot data points
            for i, cluster_assign in enumerate(np.unique(assignments)):
                mask = assignments == cluster_assign
                indices = np.where(mask)[0]
                plt.scatter(vis_data[0, indices], vis_data[1, indices], 
                            color=colors[i], alpha=0.5, label=f'Cluster {i+1}')
            
            # Plot centroids
            if data.shape[0] > 2:
                centroid_vis = pca.transform(means.T).T
            else:
                centroid_vis = means
                
            plt.scatter(centroid_vis[0], centroid_vis[1], 
                        marker='X', s=200, color='black', label='Centroids')
            
            plt.title(f'K-means Clustering - Iteration {iteration+1}')
            plt.legend()
            plt.tight_layout()
            plt.pause(0.1)  # Brief pause to allow visualization to update
    
    # Calculate final WCSS
    wcss = calculate_wcss(data, means, assignments)
    
    # Final visualization with WCSS plot
    if visualize:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        
        for i, cluster_assign in enumerate(np.unique(assignments)):
            mask = assignments == cluster_assign
            indices = np.where(mask)[0]
            plt.scatter(vis_data[0, indices], vis_data[1, indices], 
                        color=colors[i], alpha=0.5, label=f'Cluster {i+1}')
        
        # Plot final centroids
        if data.shape[0] > 2:
            centroid_vis = pca.transform(means.T).T
        else:
            centroid_vis = means
            
        plt.scatter(centroid_vis[0], centroid_vis[1], 
                    marker='X', s=200, color='black', label='Centroids')
        
        plt.title('Final K-means Clustering')
        plt.legend()
        
        # Plot WCSS history
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(wcss_history) + 1), wcss_history, marker='o')
        plt.title('Within-Cluster Sum of Squares (WCSS)')
        plt.xlabel('Iteration')
        plt.ylabel('WCSS')
        plt.tight_layout()
        plt.show()
    
    return means, wcss, assignments