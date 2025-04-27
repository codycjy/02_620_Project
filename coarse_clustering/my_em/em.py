import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# EM Algorithm for Gaussian Mixture Models
def initialize_means(data, n_components):
    """Initialize means by selecting random data points."""
    n_features, n_samples = data.shape
    indices = np.random.choice(n_samples, size=n_components, replace=False)
    return data[:, indices]

def initialize_covariances(data, n_components):
    """Initialize covariances as identity matrices scaled by data variance."""
    n_features, n_samples = data.shape
    variance = np.var(data, axis=1).mean()
    covariances = np.array([np.eye(n_features) * variance for _ in range(n_components)])
    return covariances

def expectation_step(data, means, covariances, weights):
    """E-step: Calculate responsibilities."""
    n_components = len(weights)
    n_features, n_samples = data.shape
    
    # Calculate likelihoods for each data point under each component
    likelihoods = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        try:
            mvn = multivariate_normal(mean=means[:, k], cov=covariances[k])
            likelihoods[:, k] = mvn.pdf(data.T)
        except np.linalg.LinAlgError:
            # Handle numerical instability
            covariances[k] += np.eye(n_features) * 1e-6
            mvn = multivariate_normal(mean=means[:, k], cov=covariances[k])
            likelihoods[:, k] = mvn.pdf(data.T)
    
    # Calculate weighted likelihoods
    weighted_likelihoods = likelihoods * weights
    
    # Calculate total likelihood and responsibilities
    total_likelihood = weighted_likelihoods.sum(axis=1, keepdims=True)
    responsibilities = weighted_likelihoods / total_likelihood
    
    # Calculate log-likelihood
    log_likelihood = np.sum(np.log(total_likelihood))
    
    return responsibilities, log_likelihood

def maximization_step(data, responsibilities):
    """M-step: Update parameters based on responsibilities."""
    n_features, n_samples = data.shape
    n_components = responsibilities.shape[1]
    
    # Calculate effective number of points in each cluster
    Nk = responsibilities.sum(axis=0)
    
    # Update weights
    weights = Nk / n_samples
    
    # Update means
    means = np.zeros((n_features, n_components))
    for k in range(n_components):
        for i in range(n_samples):
            means[:, k] += responsibilities[i, k] * data[:, i]
        means[:, k] /= Nk[k]
    
    # Update covariances
    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        for i in range(n_samples):
            diff = (data[:, i] - means[:, k]).reshape(-1, 1)
            covariances[k] += responsibilities[i, k] * (diff @ diff.T)
        covariances[k] /= Nk[k]
        # Add small regularization to prevent singularity
        covariances[k] += np.eye(n_features) * 1e-6
    
    return means, covariances, weights

def calculate_bic(data, means, covariances, weights, log_likelihood):
    """Calculate Bayesian Information Criterion (BIC) for model selection."""
    n_features, n_samples = data.shape
    n_components = len(weights)
    
    # Number of free parameters
    n_params = n_components - 1  # weights (sum to 1, so one less free parameter)
    n_params += n_components * n_features  # means
    n_params += n_components * n_features * (n_features + 1) // 2  # covariances (symmetric)
    
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic

def gaussian_mixture_em(data, n_components, max_iterations=100, tol=1e-6, random_state=42, 
                       verbose=False, visualize=False):
    """
    Performs the Expectation-Maximization algorithm for a Gaussian Mixture Model.
    
    Parameters:
    -----------
    data : array-like of shape (n_features, n_samples) or pandas DataFrame
        The input data.
    n_components : int
        The number of mixture components.
    max_iterations : int, default=100
        Maximum number of iterations to perform.
    tol : float, default=1e-6
        Tolerance for convergence.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress information.
    visualize : bool, default=False
        Whether to generate visualization during EM iterations.
        
    Returns:
    --------
    means : array-like of shape (n_features, n_components)
        The means of each mixture component.
    covariances : array-like of shape (n_components, n_features, n_features)
        The covariance matrices of each mixture component.
    weights : array-like of shape (n_components,)
        The weights of each mixture component.
    log_likelihood : float
        The log-likelihood of the data.
    responsibilities : array-like of shape (n_samples, n_components)
        The posterior probabilities of each data point belonging to each component.
    """
    np.random.seed(random_state)
    
    # Convert DataFrame to NumPy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    
    n_features, n_samples = data.shape
    
    # Initialize parameters
    means = initialize_means(data, n_components)
    covariances = initialize_covariances(data, n_components)
    weights = np.ones(n_components) / n_components  # Equal weights initially
    
    log_likelihood = -np.inf
    ll_history = []
    converged = False
    
    # Setup for visualization
    if visualize:
        # If more than 2D, use PCA for visualization
        if n_features > 2:
            pca = PCA(n_components=2)
            vis_data = pca.fit_transform(data.T).T
            # Project means to 2D for visualization
            vis_means = pca.transform(means.T).T
        else:
            vis_data = data
            vis_means = means
            
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    
    for iteration in range(max_iterations):
        if verbose and iteration % 10 == 0:
            print(f"EM iteration {iteration}/{max_iterations}, Log-likelihood: {log_likelihood:.4f}")
            
        # E-step: Calculate responsibilities
        responsibilities, current_log_likelihood = expectation_step(data, means, covariances, weights)
        ll_history.append(current_log_likelihood)
        
        # Check for convergence
        if np.abs(current_log_likelihood - log_likelihood) < tol:
            if verbose:
                print(f"EM converged after {iteration+1} iterations")
            converged = True
            break
        
        log_likelihood = current_log_likelihood
        
        # M-step: Update parameters
        means, covariances, weights = maximization_step(data, responsibilities)
        
        # Visualize current state
        if visualize and (iteration % 5 == 0 or iteration == max_iterations - 1):
            plt.clf()
            
            # Get cluster assignments for coloring
            assignments = np.argmax(responsibilities, axis=1)
            
            # Plot data points colored by most likely component
            for i in range(n_components):
                mask = assignments == i
                indices = np.where(mask)[0]
                plt.scatter(vis_data[0, indices], vis_data[1, indices], 
                            color=colors[i], alpha=0.5, label=f'Component {i+1}')
            
            # Plot component means
            if n_features > 2:
                component_vis = pca.transform(means.T).T
            else:
                component_vis = means
                
            plt.scatter(component_vis[0], component_vis[1], 
                        marker='X', s=200, color='black', label='Means')
            
            # Draw covariance ellipses
            for i in range(n_components):
                if n_features > 2:
                    # Project covariance to 2D
                    cov_2d = pca.components_ @ covariances[i] @ pca.components_.T
                    mean_2d = component_vis[:, i]
                else:
                    cov_2d = covariances[i]
                    mean_2d = means[:, i]
                
                # Calculate eigenvalues and eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
                
                # Angle of ellipse
                angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
                
                # Width and height of ellipse (2 standard deviations)
                width, height = 2 * np.sqrt(eigenvals) * 2
                
                # Create ellipse
                ellipse = Ellipse((mean_2d[0], mean_2d[1]), width, height, 
                                  angle=np.degrees(angle), edgecolor=colors[i], 
                                  facecolor='none', linewidth=2)
                plt.gca().add_patch(ellipse)
            
            plt.title(f'EM for GMM - Iteration {iteration+1}')
            plt.legend()
            plt.tight_layout()
            plt.pause(0.1)  # Brief pause to allow visualization to update
    
    # Final visualization
    if visualize:
        plt.figure(figsize=(12, 5))
        
        # Plot final clustering
        plt.subplot(1, 2, 1)
        assignments = np.argmax(responsibilities, axis=1)
        
        for i in range(n_components):
            mask = assignments == i
            indices = np.where(mask)[0]
            plt.scatter(vis_data[0, indices], vis_data[1, indices], 
                        color=colors[i], alpha=0.5, label=f'Component {i+1}')
        
        # Plot component means and covariance ellipses
        if n_features > 2:
            component_vis = pca.transform(means.T).T
        else:
            component_vis = means
            
        plt.scatter(component_vis[0], component_vis[1], 
                    marker='X', s=200, color='black', label='Means')
        
        for i in range(n_components):
            if n_features > 2:
                cov_2d = pca.components_ @ covariances[i] @ pca.components_.T
                mean_2d = component_vis[:, i]
            else:
                cov_2d = covariances[i]
                mean_2d = means[:, i]
            
            eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
            angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
            width, height = 2 * np.sqrt(eigenvals) * 2
            
            ellipse = Ellipse((mean_2d[0], mean_2d[1]), width, height, 
                              angle=np.degrees(angle), edgecolor=colors[i], 
                              facecolor='none', linewidth=2)
            plt.gca().add_patch(ellipse)
        
        plt.title('Final GMM Clustering')
        plt.legend()
        
        # Plot log-likelihood history
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(ll_history) + 1), ll_history, marker='o')
        plt.title('Log-Likelihood Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.tight_layout()
        plt.show()
        
        # Plot a new figure to show the cluster responsibilities as pie charts
        plt.figure(figsize=(10, 6))
        
        # Sample a subset of points to display as pie charts
        n_samples_to_show = min(20, n_samples)
        sample_indices = np.random.choice(n_samples, n_samples_to_show, replace=False)
        
        # Create grid layout for pie charts
        grid_size = int(np.ceil(np.sqrt(n_samples_to_show)))
        for idx, sample_idx in enumerate(sample_indices):
            plt.subplot(grid_size, grid_size, idx + 1)
            plt.pie(responsibilities[sample_idx], colors=colors, 
                   wedgeprops=dict(width=0.5, edgecolor='w'))
            plt.axis('equal')
            plt.title(f'Point {sample_idx}', fontsize=8)
        
        plt.suptitle('Sample Point Cluster Membership Probabilities')
        plt.tight_layout()
        plt.show()
    
    # Calculate BIC for model selection
    bic = calculate_bic(data, means, covariances, weights, log_likelihood)
    if verbose:
        print(f"Final Log-Likelihood: {log_likelihood:.4f}, BIC: {bic:.4f}")
    
    return means, covariances, weights, log_likelihood, responsibilities


def predict_em(data, means, covariances, weights):
    """Predict cluster assignments for new data using GMM."""
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Ensure data is in the right shape (n_features, n_samples)
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    responsibilities, _ = expectation_step(data, means, covariances, weights)
    return np.argmax(responsibilities, axis=1)