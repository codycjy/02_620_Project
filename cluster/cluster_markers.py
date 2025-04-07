import process
import my_kmeans
import numpy as np
import scanpy as sc
import gc
from tqdm import tqdm
import checker
import pandas as pd
import scipy

def cluster_all_data(training_set, n_samples_per_donor_id, n_simulations, k = 7, random_state=42, max_iterations = 1000):
    """
    Perform multiple simulations of sampling cells, clustering, and calculating cluster means.
    
    Parameters:
    -----------
    training_set : AnnData
        The training dataset containing cells from different donors
    n_samples_per_donor_id : int
        Number of cells to sample from each donor
    n_simulations : int
        Number of simulations to run
    k: int, default = 7
        Number of clusters
    random_state : int, default=42
        Random seed for reproducibility
    max_iterations : int, default = 1000
        Number of iterations for kmeans
    
    
    Returns:
    --------
    all_cluster_means : list
        List of cluster means for each simulation
        Each element is a dictionary with cluster ID as key and mean expression as value
    simulation_results : list
        List of dictionaries containing detailed results for each simulation
    """
    print(f"Starting {n_simulations} simulations with {n_samples_per_donor_id} samples per donor_id")
    
    # Get unique donor ids
    donor_ids = training_set.obs['Donor ID'].unique().tolist()
    print(f"Found {len(donor_ids)} unique donor ID")
    
    # Store all cluster means from all simulations
    all_cluster_means = np.empty((training_set.X.shape[1],0))
    for sim in tqdm(range(n_simulations),desc = "Processing"):
        np.random.seed(random_state + sim)  # Different seed for each simulation
        
        # Sample cells from each donor
        sampled_indices = []
        for donor_id in donor_ids:
            # Get indices of cells with this donor
            sample_indices = np.where(training_set.obs['Donor ID'] == donor_id)[0]
            
            # Sample from this group
            if len(sample_indices) > n_samples_per_donor_id:
                # If enough cells, sample without replacement
                selected_indices = np.random.choice(sample_indices, n_samples_per_donor_id, replace=False)
            else:
                # If not enough cells, sample with replacement to reach target count
                selected_indices = np.random.choice(sample_indices, n_samples_per_donor_id, replace=True)
                
            sampled_indices.extend(selected_indices)
        
        # Create a view (not a copy) of the AnnData object with the sampled cells
        adata_sample = training_set[sampled_indices]
        
        # PCA - necessary for clustering
        X = adata_sample.X
        X = X.toarray().T if scipy.sparse.issparse(X) else X.T

        # PCA and KMeans

        means, _ , _ = my_kmeans.k_means(X,k,max_iterations,random_state)
        # Store results - only store the means, not the full data
        all_cluster_means = np.concatenate((all_cluster_means, means), axis=1)
        
        # Force garbage collection to free memory after each simulation
        gc.collect()
    
    print(f"All {n_simulations} simulations completed. Final mean vectors shape: {all_cluster_means.shape}")
    return all_cluster_means



def create_cluster_means_dict(overall_cluster_means, feature_names=None):
    """
    Convert a cluster means matrix to a dictionary format required by label_dataset_with_clusters.
    
    Parameters:
    -----------
    overall_cluster_means : numpy.ndarray
        2D array of shape (n_features, n_clusters) or (n_clusters, n_features)
        Contains the mean expression values for each cluster
    feature_names : list, optional
        List of feature names. If None, feature indices will be used.
        
    Returns:
    --------
    cluster_means_dict : dict
        Dictionary with cluster IDs as keys and mean expression vectors as values
    """
    # Determine the shape and orientation of the matrix
    means_array = overall_cluster_means.T
    n_clusters = overall_cluster_means.shape[1]

    
    # Create the dictionary
    cluster_means_dict = {}
    for i in range(n_clusters):
        # Use string keys to match the expected format
        cluster_id = str(i)
        cluster_means_dict[cluster_id] = means_array[i]
    
    return cluster_means_dict

def label_dataset_with_clusters(dataset, cluster_means_dict, output_path=None):
    from scipy.spatial.distance import cdist
    """
    Label a dataset based on similarity to pre-calculated cluster means.
    
    Parameters:
    -----------
    dataset : AnnData or str
        The AnnData object to label or path to h5ad file.
    cluster_means_dict : dict
        Dictionary with cluster IDs as keys and mean expression vectors as values.
    output_path : str, optional
        Path to save the resulting DataFrame. If None, only returns the DataFrame.
        
    Returns:
    --------
    patient_cluster_expr_df : pandas.DataFrame
        DataFrame containing sample IDs, cluster assignments, and mean expression values.
    """
    print("Starting dataset labeling based on cluster similarity...")
    
    # Convert cluster means dictionary to array for faster distance calculation
    cluster_ids = list(cluster_means_dict.keys())
    cluster_means_array = np.array([cluster_means_dict[cluster_id] for cluster_id in cluster_ids])
    
    # Initialize DataFrame and dictionary for cognitive status
    patient_cluster_expr_df = pd.DataFrame()
    sample_status = {}
    
    # Load the dataset if a path is provided
    if isinstance(dataset, str):
        print(f"Loading dataset from {dataset}")
        adata = sc.read_h5ad(dataset)
    else:
        adata = dataset
        
    # Get unique sample IDs
    if 'Donor ID' not in adata.obs.columns:
        print("Warning: 'Donor ID' column not found. Using first available sample ID for all cells.")
        # Extract a sample ID if available in the file name or metadata
        if hasattr(adata, 'filename') and adata.filename is not None:
            donor_id = adata.filename.split('/')[-1].split('.')[0]
        else:
            donor_id = "unknown_donor"
        adata.obs['Donor ID'] = donor_id
    
    donor_ids = adata.obs['Donor ID'].unique()
    print(f"Found {len(donor_ids)} unique Donor IDs")
    
    # Dictionary to track cell counts per donor and cluster
    donor_cluster_counts = {donor_id: {} for donor_id in donor_ids}
    
    # Process each sample separately to save memory
    for donor_id in tqdm(donor_ids, desc="Processing samples"):
        # Subset to the current sample
        sample_mask = adata.obs['Donor ID'] == donor_id
        sample_data = adata[sample_mask].copy()  # Explicitly create a copy to avoid ImplicitModificationWarning
        
        # Store cognitive status if available
        if 'Cognitive Status' in sample_data.obs.columns:
            sample_status[donor_id] = sample_data.obs['Cognitive Status'].iloc[0]
        
        # Convert sparse matrix to dense for distance calculation if needed
        X = sample_data.X
        if hasattr(X, "toarray"):
            # For larger matrices, process in batches to avoid memory issues
            batch_size = 1000  # Adjust based on available memory
            n_cells = X.shape[0]
            sample_data.obs['leiden'] = np.zeros(n_cells, dtype=str)  # Change to string type for cluster IDs
            
            for batch_start in range(0, n_cells, batch_size):
                batch_end = min(batch_start + batch_size, n_cells)
                batch_X = X[batch_start:batch_end].toarray()
                
                # Calculate distances and assign clusters
                distances = cdist(batch_X, cluster_means_array, metric='euclidean')
                cluster_assignments = np.argmin(distances, axis=1)
                
                # Convert numeric indices to original cluster IDs
                sample_data.obs.iloc[batch_start:batch_end, 
                                    sample_data.obs.columns.get_loc('leiden')] = [
                    cluster_ids[idx] for idx in cluster_assignments
                ]
                
                # Free memory
                del batch_X, distances, cluster_assignments
                gc.collect()
        else:
            # For small or already dense matrices
            distances = cdist(X, cluster_means_array, metric='euclidean')
            cluster_assignments = np.argmin(distances, axis=1)
            sample_data.obs['leiden'] = [cluster_ids[idx] for idx in cluster_assignments]
        
        # Count cells in each cluster for this donor
        cluster_counts = sample_data.obs['leiden'].value_counts().to_dict()
        donor_cluster_counts[donor_id] = cluster_counts
        
        # Compute and store mean expression per cluster
        rows = []
        for cluster in sample_data.obs['leiden'].unique():
            cluster_cells = sample_data[sample_data.obs['leiden'] == cluster]
            
            # Efficiently compute mean expression
            if hasattr(cluster_cells.X, "toarray"):
                mean_expr = np.array(cluster_cells.X.mean(axis=0)).flatten()
            else:
                mean_expr = np.mean(cluster_cells.X, axis=0)
                if mean_expr.ndim > 1:
                    mean_expr = mean_expr.flatten()
            
            # Create a dictionary for the current sample and cluster
            row_dict = {'Donor ID': donor_id, 'cluster': cluster}
            
            # Store the mean expression vector as a column
            row_dict['mean_expression'] = mean_expr.tolist()
            
            # Count cells in this cluster for this donor
            row_dict['cell_count'] = len(cluster_cells)
            
            # Add percent of cells in this cluster for this donor
            row_dict['percent_cells'] = (len(cluster_cells) / len(sample_data)) * 100
            
            rows.append(row_dict)
        
        # Append the results for this sample to the main DataFrame
        sample_df = pd.DataFrame(rows)
        patient_cluster_expr_df = pd.concat([patient_cluster_expr_df, sample_df], ignore_index=True)
        
        # Clear memory
        del sample_data, rows, sample_df
        gc.collect()
    
    # Add cognitive status to the final DataFrame if available
    if sample_status:
        patient_cluster_expr_df['Cognitive Status'] = patient_cluster_expr_df['Donor ID'].map(sample_status)
    
    # Create a separate DataFrame with complete cell count matrix
    # Fill in zeros for missing donor-cluster combinations
    cell_count_rows = []
    for donor_id in donor_ids:
        for cluster_id in cluster_ids:
            count = donor_cluster_counts[donor_id].get(cluster_id, 0)
            cell_count_rows.append({
                'Donor ID': donor_id,
                'cluster': cluster_id,
                'cell_count': count
            })
    
    cell_count_df = pd.DataFrame(cell_count_rows)
    
    # Save the cell count matrix if output path is provided
    if output_path:
        cell_count_output = output_path.replace('.csv', '_cell_counts.csv')
        print(f"Saving cell count matrix to {cell_count_output}")
        cell_count_df.to_csv(cell_count_output, index=False)
        
        # Create a pivot table for easier visualization
        pivot_df = cell_count_df.pivot(index='Donor ID', columns='cluster', values='cell_count').fillna(0)
        pivot_output = output_path.replace('.csv', '_cell_counts_pivot.csv')
        pivot_df.to_csv(pivot_output)
        print(f"Saved pivot table to '{pivot_output}'")
    
    # Save results if output path is provided
    if output_path:
        print(f"Saving results to {output_path}")
        patient_cluster_expr_df.to_csv(output_path, index=False)
        print(f"Saved to '{output_path}'")
    
    return patient_cluster_expr_df, cell_count_df

