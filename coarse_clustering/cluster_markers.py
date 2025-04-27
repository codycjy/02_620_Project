import process
import my_kmeans
import numpy as np
import scanpy as sc
import gc
from tqdm import tqdm
import checker
import pandas as pd
import scipy
import my_em

def cluster_all_data(training_set, n_samples_per_donor_id, n_simulations, k = 7, random_state=42, max_iterations = 1000, methods = 'kmeans'):
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
    result = {}
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
        if methods == 'kmeans':
            means, wcss , _ = my_kmeans.k_means(X,k,max_iterations,random_state)
            if 'wcss' not in result:
                result['wcss'] = []
            result['wcss'].append(wcss)
        elif methods == 'em':
            means, covariances, weights, log_likelihood, responsibilities = my_em.gaussian_mixture_em(X, k, max_iterations=max_iterations * 10, random_state=random_state+sim, verbose=False)
            if 'covariance' not in result:
                result['covariances'] = []
                result['weights'] = []
                result['log_likelihood'] = []
                result['responsibilities'] = []
            result['covariances'] = result['covariances'] + [covariances]
            result['weights'] = result['weights'] + [weights]
            result['log_likelihood'] = result['log_likelihood'] + [log_likelihood]
            result['responsibilities'] = result['responsibilities'] + [responsibilities]
        # Store results - only store the means, not the full data
        else:
            raise ValueError(f"Unknown method: {methods}. Use 'kmeans' or 'em'.")
        print(X.shape, means.shape)
        all_cluster_means = np.concatenate((all_cluster_means, means), axis=1)
        
        # Force garbage collection to free memory after each simulation
        gc.collect()
    
    
    print(f"All {n_simulations} simulations completed. Final mean vectors shape: {all_cluster_means.shape}")
    return all_cluster_means, result



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
    n_clusters = overall_cluster_means.shape[1]
    means_array = overall_cluster_means.T


    
    # Create the dictionary
    cluster_means_dict = {}
    for i in range(n_clusters):
        # Use string keys to match the expected format
        cluster_id = str(i)
        cluster_means_dict[cluster_id] = means_array[i,:]
    
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
            sample_data.obs['cluster'] = np.zeros(n_cells, dtype=str)  # Change to string type for cluster IDs
            
            for batch_start in range(0, n_cells, batch_size):
                batch_end = min(batch_start + batch_size, n_cells)
                batch_X = X[batch_start:batch_end].toarray()
                
                # Calculate distances and assign clusters
                distances = cdist(batch_X, cluster_means_array, metric='euclidean')
                cluster_assignments = np.argmin(distances, axis=1)
                
                # Convert numeric indices to original cluster IDs
                sample_data.obs.iloc[batch_start:batch_end, 
                                    sample_data.obs.columns.get_loc('cluster')] = [
                    cluster_ids[idx] for idx in cluster_assignments
                ]
                
                # Free memory
                del batch_X, distances, cluster_assignments
                gc.collect()
        else:
            # For small or already dense matrices
            distances = cdist(X, cluster_means_array, metric='euclidean')
            cluster_assignments = np.argmin(distances, axis=1)
            sample_data.obs['cluster'] = [cluster_ids[idx] for idx in cluster_assignments]
        
        # Count cells in each cluster for this donor
        cluster_counts = sample_data.obs['cluster'].value_counts().to_dict()
        donor_cluster_counts[donor_id] = cluster_counts
        
        # Compute and store mean expression per cluster
        rows = []
        for cluster in sample_data.obs['cluster'].unique():
            cluster_cells = sample_data[sample_data.obs['cluster'] == cluster]
            
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

def calculate_final_mean_results(adata, patient_cluster_expr_df, output_path=None):
    """
    Calculate the final mean expression results for all genes in the AnnData table
    after labeling the dataset with clusters.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing gene expression data.
    patient_cluster_expr_df : pandas.DataFrame
        DataFrame containing sample IDs, cluster assignments, and mean expression values
        as returned by label_dataset_with_clusters function.
    output_path : str, optional
        Path to save the resulting mean expression DataFrame. If None, only returns the DataFrame.
        
    Returns:
    --------
    final_mean_results : pandas.DataFrame
        DataFrame containing the final mean expression values for all genes across clusters.
    """
    import pandas as pd
    import numpy as np
    import gc
    from tqdm import tqdm
    
    print("Calculating final mean results for all genes...")
    
    # Extract gene names from AnnData if available
    if hasattr(adata, 'var_names') and len(adata.var_names) > 0:
        gene_names = adata.var_names.tolist()
    else:
        # If gene names not available, use generic feature names
        gene_names = [f"feature_{i}" for i in range(adata.X.shape[1])]
    
    # Get unique clusters
    clusters = patient_cluster_expr_df['cluster'].unique()
    
    # Initialize DataFrame for final results
    final_mean_results = pd.DataFrame(index=gene_names)
    
    # Calculate mean expression for each cluster
    for cluster in tqdm(clusters, desc="Processing clusters"):
        # Filter data for current cluster
        cluster_data = patient_cluster_expr_df[patient_cluster_expr_df['cluster'] == cluster]
        
        # Extract mean expression arrays and convert if they're stored as lists
        mean_expr_arrays = []
        for _, row in cluster_data.iterrows():
            mean_expr = row['mean_expression']
            # Convert to numpy array if stored as list
            if isinstance(mean_expr, list):
                mean_expr = np.array(mean_expr)
            mean_expr_arrays.append(mean_expr)
        
        # Stack arrays and calculate overall mean
        if mean_expr_arrays:
            # Ensure all arrays have the same shape
            if all(len(arr) == len(mean_expr_arrays[0]) for arr in mean_expr_arrays):
                stacked_means = np.vstack(mean_expr_arrays)
                cluster_final_mean = np.mean(stacked_means, axis=0)
                
                # Add to results DataFrame
                final_mean_results[f"cluster_{cluster}"] = cluster_final_mean
            else:
                print(f"Warning: Inconsistent array lengths for cluster {cluster}")
        
        # Free memory
        del mean_expr_arrays
        gc.collect()
    
    # Calculate overall mean across all clusters
    final_mean_results['overall_mean'] = final_mean_results.mean(axis=1)
    
    # Save results if output path is provided
    if output_path:
        print(f"Saving final mean results to {output_path}")
        final_mean_results.to_csv(output_path)
        print(f"Saved to '{output_path}'")
    
    return final_mean_results


# Example usage:
# 1. First run the clustering and labeling
# adata = sc.read_h5ad("your_file.h5ad")
# all_cluster_means = cluster_all_data(adata, n_samples_per_donor_id=100, n_simulations=10)
# cluster_means_dict = create_cluster_means_dict(all_cluster_means)
# patient_cluster_expr_df, cell_count_df = label_dataset_with_clusters(adata, cluster_means_dict)

# 2. Then calculate the final mean results
# final_mean_results = calculate_final_mean_results(adata, patient_cluster_expr_df, output_path="final_mean_results.csv")
def label_dataset_with_clusters_marker_distance_all_genes_mean(
    dataset, cluster_means_dict, marker_genes, output_path=None):
    from scipy.spatial.distance import cdist

    print("Starting dataset labeling with marker-gene-based cluster assignment...")
    
    cluster_ids = list(cluster_means_dict.keys())
    marker_cluster_means_array = np.array([cluster_means_dict[cid] for cid in cluster_ids])
    
    patient_cluster_expr_df = pd.DataFrame()
    sample_status = {}
    sample_score = {}
    sample_apoe = {}
    if isinstance(dataset, str):
        print(f"Loading dataset from {dataset}")
        adata = sc.read_h5ad(dataset)
    else:
        adata = dataset

    if 'Donor ID' not in adata.obs.columns:
        print("Warning: 'Donor ID' column not found. Defaulting to filename or 'unknown_donor'.")
        if hasattr(adata, 'filename') and adata.filename:
            donor_id = adata.filename.split('/')[-1].split('.')[0]
        else:
            donor_id = "unknown_donor"
        adata.obs['Donor ID'] = donor_id

    donor_ids = adata.obs['Donor ID'].unique()
    print(f"Found {len(donor_ids)} unique Donor IDs")

    # Ensure marker genes are in dataset
    valid_markers = [gene for gene in marker_genes if gene in adata.var_names]
    marker_indices = [adata.var_names.get_loc(g) for g in valid_markers]

    donor_cluster_counts = {donor_id: {} for donor_id in donor_ids}

    for donor_id in tqdm(donor_ids, desc="Processing samples"):
        sample_mask = adata.obs['Donor ID'] == donor_id
        sample_data = adata[sample_mask].copy()

        if 'Cognitive Status' in sample_data.obs.columns:
            sample_status[donor_id] = sample_data.obs['Cognitive Status'].iloc[0]

        if 'Last CASI Score' in sample_data.obs.columns:
            sample_score[donor_id] = sample_data.obs['Last CASI Score'].iloc[0]

        if 'APOE Genotype' in sample_data.obs.columns:
             # Deal with APOE status, count the number of APOE for each donor
            sample_apoe[donor_id] = sample_data.obs['APOE Genotype'].iloc[0].count('4')

        X_marker = sample_data.X[:, marker_indices]
        if hasattr(X_marker, "toarray"):
            X_marker = X_marker.toarray()
        distances = cdist(X_marker, marker_cluster_means_array, metric='euclidean')
        cluster_assignments = np.argmin(distances, axis=1)
        sample_data.obs['cluster'] = [cluster_ids[i] for i in cluster_assignments]

        cluster_counts = sample_data.obs['cluster'].value_counts().to_dict()
        donor_cluster_counts[donor_id] = cluster_counts

        # Compute means over all genes
        rows = []
        for cluster in sample_data.obs['cluster'].unique():
            cluster_cells = sample_data[sample_data.obs['cluster'] == cluster]
            X_all = cluster_cells.X
            if hasattr(X_all, "toarray"):
                mean_expr = np.array(X_all.mean(axis=0)).flatten()
            else:
                mean_expr = np.mean(X_all, axis=0)
                if mean_expr.ndim > 1:
                    mean_expr = mean_expr.flatten()
            row_dict = {
                'Donor ID': donor_id,
                'cluster': cluster,
                'mean_expression': mean_expr.tolist(),
                'cell_count': len(cluster_cells),
                'percent_cells': (len(cluster_cells) / len(sample_data)) * 100
            }
            rows.append(row_dict)

        sample_df = pd.DataFrame(rows)
        patient_cluster_expr_df = pd.concat([patient_cluster_expr_df, sample_df], ignore_index=True)

        del sample_data, rows, sample_df
        gc.collect()

    if sample_status:
        patient_cluster_expr_df['Cognitive Status'] = patient_cluster_expr_df['Donor ID'].map(sample_status)

    if sample_score:
        patient_cluster_expr_df['Last CASI Score'] = patient_cluster_expr_df['Donor ID'].map(sample_score)
    
    if sample_apoe:
        patient_cluster_expr_df['APOE Values'] = patient_cluster_expr_df['Donor ID'].map(sample_apoe)

    cell_count_rows = []
    for donor_id in donor_ids:
        for cluster_id in cluster_ids:
            count = donor_cluster_counts[donor_id].get(cluster_id, 0)
            cell_count_rows.append({'Donor ID': donor_id, 'cluster': cluster_id, 'cell_count': count})

    cell_count_df = pd.DataFrame(cell_count_rows)

    if output_path:
        cell_count_output = output_path.replace('.csv', '_cell_counts.csv')
        print(f"Saving cell count matrix to {cell_count_output}")
        cell_count_df.to_csv(cell_count_output, index=False)

        pivot_df = cell_count_df.pivot(index='Donor ID', columns='cluster', values='cell_count').fillna(0)
        pivot_output = output_path.replace('.csv', '_cell_counts_pivot.csv')
        pivot_df.to_csv(pivot_output)
        print(f"Saved pivot table to '{pivot_output}'")

        print(f"Saving main results to {output_path}")
        patient_cluster_expr_df.to_csv(output_path, index=False)

    return patient_cluster_expr_df, cell_count_df

