import os
import glob
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Apply PMI correction to single-cell data using kernel regression parameters')
    
 
    parser.add_argument('--param_file', type=str, required=True, 
                        help='Path to CSV file containing kernel regression parameters')
    parser.add_argument('--meta_file', type=str, required=True, 
                        help='Path to CSV file containing metadata with Donor ID and PMI information')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing h5ad files to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save corrected files (defaults to data_dir)')
    
    return parser.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# 1. Load kernel regression parameters data
kernel_params = pd.read_csv(args.param_file)
print(f"Loaded kernel regression parameters for {len(kernel_params)} genes")

# 2. Load PMI metadata
metadata = pd.read_csv(args.meta_file)
print(f"Loaded metadata with {len(metadata)} entries")

# Make sure the metadata has the necessary columns
if 'Donor ID' not in metadata.columns or 'PMI' not in metadata.columns:
    print("Error: Metadata file must contain Donor ID and PMI columns")
    exit(1)

# 3. Load all h5ad files from a directory
h5ad_files = glob.glob(os.path.join(args.data_dir, '*.h5ad'))
print(f"Found {len(h5ad_files)} h5ad files in directory")

# Process each file one by one with progress bar
for h5ad_idx, h5ad_file in enumerate(h5ad_files):
    file_name = os.path.basename(h5ad_file)
    print(f"\nProcessing file {h5ad_idx+1}/{len(h5ad_files)}: {file_name}")
    
    # Extract donor ID from filename
    donor_ID = '.'.join(file_name.split('.')[:-1])
    
    # Load the current h5ad file
    print(f"Loading file: {file_name}...")
    adata = sc.read(h5ad_file)
    print(f"Loaded single-cell data with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Look up PMI for this sample in the metadata
    sample_metadata = metadata[metadata['Donor ID'] == donor_ID]
    
    if len(sample_metadata) == 0:
        print(f"Warning: No metadata found for sample {donor_ID}. Skipping this file.")
        continue
    
    # Extract PMI value
    pmi_value = sample_metadata['PMI'].iloc[0]
    print(f"Found PMI value: {pmi_value} for sample {donor_ID}")
    
    # Add PMI to adata.obs
    adata.obs['PMI'] = pmi_value
    
    # Match gene names to single-cell data gene names
    common_genes = np.intersect1d(kernel_params['Gene'].values, adata.var_names)
    print(f"Found {len(common_genes)} common genes in both datasets")
    
    if len(common_genes) == 0:
        print("Warning: No common genes found. Skipping this file.")
        continue
    
    # Get PMI values as an array of the same length as number of cells
    pmi_values = np.full(adata.n_obs, pmi_value)
    
    # Prepare for efficient matrix operations
    print("Preparing expression matrix for correction...")
    
    # Check if the matrix is sparse and convert to the appropriate format
    is_sparse = scipy.sparse.issparse(adata.X)
    
    if is_sparse:
        # For sparse matrix, convert to dense for faster processing if memory allows
        # If memory is a concern, use lil_matrix which is more efficient for updates
        print("Converting sparse matrix to appropriate format for efficient processing...")
        try:
            # Try using dense arrays if memory allows (much faster)
            X_array = adata.X.toarray()
            use_dense = True
            print("Using dense array format for faster processing")
        except MemoryError:
            # Fall back to lil_matrix if not enough memory
            X_array = adata.X.tolil()
            use_dense = False
            print("Using lil_matrix format (memory-efficient but slower)")
    else:
        # If already dense, use as is
        X_array = adata.X.copy()
        use_dense = True
    
    # Create gene parameters dictionary with vectorized calculation
    gene_params = {}
    for gene in common_genes:
        param_row = kernel_params[kernel_params['Gene'] == gene].iloc[0]
        gene_idx = np.where(adata.var_names == gene)[0][0]
        gene_params[gene] = {
            'Alpha': param_row['Alpha'],
            'Gamma': param_row['Gamma'],
            'idx': gene_idx
        }
    
    # Apply PMI correction in batches for more efficient processing
    print("Applying PMI correction to genes:")
    batch_size = 50  # Process genes in batches for progress display
    gene_list = list(gene_params.keys())
    
    for batch_start in tqdm(range(0, len(gene_list), batch_size), desc="Processing gene batches"):
        batch_genes = gene_list[batch_start:batch_start + batch_size]
        
        for gene in batch_genes:
            params = gene_params[gene]
            gene_idx = params['idx']
            alpha = params['Alpha']
            gamma = params['Gamma']
            
            # Calculate PMI effect
            pmi_effect = alpha * pmi_values + gamma * (pmi_values ** 2)
            
            # Apply correction efficiently based on matrix type
            if use_dense:
                # For dense arrays, direct subtraction is fast
                X_array[:, gene_idx] -= pmi_effect
            else:
                # For lil_matrix, we can update the whole column at once
                # Get the original values
                orig_values = X_array[:, gene_idx].toarray().flatten()
                # Apply correction
                corrected_values = orig_values - pmi_effect
                # Update the whole column
                X_array[:, gene_idx] = scipy.sparse.lil_matrix(corrected_values.reshape(-1, 1))
    
    # Convert back to appropriate format for AnnData
    print("Creating corrected AnnData object...")
    if is_sparse and not use_dense:
        # If we used lil_matrix, convert back to csr for storage efficiency
        corrected_expr = X_array.tocsr()
    else:
        corrected_expr = X_array
        
        # If original was sparse but we used dense, convert back to sparse
        if is_sparse:
            corrected_expr = scipy.sparse.csr_matrix(corrected_expr)
    
    # Create a new AnnData object with corrected expression
    adata_corrected = sc.AnnData(
        X=corrected_expr,
        obs=adata.obs,
        var=adata.var,
        uns=adata.uns
    )
    
    # Save corrected data with a new filename
    output_file = os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}_pmi_corrected.h5ad")
    print(f"Saving corrected data...")
    adata_corrected.write(output_file)
    print(f"Saved PMI-corrected data to {output_file}")
    
    # Create visualization for the first file only
    if h5ad_idx == 0 and len(common_genes) > 0:
        print("Creating visualization for an example gene...")
        example_gene = common_genes[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Before correction
        if scipy.sparse.issparse(adata.X):
            gene_expr_before = adata[:, example_gene].X.toarray().flatten()
        else:
            gene_expr_before = adata[:, example_gene].X.flatten()
            
        sns.scatterplot(
            x=[pmi_value] * len(gene_expr_before), 
            y=gene_expr_before, 
            ax=axes[0]
        )
        axes[0].set_title(f"{example_gene} - Before Correction")
        axes[0].set_xlabel("PMI")
        axes[0].set_ylabel("Expression")
        
        # After correction
        if scipy.sparse.issparse(adata_corrected.X):
            gene_expr_after = adata_corrected[:, example_gene].X.toarray().flatten()
        else:
            gene_expr_after = adata_corrected[:, example_gene].X.flatten()
            
        sns.scatterplot(
            x=[pmi_value] * len(gene_expr_after), 
            y=gene_expr_after, 
            ax=axes[1]
        )
        axes[1].set_title(f"{example_gene} - After Correction")
        axes[1].set_xlabel("PMI")
        axes[1].set_ylabel("Expression")
        
        plt.tight_layout()
        visualization_file = os.path.join(args.output_dir, "pmi_correction_example.png")
        plt.savefig(visualization_file)
        print(f"Saved visualization to {visualization_file}")

print("\nAll files processed successfully!")
