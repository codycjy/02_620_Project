import os
import glob
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
from tqdm.auto import tqdm

def create_pseudobulk_from_directory(directory_path):
    """
    Create simple pseudobulk data from all h5ad files in a directory and save to CSV.
    
    Parameters:
        directory_path: Path to directory containing h5ad files
        
    Returns:
        pseudobulk_data: DataFrame containing pseudobulk data (averaged)
    """
    # Get all h5ad files in the directory
    h5ad_files = glob.glob(os.path.join(directory_path, "*.h5ad"))
    
    if not h5ad_files:
        raise ValueError(f"No h5ad files found in directory: {directory_path}")
    
    print(f"Found {len(h5ad_files)} h5ad files in {directory_path}")
    
    # Dictionary to store donor aggregated data
    donor_data = {}
    donor_genes = {}
    
    # Common genes tracking
    all_genes = set()
    
    # Process each file
    for file_path in tqdm(h5ad_files, desc="Processing files"):
        file_name = os.path.basename(file_path)  
        donor_id = os.path.splitext(file_name)[0] 
        
        try:
            # Load individual file
            adata = sc.read_h5ad(file_path)
            
            # Store gene names for this donor
            donor_genes[donor_id] = list(adata.var_names)
            
            # Calculate average expression values for this donor
            if scipy.sparse.issparse(adata.X):
                expr_sum = adata.X.sum(axis=0).A1  # Sum along cells axis
            else:
                expr_sum = adata.X.sum(axis=0)
            
            n_cells = adata.n_obs
            if n_cells > 0:
                donor_avg = expr_sum / n_cells
            else:
                donor_avg = expr_sum 
            
            # Store the averaged data
            donor_data[donor_id] = donor_avg
            
            # Update set of all genes
            all_genes.update(adata.var_names)
            
            # Free memory
            del adata
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    if not donor_data:
        raise ValueError("No valid data was processed from any h5ad file.")
    
    # Convert all_genes to a sorted list
    all_genes = sorted(list(all_genes))
    print(f"Found {len(all_genes)} unique genes across all donors")
    
    # Create a DataFrame with donors as rows and genes as columns
    pseudobulk_data = pd.DataFrame(0.0, index=donor_data.keys(), columns=all_genes)
    
    # Fill in the data for each donor
    for donor_id, expr in tqdm(donor_data.items(), desc="Creating pseudobulk matrix"):
        # Get the genes for this donor's dataset
        donor_gene_list = donor_genes[donor_id]
        
        # For each gene in this donor's dataset, update the pseudobulk DataFrame
        for i, gene in enumerate(donor_gene_list):
            if gene in pseudobulk_data.columns:
                pseudobulk_data.loc[donor_id, gene] = expr[i]
    
    # Save the pseudobulk data to a CSV file
    output_path = os.path.join(directory_path, "pseudobulk_data.csv")
    pseudobulk_data.to_csv(output_path)
    print(f"Successfully created pseudobulk data with {len(pseudobulk_data)} donors and {len(all_genes)} genes")
    print(f"Pseudobulk data saved to {output_path}")
    
    return pseudobulk_data