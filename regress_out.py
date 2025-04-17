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
import random
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Apply PMI correction to single-cell data and split into train/test/valid datasets')
    
    # Basic parameters
    parser.add_argument('--param_file', type=str, required=True, 
                        help='Path to CSV file containing kernel regression parameters')
    parser.add_argument('--meta_file', type=str, required=True, 
                        help='Path to CSV file containing metadata with Donor ID and PMI information')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing h5ad files to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save corrected files (defaults to data_dir)')
    
    # Dataset splitting parameters
    parser.add_argument('--split_data', action='store_true',
                        help='Split corrected data into train/test/valid sets')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of donors to use for training set (default: 0.7)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of donors to use for test set (default: 0.15)')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='Ratio of donors to use for validation set (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducible splitting (default: 42)')
    parser.add_argument('--merge_datasets', action='store_true',
                        help='Merge all donors in each split into a single h5ad file')
    
    return parser.parse_args()

def apply_pmi_correction(args):
    """
    Apply PMI correction to h5ad files based on kernel regression parameters.
    Returns a list of processed donor IDs and their corresponding corrected files.
    """
    processed_donors = []
    corrected_files = []
    
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
        adata.obs['donor'] = donor_ID  # Add donor information to observations
        
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
                
                # Calculate PMI effect using quadratic model: alpha*PMI + gamma*PMI^2
                pmi_effect = alpha * pmi_values + gamma * (pmi_values ** 2)
                
                # Apply correction efficiently based on matrix type
                if use_dense:
                    # For dense arrays, direct subtraction is fast
                    X_array[:, gene_idx] -= pmi_effect
                else:
                    # For lil_matrix, we can update the whole column at once
                    orig_values = X_array[:, gene_idx].toarray().flatten()
                    corrected_values = orig_values - pmi_effect
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
        
        # Track processed donors and files
        processed_donors.append(donor_ID)
        corrected_files.append(output_file)
        
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
    return processed_donors, corrected_files

def split_dataset(processed_donors, corrected_files, args):
    """
    Split corrected data files into train/test/valid sets based on donor IDs.
    This ensures that cells from the same donor are not split across different sets.
    
    Args:
        processed_donors: List of donor IDs
        corrected_files: List of corrected h5ad file paths
        args: Command line arguments
        
    Returns:
        Dictionary containing split information
    """
    print("\n=== Splitting datasets into train/test/valid sets ===")
    
    # Ensure ratios sum to 1
    total_ratio = args.train_ratio + args.test_ratio + args.valid_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Train/test/valid ratios sum to {total_ratio}, not 1.0")
        args.train_ratio /= total_ratio
        args.test_ratio /= total_ratio
        args.valid_ratio /= total_ratio
        print(f"Adjusted ratios: train={args.train_ratio}, test={args.test_ratio}, valid={args.valid_ratio}")
    
    # Set random seed for reproducible results
    random.seed(args.random_seed)
    
    # Create directories for each split
    train_dir = os.path.join(args.output_dir, 'train')
    test_dir = os.path.join(args.output_dir, 'test')
    valid_dir = os.path.join(args.output_dir, 'valid')
    
    for directory in [train_dir, test_dir, valid_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Group files by donor
    donor_to_files = {}
    for donor_id, file_path in zip(processed_donors, corrected_files):
        if donor_id not in donor_to_files:
            donor_to_files[donor_id] = []
        donor_to_files[donor_id].append(file_path)
    
    # Shuffle donor list
    donor_ids = list(donor_to_files.keys())
    random.shuffle(donor_ids)
    
    # Calculate number of donors for each split
    n_donors = len(donor_ids)
    n_train = max(1, int(n_donors * args.train_ratio))
    n_test = max(1, int(n_donors * args.test_ratio))
    n_valid = n_donors - n_train - n_test
    
    # Ensure at least one donor in each split
    if n_valid < 1:
        if n_train > 1:
            n_train -= 1
            n_valid = 1
        elif n_test > 1:
            n_test -= 1
            n_valid = 1
        else:
            print("Warning: Not enough donors to split into three sets!")
    
    # Split donor IDs
    train_donors = donor_ids[:n_train]
    test_donors = donor_ids[n_train:n_train+n_test]
    valid_donors = donor_ids[n_train+n_test:]
    
    print(f"Split {n_donors} donors into: {n_train} train, {n_test} test, {n_valid} validation")
    
    # Copy files to respective directories
    split_files = {'train': [], 'test': [], 'valid': []}
    
    for donor_id in train_donors:
        for file_path in donor_to_files[donor_id]:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(train_dir, filename)
            shutil.copy2(file_path, dest_path)
            split_files['train'].append(dest_path)
        print(f"Donor {donor_id} assigned to training set")
    
    for donor_id in test_donors:
        for file_path in donor_to_files[donor_id]:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(test_dir, filename)
            shutil.copy2(file_path, dest_path)
            split_files['test'].append(dest_path)
        print(f"Donor {donor_id} assigned to test set")
    
    for donor_id in valid_donors:
        for file_path in donor_to_files[donor_id]:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(valid_dir, filename)
            shutil.copy2(file_path, dest_path)
            split_files['valid'].append(dest_path)
        print(f"Donor {donor_id} assigned to validation set")
    
    # Optional: Merge datasets within each split
    if args.merge_datasets:
        merge_split_datasets(split_files, args)
    
    # Save split information
    split_info = {
        'train_donors': train_donors,
        'test_donors': test_donors,
        'valid_donors': valid_donors,
        'train_files': [os.path.basename(f) for f in split_files['train']],
        'test_files': [os.path.basename(f) for f in split_files['test']],
        'valid_files': [os.path.basename(f) for f in split_files['valid']]
    }
    
    with open(os.path.join(args.output_dir, 'split_info.txt'), 'w') as f:
        for key, value in split_info.items():
            f.write(f"{key}: {value}\n")
    
    print("\nDataset splitting completed successfully!")
    return split_info

def merge_split_datasets(split_files, args):
    """
    Merge all h5ad files within each split (train/test/valid) into a single h5ad file.
    This is useful for downstream analysis that requires a single file.
    
    Args:
        split_files: Dictionary containing file paths for each split
        args: Command line arguments
    """
    print("\n=== Merging files within each split ===")
    
    for split_name, files in split_files.items():
        if not files:
            print(f"No files to merge for {split_name} split")
            continue
        
        print(f"Merging {len(files)} files for {split_name} split...")
        
        # Load first file as base
        combined_adata = sc.read(files[0])
        
        # Add remaining files
        for i, file_path in enumerate(files[1:], 1):
            print(f"  Adding file {i}/{len(files)-1}: {os.path.basename(file_path)}")
            current_adata = sc.read(file_path)
            
            # Ensure var_names match (gene names must match)
            if not np.array_equal(combined_adata.var_names, current_adata.var_names):
                common_genes = np.intersect1d(combined_adata.var_names, current_adata.var_names)
                if len(common_genes) == 0:
                    print(f"Warning: No common genes between files, skipping {os.path.basename(file_path)}")
                    continue
                print(f"  Subsetting to {len(common_genes)} common genes")
                combined_adata = combined_adata[:, common_genes]
                current_adata = current_adata[:, common_genes]
            
            # Concatenate datasets
            combined_adata = combined_adata.concatenate(current_adata, 
                                                      join='inner', 
                                                      batch_key=f"{split_name}_batch",
                                                      batch_categories=[os.path.basename(files[0]), 
                                                                     os.path.basename(file_path)])
        
        # Save merged dataset
        output_file = os.path.join(args.output_dir, f"{split_name}_merged.h5ad")
        print(f"Saving merged {split_name} dataset with {combined_adata.n_obs} cells and {combined_adata.n_vars} genes")
        combined_adata.write(output_file)
    
    print("Dataset merging completed!")

def main():
    """
    Main function to execute the PMI correction and dataset splitting workflow.
    """
    # Parse command line arguments
    args = parse_args()
    
    # If no output directory specified, use input directory
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Apply PMI correction
    processed_donors, corrected_files = apply_pmi_correction(args)
    
    # If splitting data is requested and we have processed donors
    if args.split_data and processed_donors:
        split_info = split_dataset(processed_donors, corrected_files, args)
        
        # Print split statistics
        print("\n=== Dataset Split Summary ===")
        print(f"Total donors: {len(processed_donors)}")
        print(f"Training set: {len(split_info['train_donors'])} donors")
        print(f"Test set: {len(split_info['test_donors'])} donors")
        print(f"Validation set: {len(split_info['valid_donors'])} donors")
        
        if args.merge_datasets:
            print("\nMerged datasets created:")
            print(f"  - {os.path.join(args.output_dir, 'train_merged.h5ad')}")
            print(f"  - {os.path.join(args.output_dir, 'test_merged.h5ad')}")
            print(f"  - {os.path.join(args.output_dir, 'valid_merged.h5ad')}")

if __name__ == "__main__":
    main()
