import argparse
import process
import my_kmeans
import cluster_markers
import hdf5plugin
def main():
    parser = argparse.ArgumentParser(description='Cell clustering based on marker genes')
    
    # Required parameters
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training data file (e.g., AnnData h5ad file)')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test data file (e.g., AnnData h5ad file)')
    parser.add_argument('--marker_file', type=str, required=True,
                        help='Path to file containing marker gene names')
    parser.add_argument('--meta_file', type=str, required=True,
                        help='Path to metadata file with donor information')
    
    # Optional parameters with defaults
    parser.add_argument('--n_samples_per_donor', type=int, default=100,
                        help='Number of cells to sample from each donor (default: 100)')
    parser.add_argument('--n_simulations', type=int, default=10,
                        help='Number of sampling and clustering iterations (default: 50)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of clusters for k-means algorithm (default: 10)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_iterations', type=int, default=300,
                        help='Maximum number of iterations for k-means (default: 300)')
    parser.add_argument('--train_output', type=str, default='training_cell_count_summary.xlsx',
                        help='Output file for training data cluster summary (default: training_cell_count_summary.xlsx)')
    parser.add_argument('--test_output', type=str, default='test_cell_count_summary.xlsx',
                        help='Output file for test data cluster summary (default: test_cell_count_summary.xlsx)')
    
    args = parser.parse_args()
    
    # Print all parameters before running
    print("\n" + "="*50)
    print("RUNNING WITH THE FOLLOWING PARAMETERS:")
    print("="*50)
    print(f"Train file:           {args.train_file}")
    print(f"Test file:            {args.test_file}")
    print(f"Marker file:          {args.marker_file}")
    print(f"Metadata file:        {args.meta_file}")
    print(f"Samples per donor:    {args.n_samples_per_donor}")
    print(f"Number of simulations: {args.n_simulations}")
    print(f"Number of clusters (k): {args.k}")
    print(f"Random state:         {args.random_state}")
    print(f"Max iterations:       {args.max_iterations}")
    print(f"Train output file:    {args.train_output}")
    print(f"Test output file:     {args.test_output}")
    print("="*50 + "\n")
    
    # Parse marker genes and metadata
    print("Parsing marker genes and metadata...")
    marker_dict = process.parse_marker_gene_file(args.marker_file)
    meta_data = process.parse_meta_data(args.meta_file)
    
   
    marker_list = []
    for marker_type in marker_dict.keys():
        marker_list.extend(marker_dict[marker_type])
    # Parse dataset
    print("Parsing dataset...")
    train_data, test_data, donor_groups = process.parse_dataset(args.train_file, args.test_file, meta_data)
    # Filter data to only include marker genes that are present in the dataset
    marker_genes_present = [g for g in marker_list if g in train_data.var_names]

    print(f"Found {len(marker_genes_present)} marker genes in the dataset out of {len(marker_list)} total markers")
    
    train_data = train_data[:, marker_genes_present]
    test_data = test_data[:, marker_genes_present]
    print(f"Training data shape after filtering: {train_data.shape}")
    print(f"Test data shape after filtering: {test_data.shape}")
    
    # Cluster all data through multiple simulations
    print(f"Running clustering with {args.n_simulations} simulations...")
    all_means = cluster_markers.cluster_all_data(
        train_data,
        args.n_samples_per_donor,
        args.n_simulations,
        args.k,
        args.random_state,
        args.max_iterations
    )
    
    # Perform final k-means clustering on the collected means
    print("Performing final k-means clustering...")
    final_means,_,_ = my_kmeans.k_means(all_means, args.k, args.max_iterations, args.random_state)
    
    # Create dictionary mapping cluster indices to mean vectors
    cluster_means_dict = cluster_markers.create_cluster_means_dict(final_means)
    
    # Label datasets with clusters and generate summaries
    print("Labeling datasets with clusters...")
    labeled_train_data = cluster_markers.label_dataset_with_clusters(train_data, cluster_means_dict, args.train_output)
    labeled_test_data = cluster_markers.label_dataset_with_clusters(test_data, cluster_means_dict, args.test_output)
    
    print(f"Clustering complete. Training data summary saved to {args.train_output}")
    print(f"Test data summary saved to {args.test_output}")
    
    return labeled_train_data, labeled_test_data, cluster_means_dict

if __name__ == "__main__":
    main()