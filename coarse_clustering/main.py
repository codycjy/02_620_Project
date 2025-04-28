import argparse
import os
import pandas as pd
import scanpy as sc
import process
import my_kmeans
import cluster_markers
import my_em
import hdf5plugin
import gc  # Add this with other imports


def main():
    parser = argparse.ArgumentParser(description='Cell clustering based on marker genes')

    # Required
    parser.add_argument('--marker_file', type=str, required=True)
    parser.add_argument('--meta_file', type=str, required=True)
    parser.add_argument('--combined_file', type=str, required=False)
    parser.add_argument('--fold_dir', type=str, required=False, default="cv_splits")


    # Optional preprocessing step
    parser.add_argument('--generate_cv_splits', action='store_true', help='Generate cross-validation splits before clustering')
    parser.add_argument('--n_folds', type=int, required=False,default=5, help='Number of folds for cross-validation')
    print("About to add methods argument")
    parser.add_argument('--methods', type=str, required=False, default="kmeans", 
                        help='Clustering methods to use (comma-separated)')
    print("Methods argument added successfully")
    # Clustering params
    parser.add_argument('--n_samples_per_donor', type=int, default=100)
    parser.add_argument('--n_simulations', type=int, default=10)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--max_iterations', type=int, default=300)
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("RUNNING WITH PARAMETERS:")
    for k, v in vars(args).items():
        print(f"{k:25s}: {v}")
    print("=" * 50 + "\n")

    if args.generate_cv_splits:
        print("Generating cross-validation splits...")
        # call your CV split function here, e.g.:
        process.process_and_split_dataset(
            args.meta_file, n_folds=args.n_folds, output_dir=args.fold_dir
        )
        print("CV splits generated.")
        return
    
    # Parse marker and metadata
    marker_dict = process.parse_marker_gene_file(args.marker_file)
    meta_data = process.parse_meta_data(args.meta_file)
    marker_list = [g for v in marker_dict.values() for g in v]

    print("Loading combined dataset...")
    adata = sc.read_h5ad(args.combined_file)
    print("Combined dataset loaded.")

    # Process all folds under args.fold_dir
    fold_dirs = sorted([
        os.path.join(args.fold_dir, d)
        for d in os.listdir(args.fold_dir)
        if os.path.isdir(os.path.join(args.fold_dir, d)) and d.startswith("fold")
    ])

    for fold_path in fold_dirs:
        fold_name = os.path.basename(fold_path)
        print(f"\nProcessing {fold_name}...")

        train_donors_path = os.path.join(fold_path, 'train_donors.csv')
        test_donors_path = os.path.join(fold_path, 'test_donors.csv')

        if not os.path.exists(train_donors_path) or not os.path.exists(test_donors_path):
            print(f"  Missing donor CSV files in {fold_path}. Skipping.")
            continue

        train_donors = pd.read_csv(train_donors_path)['Donor ID'].tolist()
        test_donors = pd.read_csv(test_donors_path)['Donor ID'].tolist()

        # Full data subsets (all genes)
        train_full = adata[adata.obs['Donor ID'].isin(train_donors)]
        test_full = adata[adata.obs['Donor ID'].isin(test_donors)]

        # Marker genes only for clustering
        marker_genes_present = [g for g in marker_list if g in train_full.var_names]
        print(f"  Using {len(marker_genes_present)} marker genes")
        print(marker_genes_present)

        train_marker = train_full[:, marker_genes_present]
        test_marker = test_full[:, marker_genes_present]

        print(f"  Train shape (marker only): {train_marker.shape}")
        print(f"  Test shape  (marker only): {test_marker.shape}")
        results = {}
        all_means, result = cluster_markers.cluster_all_data(
            train_marker,
            args.n_samples_per_donor,
            args.n_simulations,
            args.k,
            args.random_state,
            args.max_iterations,
            args.methods
        )
        results['intial'] = result  
        pd.DataFrame(all_means).to_csv(os.path.join(fold_path, args.methods + "_meta_means.csv"), index=False)
        if args.methods == "kmeans":
            final_means, wcss, _ = my_kmeans.k_means(
                all_means,
                args.k,
                args.max_iterations,
                args.random_state
            )
            results['final'] = {'wcss':wcss}
        elif args.methods == "em":
            final_means, covariances, weights, log_likelihood, responsibilities = my_em.gaussian_mixture_em(
                all_means,
                args.k,
                max_iterations=args.max_iterations,
                tol=1e-6,
                random_state=args.random_state
            )
            results['final'] = {
                'covariances': covariances,
                'weights': weights,
                'log_likelihood': log_likelihood,
                'responsibilities': responsibilities
    }
        else:
            raise ValueError(f"Unknown clustering method: {args.methods}")

        cluster_means_dict = cluster_markers.create_cluster_means_dict(final_means)
        pd.DataFrame(final_means).to_csv(os.path.join(fold_path, args.methods + "_final_means.csv"), index=False)
        train_output_path = os.path.join(fold_path, args.methods + "_train_labeled.csv")
        test_output_path = os.path.join(fold_path, args.methods + "_test_labeled.csv")

        # Use full expression matrix to calculate per-cluster mean (over all genes),
        # but use marker-based clustering
        cluster_markers.label_dataset_with_clusters_marker_distance_all_genes_mean(
            train_full, cluster_means_dict, marker_genes_present, train_output_path
        )

        cluster_markers.label_dataset_with_clusters_marker_distance_all_genes_mean(
        test_full, cluster_means_dict, marker_genes_present, test_output_path
)

        print(f"  Saved labeled data to:")
        print(f"    -> {train_output_path}")
        print(f"    -> {test_output_path}")

        # Clean up memory
        del train_donors, test_donors
        del train_full, test_full
        del train_marker, test_marker
        del all_means, final_means, cluster_means_dict
        gc.collect()

    print("\nAll folds processed.")


if __name__ == "__main__":
    main()
