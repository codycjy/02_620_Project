# Check the results of clustering & Decide parameters
# Not finished yet
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import numpy as np
from tqdm import tqdm

def check_cluster_numbers(means, final_means):

    ## cluster of cluster_mean
    import my_kmeans.kmeans as kmeans
    importlib.reload(kmeans)
    from sklearn.metrics import silhouette_score
    # ------------------------------------------
    # 1. Compute WCSS for different K values
    # ------------------------------------------
    silhouette_scores = []
    wcss_values = []
    K_range = range(10, 50)  # Test cluster numbers from 1 to 10
    for k in tqdm(K_range,desc = "Processing"):
        _, wcss, labels = kmeans.k_means(np.array(means.T), k, max_iterations= None, random_state=42)
        wcss_values.append(wcss)  # Inertia is the WCSS (sum of squared distances)
        score = silhouette_score(means.T, labels)
        silhouette_scores.append(score)

    # ------------------------------------------
    # 2. Plot WCSS vs. Number of Clusters (Elbow Method)
    # ------------------------------------------
    plt.figure(figsize=(10, 6))  # Increase figure size for clarity
    plt.plot(K_range, np.log(wcss_values), marker='o', linestyle='-', color='b', markersize=8, linewidth=2, label='WCSS (Log Scale)') 
    plt.xlabel("Number of Clusters (K)", fontsize=14)
    plt.ylabel("Log Within-Cluster Sum of Squares (WCSS)", fontsize=14)
    plt.title("Elbow Method for Optimal K", fontsize=16)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    # ------------------------------------------
    # Plotting the silhouette scores
    # ------------------------------------------
    plt.figure(figsize=(10, 6))  # Increase figure size for clarity
    plt.plot(K_range, silhouette_scores, marker='s', linestyle='-', color='g', markersize=8, linewidth=2, label='Silhouette Scores')
    plt.title('Silhouette Scores for Different K', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    return


def check_cluster_results(adata):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    true_labels = adata.obs['Class']  # or 'Class'
    pred_labels = adata.obs['kmeans_labels']

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
    return 

def plot_cluster_dist(adata):

    # Count the number of cells in each original class (e.g., 'Class' or 'Subclass')
    cell_counts = adata.obs['Class'].value_counts()
    subcell_counts = adata.obs['Subclass'].value_counts()
    print(cell_counts,subcell_counts)
    # Create a contingency table (cross-tabulation) between cluster labels and true labels
    cluster_vs_true = pd.crosstab(adata.obs['kmeans_labels'], adata.obs['Class'], dropna=False)

    # Plot as a heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cluster_vs_true, annot=True, cmap='Blues', fmt='d', cbar=False, 
                xticklabels=cluster_vs_true.columns, yticklabels=cluster_vs_true.index)
    plt.title("Cluster vs True Cell Type Distribution (Heatmap)")
    plt.xlabel("True Cell Types")
    plt.ylabel("Cluster Labels")
    plt.show()
    return

# Generate summary statistics for cell counts
def check_cell_count(cell_count_df, output_path=None):
    """
    Generate summary statistics for cell counts across donors and clusters.
    
    Parameters:
    -----------
    cell_count_df : pandas.DataFrame
        DataFrame containing 'Donor ID', 'cluster', and 'cell_count' columns.
    output_path : str, optional
        Path to save the summary statistics.
        
    Returns:
    --------
    summary_dict : dict
        Dictionary containing summary statistics.
    """
    # Summary by donor
    donor_summary = cell_count_df.groupby('Donor ID')['cell_count'].agg(['sum', 'mean', 'median', 'std', 'min', 'max'])
    donor_summary.columns = ['Total Cells', 'Mean Cells per Cluster', 'Median Cells per Cluster', 
                            'Std Dev Cells per Cluster', 'Min Cells per Cluster', 'Max Cells per Cluster']
    
    # Calculate number of clusters with cells
    clusters_with_cells = cell_count_df[cell_count_df['cell_count'] > 0].groupby('Donor ID')['cluster'].nunique()
    donor_summary['Clusters with Cells'] = clusters_with_cells
    donor_summary['Percent Clusters with Cells'] = (clusters_with_cells / len(cell_count_df['cluster'].unique())) * 100
    
    # Summary by cluster
    cluster_summary = cell_count_df.groupby('cluster')['cell_count'].agg(['sum', 'mean', 'median', 'std', 'min', 'max'])
    cluster_summary.columns = ['Total Cells', 'Mean Cells per Donor', 'Median Cells per Donor', 
                              'Std Dev Cells per Donor', 'Min Cells per Donor', 'Max Cells per Donor']
    
    # Calculate number of donors with cells in each cluster
    donors_with_cells = cell_count_df[cell_count_df['cell_count'] > 0].groupby('cluster')['Donor ID'].nunique()
    cluster_summary['Donors with Cells'] = donors_with_cells
    cluster_summary['Percent Donors with Cells'] = (donors_with_cells / len(cell_count_df['Donor ID'].unique())) * 100
    
    # Overall summary
    overall = {
        'Total Cells': cell_count_df['cell_count'].sum(),
        'Total Donors': len(cell_count_df['Donor ID'].unique()),
        'Total Clusters': len(cell_count_df['cluster'].unique()),
        'Mean Cells per Donor': donor_summary['Total Cells'].mean(),
        'Median Cells per Donor': donor_summary['Total Cells'].median(),
        'Min Cells per Donor': donor_summary['Total Cells'].min(),
        'Max Cells per Donor': donor_summary['Total Cells'].max(),
        'Mean Cells per Cluster': cluster_summary['Total Cells'].mean(),
        'Median Cells per Cluster': cluster_summary['Total Cells'].median(),
        'Min Cells per Cluster': cluster_summary['Total Cells'].min(),
        'Max Cells per Cluster': cluster_summary['Total Cells'].max(),
    }
    
    # Add percentage of donor-cluster combinations with cells
    nonzero_combos = (cell_count_df['cell_count'] > 0).sum()
    total_combos = len(cell_count_df)
    overall['Percent Donor-Cluster Combinations with Cells'] = (nonzero_combos / total_combos) * 100
    
    # Create summary dictionary
    summary_dict = {
        'overall': pd.Series(overall),
        'donor_summary': donor_summary,
        'cluster_summary': cluster_summary
    }
    
    # Save if output path is provided
    if output_path:
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame([overall]).T.reset_index().rename(columns={'index': 'Metric', 0: 'Value'}).to_excel(writer, sheet_name='Overall', index=False)
            donor_summary.reset_index().to_excel(writer, sheet_name='By Donor', index=False)
            cluster_summary.reset_index().to_excel(writer, sheet_name='By Cluster', index=False)
        print(f"Saved summary statistics to '{output_path}'")
    
    return summary_dict

