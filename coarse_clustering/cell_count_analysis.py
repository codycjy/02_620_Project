def visualize_cluster_boxplots(reassigned_counts_df):
    """
    Create boxplots to show the distribution of cell counts for each cluster across donors
    """
    n_clusters = len([col for col in reassigned_counts_df.columns if col.startswith('Cluster')])
    n_folds = reassigned_counts_df['Fold'].nunique()
    
    # Create a figure with subplots - one row for each cluster
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]
    
    # Prepare data for boxplots
    for cluster_idx in range(n_clusters):
        ax = axes[cluster_idx]
        
        # Collect data for each fold
        fold_data = []
        fold_labels = []
        
        for fold in range(1, n_folds + 1):
            # Get this fold's data
            fold_counts = reassigned_counts_df[reassigned_counts_df['Fold'] == fold]
            
            # Extract cell counts for this cluster
            cluster_counts = fold_counts[f'Cluster {cluster_idx}'].values
            
            # Add to data list
            fold_data.append(cluster_counts)
            fold_labels.append(f'Fold {fold}')
        
        # Create boxplot
        bp = ax.boxplot(fold_data, labels=fold_labels, patch_artist=True)
        
        # Customize boxplot colors
        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 0.8, n_folds))):
            patch.set_facecolor(color)
        
        # Add jittered data points for better visibility
        for i, data in enumerate(fold_data):
            # Calculate jitter
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=20, color='darkblue')
        
        # Customize plot
        ax.set_title(f'Cluster {cluster_idx} Cell Count Distribution')
        ax.set_ylabel('Cell Count')
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add statistics as text
        stats_text = []
        for i, data in enumerate(fold_data):
            stats_text.append(f"Fold {i+1}: Mean={np.mean(data):.1f}, Median={np.median(data):.1f}, SD={np.std(data):.1f}")
        
        ax.text(0.02, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    return fig

def visualize_cluster_boxplots_combined(reassigned_counts_df):
    """
    Create a combined boxplot showing cell count distributions for all clusters side by side
    """
    n_clusters = len([col for col in reassigned_counts_df.columns if col.startswith('Cluster')])
    n_folds = reassigned_counts_df['Fold'].nunique()
    
    # Prepare data for plotting
    plot_data = []
    
    # Reshape data into format for grouped boxplot
    for fold in range(1, n_folds + 1):
        fold_data = reassigned_counts_df[reassigned_counts_df['Fold'] == fold]
        
        for cluster in range(n_clusters):
            cluster_counts = fold_data[f'Cluster {cluster}'].values
            
            for count in cluster_counts:
                plot_data.append({
                    'Fold': f'Fold {fold}',
                    'Cluster': f'Cluster {cluster}',
                    'Cell Count': count
                })
    
    # Convert to DataFrame
    df_plot = pd.DataFrame(plot_data)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create boxplot
    ax = sns.boxplot(data=df_plot, x='Cluster', y='Cell Count', hue='Fold', palette='viridis')
    
    # Add swarm plot for better visualization of distribution
    sns.swarmplot(data=df_plot, x='Cluster', y='Cell Count', hue='Fold', 
                 dodge=True, alpha=0.5, size=2, color='black')
    
    # Customize plot
    plt.title('Cell Count Distribution by Cluster and Fold', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Cell Count', fontsize=14)
    
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    n_labels = len(labels) // 2  # Because swarmplot adds duplicate legend entries
    plt.legend(handles[:n_labels], labels[:n_labels], title='Fold', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()  # Get current figure
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec

def load_final_means(base_path, fold_number, methods):
    """Load the kmeans_final_means.csv file from a specific fold directory"""
    fold_path = os.path.join(base_path, f'fold_{fold_number}', methods + '_final_means.csv')
    return pd.read_csv(fold_path)

def load_cell_counts(base_path, fold_number, methods):
    """Load the labeled_cell_count_pivot.csv file"""
    file_path = os.path.join(base_path, f'fold_{fold_number}', methods + '_train_labeled_cell_counts_pivot.csv')
    return pd.read_csv(file_path)

def calculate_assignment_matrix(reference_centroids, target_centroids):
    """Calculate cost matrix for Hungarian algorithm (centroid distances)"""
    return pairwise_distances(reference_centroids, target_centroids)

def reassign_clusters(all_folds_means):
    """
    Reassign cluster labels to maintain consistency across folds
    using the Hungarian algorithm (minimum assignment problem)
    """
    n_folds = len(all_folds_means)
    n_clusters = all_folds_means[0].shape[0]
    n_features = all_folds_means[0].shape[1]
    
    # Use the first fold as reference
    reference_means = all_folds_means[0]
    
    reassigned_means = [reference_means]
    reassignment_maps = [np.arange(n_clusters)]  # Identity map for the reference fold
    
    # For each other fold, find optimal assignment to reference
    for fold_idx in range(1, n_folds):
        target_means = all_folds_means[fold_idx]
        
        # Calculate assignment cost matrix (distances between centroids)
        cost_matrix = calculate_assignment_matrix(reference_means, target_means)
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Reorder the target fold's clusters based on assignment
        reordered_means = target_means[col_ind, :]
        reassigned_means.append(reordered_means)
        
        # Save the reordering map for this fold
        reassignment_maps.append(col_ind)
    
    return np.array(reassigned_means), reassignment_maps

def validate_reassignment(reassigned_counts_df):
    """
    Validate that clusters have similar sizes across folds after reassignment
    """
    n_folds = reassigned_counts_df['Fold'].nunique()
    n_clusters = len([col for col in reassigned_counts_df.columns if col.startswith('Cluster')])
    
    # Calculate total cell count for each cluster in each fold
    fold_cluster_totals = []
    
    for fold in range(1, n_folds + 1):
        fold_data = reassigned_counts_df[reassigned_counts_df['Fold'] == fold]
        cluster_totals = []
        
        for cluster in range(n_clusters):
            total = fold_data[f'Cluster {cluster}'].sum()
            cluster_totals.append(total)
        
        fold_cluster_totals.append({
            'Fold': fold,
            'Cluster Totals': cluster_totals
        })
    
    # Calculate coefficient of variation for each cluster's size across folds
    cluster_cv = []
    for cluster in range(n_clusters):
        cluster_sizes = [fold_data['Cluster Totals'][cluster] for fold_data in fold_cluster_totals]
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        cv = 100 * std_size / mean_size if mean_size > 0 else 0
        
        cluster_cv.append({
            'Cluster': cluster,
            'Mean Size': mean_size,
            'Std Dev': std_size,
            'CV (%)': cv
        })
    
    # Print validation results
    print("\nCluster Size Consistency Check:")
    for cluster_data in cluster_cv:
        print(f"Cluster {cluster_data['Cluster']}: " +
              f"Mean Size = {cluster_data['Mean Size']:.1f}, " +
              f"CV = {cluster_data['CV (%)']:.2f}%")
    
    # Calculate overall consistency metric
    avg_cv = np.mean([data['CV (%)'] for data in cluster_cv])
    print(f"\nAverage Coefficient of Variation: {avg_cv:.2f}%")
    print(f"Lower CV indicates more consistent cluster sizes across folds")
    
    # If CV is high, the reassignment may not be optimal
    if avg_cv > 20:  # Arbitrary threshold
        print("\nWARNING: High variation in cluster sizes across folds.")
        print("Consider alternative cluster matching methods.")
    
    return cluster_cv

def reassign_cell_counts(all_cell_counts_dfs, reassignment_maps):
    """
    Apply the cluster reassignment maps to the cell counts dataframe for each fold
    """
    n_folds = len(reassignment_maps)
    reassigned_counts = []
    
    for fold_idx, (fold_df, remap) in enumerate(zip(all_cell_counts_dfs, reassignment_maps)):
        # Create a copy of the dataframe for this fold
        fold_counts = fold_df.copy()
        
        # Get donor IDs
        donor_ids = fold_counts['Donor ID'].values
        
        # Extract the cluster columns (excluding 'Donor ID')
        cluster_cols = [col for col in fold_counts.columns if col != 'Donor ID']
        n_clusters = len(cluster_cols)
        
        # Create a new dataframe with donor IDs
        new_fold_counts = pd.DataFrame({'Donor ID': donor_ids})
        
        # Apply the reassignment map to the cluster columns
        for i in range(n_clusters):
            new_idx = remap[i]
            # Make sure the index is valid
            if new_idx < n_clusters:
                new_fold_counts[f'Cluster {i}'] = fold_counts.iloc[:, new_idx + 1].values
            else:
                print(f"Warning: Invalid cluster index {new_idx} for fold {fold_idx+1}")
                new_fold_counts[f'Cluster {i}'] = 0
        
        # Add fold identifier
        new_fold_counts['Fold'] = fold_idx + 1
        reassigned_counts.append(new_fold_counts)
    
    # Concatenate all folds
    return pd.concat(reassigned_counts, ignore_index=True)

def calculate_fold_stability(reassigned_means):
    """
    Calculate stability metrics across folds based on reassigned means
    """
    n_folds = len(reassigned_means)
    n_clusters = reassigned_means[0].shape[0]
    
    # Calculate pairwise distances between fold means
    stability_matrix = np.zeros((n_folds, n_folds))
    
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            # Calculate average distance between corresponding clusters
            cluster_distances = np.sqrt(np.sum((reassigned_means[i] - reassigned_means[j])**2, axis=1))
            avg_distance = np.mean(cluster_distances)
            stability_matrix[i, j] = avg_distance
            stability_matrix[j, i] = avg_distance
    
    # Calculate stability metrics
    upper_triangle = np.triu_indices(n_folds, k=1)
    stability_scores = stability_matrix[upper_triangle]
    
    mean_stability = np.mean(stability_scores)
    std_stability = np.std(stability_scores)
    
    return {
        'stability_matrix': stability_matrix,
        'mean_stability': mean_stability,
        'std_stability': std_stability,
        'all_scores': stability_scores
    }

def calculate_cell_count_similarity(reassigned_counts_df):
    """
    Calculate similarity metrics for cell counts across folds,
    handling the case where folds have different donor sets
    """
    n_folds = reassigned_counts_df['Fold'].nunique()
    n_clusters = len([col for col in reassigned_counts_df.columns if col.startswith('Cluster')])
    
    # Create a similarity matrix for folds
    similarity_matrix = np.zeros((n_folds, n_folds))
    
    # For each fold pair, calculate correlation between cell counts
    for i in range(1, n_folds + 1):
        for j in range(i, n_folds + 1):
            if i == j:
                similarity_matrix[i-1, j-1] = 1.0  # Perfect correlation with self
                continue
            
            fold_i_data = reassigned_counts_df[reassigned_counts_df['Fold'] == i]
            fold_j_data = reassigned_counts_df[reassigned_counts_df['Fold'] == j]
            
            # Find common donors between the two folds
            common_donors = np.intersect1d(fold_i_data['Donor ID'].values, 
                                           fold_j_data['Donor ID'].values)
            
            if len(common_donors) == 0:
                print(f"Warning: No common donors between Fold {i} and Fold {j}. Using default similarity of 0.5.")
                similarity_matrix[i-1, j-1] = 0.5
                similarity_matrix[j-1, i-1] = 0.5
                continue
            
            # Filter data to only include common donors
            fold_i_filtered = fold_i_data[fold_i_data['Donor ID'].isin(common_donors)]
            fold_j_filtered = fold_j_data[fold_j_data['Donor ID'].isin(common_donors)]
            
            # Sort both dataframes by Donor ID to ensure alignment
            fold_i_filtered = fold_i_filtered.sort_values('Donor ID').reset_index(drop=True)
            fold_j_filtered = fold_j_filtered.sort_values('Donor ID').reset_index(drop=True)
            
            # Calculate correlation for each cluster's cell counts
            correlations = []
            
            for cluster in range(n_clusters):
                cluster_i = fold_i_filtered[f'Cluster {cluster}'].values
                cluster_j = fold_j_filtered[f'Cluster {cluster}'].values
                
                # Check for constant values (which would cause NaN correlation)
                if np.std(cluster_i) == 0 or np.std(cluster_j) == 0:
                    # If both are constant and equal, perfect correlation
                    if np.mean(cluster_i) == np.mean(cluster_j):
                        correlations.append(1.0)
                    # If both are constant but different, use similarity based on ratio
                    else:
                        mean_i = np.mean(cluster_i)
                        mean_j = np.mean(cluster_j)
                        # Avoid division by zero
                        if mean_i == 0 or mean_j == 0:
                            correlations.append(0.0)
                        else:
                            ratio = min(mean_i, mean_j) / max(mean_i, mean_j)
                            correlations.append(ratio)  # Closer to 1 means more similar
                else:
                    # Calculate correlation
                    try:
                        corr = np.corrcoef(cluster_i, cluster_j)[0, 1]
                        # Handle NaN values
                        if np.isnan(corr):
                            correlations.append(0.0)
                        else:
                            correlations.append(corr)
                    except ValueError as e:
                        print(f"Error calculating correlation for Cluster {cluster} between Fold {i} and {j}: {e}")
                        print(f"Cluster {cluster} sizes: Fold {i}: {len(cluster_i)}, Fold {j}: {len(cluster_j)}")
                        correlations.append(0.0)
            
            # Use average correlation as similarity measure
            avg_correlation = np.mean(correlations)
            similarity_matrix[i-1, j-1] = avg_correlation
            similarity_matrix[j-1, i-1] = avg_correlation
    
    # Calculate similarity metrics
    upper_triangle = np.triu_indices(n_folds, k=1)
    similarity_scores = similarity_matrix[upper_triangle]
    
    mean_similarity = np.mean(similarity_scores)
    std_similarity = np.std(similarity_scores)
    
    return {
        'similarity_matrix': similarity_matrix,
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'all_scores': similarity_scores
    }

def visualize_fold_stability(stability_results):
    """Visualize stability between folds"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(stability_results['stability_matrix'], annot=True, cmap='viridis_r', ax=ax, fmt='.3f',
               cbar_kws={'label': 'Average Distance (lower is more stable)'})
    ax.set_title(f'Cross-Fold Stability\nMean: {stability_results["mean_stability"]:.4f} ± {stability_results["std_stability"]:.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Fold')
    
    plt.tight_layout()
    return fig

def visualize_cell_count_similarity(similarity_results):
    """Visualize cell count similarity between folds"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(similarity_results['similarity_matrix'], annot=True, cmap='viridis', ax=ax, fmt='.3f',
               cbar_kws={'label': 'Correlation (higher is more similar)'})
    ax.set_title(f'Cross-Fold Cell Count Similarity\nMean: {similarity_results["mean_similarity"]:.4f} ± {similarity_results["std_similarity"]:.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Fold')
    
    plt.tight_layout()
    return fig

def visualize_cluster_sizes(reassigned_counts_df):
    """Visualize relative cluster sizes across folds"""
    n_folds = reassigned_counts_df['Fold'].nunique()
    n_clusters = len([col for col in reassigned_counts_df.columns if col.startswith('Cluster')])
    
    # Calculate total cell count for each cluster in each fold
    fold_cluster_totals = []
    
    for fold in range(1, n_folds + 1):
        fold_data = reassigned_counts_df[reassigned_counts_df['Fold'] == fold]
        cluster_totals = []
        
        for cluster in range(n_clusters):
            total = fold_data[f'Cluster {cluster}'].sum()
            cluster_totals.append(total)
        
        # Convert to percentages
        total_cells = sum(cluster_totals)
        cluster_percentages = [100 * count / total_cells for count in cluster_totals]
        
        fold_cluster_totals.append({
            'Fold': fold,
            'Cluster Totals': cluster_totals,
            'Cluster Percentages': cluster_percentages
        })
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Absolute counts
    cluster_data = []
    for fold_data in fold_cluster_totals:
        for cluster, count in enumerate(fold_data['Cluster Totals']):
            cluster_data.append({
                'Fold': f'Fold {fold_data["Fold"]}',
                'Cluster': f'Cluster {cluster}',
                'Count': count
            })
    
    df_counts = pd.DataFrame(cluster_data)
    
    # Create the grouped bar chart for absolute counts
    sns.barplot(data=df_counts, x='Cluster', y='Count', hue='Fold', ax=ax1)
    ax1.set_title('Absolute Cell Counts by Cluster and Fold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Cell Count')
    
    # Percentage counts
    percentage_data = []
    for fold_data in fold_cluster_totals:
        for cluster, percentage in enumerate(fold_data['Cluster Percentages']):
            percentage_data.append({
                'Fold': f'Fold {fold_data["Fold"]}',
                'Cluster': f'Cluster {cluster}',
                'Percentage': percentage
            })
    
    df_percentages = pd.DataFrame(percentage_data)
    
    # Create the grouped bar chart for percentages
    sns.barplot(data=df_percentages, x='Cluster', y='Percentage', hue='Fold', ax=ax2)
    ax2.set_title('Relative Cell Counts by Cluster and Fold (%)')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Percentage of Cells (%)')
    
    # Remove duplicate legends
    handles, labels = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.legend(title='Fold', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def visualize_donor_distributions(reassigned_counts_df):
    """Visualize how cells from each donor are distributed across clusters"""
    n_folds = reassigned_counts_df['Fold'].nunique()
    n_clusters = len([col for col in reassigned_counts_df.columns if col.startswith('Cluster')])
    
    # Create a figure with subplots for each fold
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 10))
    if n_folds == 1:
        axes = [axes]
    
    # For each fold
    for fold_idx, fold in enumerate(range(1, n_folds + 1)):
        ax = axes[fold_idx]
        fold_data = reassigned_counts_df[reassigned_counts_df['Fold'] == fold]
        
        # Get donors for this fold
        fold_donors = fold_data['Donor ID'].unique()
        
        # Select a subset of donors if there are too many
        max_donors_to_show = min(10, len(fold_donors))
        donors_to_show = fold_donors[:max_donors_to_show]
        
        fold_data = fold_data[fold_data['Donor ID'].isin(donors_to_show)]
        
        # Reshape data for stacked bar chart
        donor_cluster_counts = []
        
        for donor in donors_to_show:
            # Find the donor rows
            donor_rows = fold_data[fold_data['Donor ID'] == donor]
            
            # Skip if no data for this donor
            if len(donor_rows) == 0:
                continue
                
            # Use the first row (should only be one per donor)
            donor_data = donor_rows.iloc[0]
            total_cells = sum([donor_data[f'Cluster {c}'] for c in range(n_clusters)])
            
            # Skip donors with no cells
            if total_cells == 0:
                continue
                
            for cluster in range(n_clusters):
                count = donor_data[f'Cluster {cluster}']
                percentage = 100 * count / total_cells
                
                donor_cluster_counts.append({
                    'Donor': donor,
                    'Cluster': f'Cluster {cluster}',
                    'Percentage': percentage
                })
        
        # Skip this fold if no valid data
        if len(donor_cluster_counts) == 0:
            ax.text(0.5, 0.5, f'No data for Fold {fold}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            continue
            
        df_donor_dist = pd.DataFrame(donor_cluster_counts)
        
        # Create a stacked bar chart
        try:
            donor_pivot = df_donor_dist.pivot(index='Donor', columns='Cluster', values='Percentage')
            donor_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            
            ax.set_title(f'Fold {fold} - Cell Distribution by Donor')
            ax.set_xlabel('Donor ID')
            ax.set_ylabel('Percentage of Cells (%)')
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust x-tick labels for readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        except ValueError as e:
            print(f"Error creating donor distribution plot for fold {fold}: {e}")
            ax.text(0.5, 0.5, f'Error plotting Fold {fold} data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def visualize_cluster_means(reassigned_means):
    """Visualize cluster means across folds"""
    n_folds = len(reassigned_means)
    n_clusters = reassigned_means[0].shape[0]
    n_features = reassigned_means[0].shape[1]
    
    # Create a figure with subplots for each cluster
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]
    
    # Different color for each fold
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
    feature_indices = range(n_features)
    
    for cluster_idx in range(n_clusters):
        ax = axes[cluster_idx]
        
        for fold_idx in range(n_folds):
            # Get this cluster's means for this fold
            cluster_means = reassigned_means[fold_idx][cluster_idx]
            
            # Plot the means as a line
            ax.plot(feature_indices, cluster_means, 'o-', 
                   color=colors[fold_idx], label=f'Fold {fold_idx+1}' if cluster_idx == 0 else "")
            
        ax.set_title(f'Cluster {cluster_idx} Mean Profiles')
        ax.set_ylabel('Value')
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add feature names to x-axis for the bottom subplot
        if cluster_idx == n_clusters - 1:
            ax.set_xlabel('Feature Index')
            ax.set_xticks(feature_indices)
        
        # Add legend to the first subplot only
        if cluster_idx == 0:
            ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def visualize_comprehensive_summary(reassigned_means, stability_results, similarity_results):
    """Create a comprehensive summary visualization without radar chart"""
    n_folds = len(reassigned_means)
    n_clusters = reassigned_means[0].shape[0]
    n_features = reassigned_means[0].shape[1]
    
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(2, 2, figure=fig)
    
    # Cluster Mean Profiles - Using parallel coordinates instead of radar
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot cluster means using parallel coordinates
    feature_indices = range(n_features)
    
    for cluster_idx in range(n_clusters):
        for fold_idx in range(n_folds):
            # Get this cluster's means for this fold
            cluster_means = reassigned_means[fold_idx][cluster_idx]
            
            # Calculate alpha based on fold number for visual distinction
            alpha = 0.3 + 0.7 * (fold_idx / (n_folds - 1)) if n_folds > 1 else 1.0
            
            # Plot the means as a line with unique color per cluster
            ax1.plot(feature_indices, cluster_means, 'o-', 
                   color=plt.cm.viridis(cluster_idx / max(1, n_clusters - 1)), 
                   alpha=alpha,
                   label=f'{cluster_idx}-{fold_idx+1}')
    
    # Customize plot
    ax1.set_title('Cluster Mean Profiles Across Folds')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Value')
    ax1.set_xticks(feature_indices)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend but keep it compact
    if n_clusters <= 6:  # Only show legend if not too cluttered
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper right', fontsize='small')
    
    # Stability Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(stability_results['stability_matrix'], annot=True, cmap='viridis_r', 
                ax=ax2, fmt='.3f', cbar_kws={'label': 'Distance'})
    ax2.set_title(f'Cluster Centroid Stability\nMean: {stability_results["mean_stability"]:.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Fold')
    
    # Cell Count Similarity Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    sns.heatmap(similarity_results['similarity_matrix'], annot=True, cmap='viridis', 
                ax=ax3, fmt='.3f', cbar_kws={'label': 'Correlation'})
    ax3.set_title(f'Cell Count Distribution Similarity\nMean: {similarity_results["mean_similarity"]:.4f}')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Fold')
    
    # Stability vs Similarity Scatter Plot
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Extract upper triangle values
    upper_triangle = np.triu_indices(n_folds, k=1)
    stability_values = stability_results['stability_matrix'][upper_triangle]
    similarity_values = similarity_results['similarity_matrix'][upper_triangle]
    
    # Create scatter plot
    scatter = ax4.scatter(stability_values, similarity_values, s=100, alpha=0.7,
                         c=np.arange(len(stability_values)), cmap='viridis')
    
    # Add labels for points
    for i, (x, y) in enumerate(zip(stability_values, similarity_values)):
        fold_i, fold_j = np.where(np.triu(np.ones((n_folds, n_folds)), k=1))
        ax4.annotate(f'F{fold_i[i]+1}-F{fold_j[i]+1}', 
                    (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax4.set_xlabel('Centroid Distance (lower is better)')
    ax4.set_ylabel('Cell Count Correlation (higher is better)')
    ax4.set_title('Stability vs Similarity Relationship')
    
    # Set axis limits with some padding
    if len(stability_values) > 0:  # Only if we have data points
        ax4.set_xlim(min(stability_values) * 0.9, max(stability_values) * 1.1)
        ax4.set_ylim(min(similarity_values) * 0.9, max(similarity_values) * 1.1)
    
    # Add grid
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Clustering Stability Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

def visualize_cluster_consistency(reassigned_counts_df):
    """Visualize cluster consistency in terms of relative sizes across folds"""
    n_folds = reassigned_counts_df['Fold'].nunique()
    n_clusters = len([col for col in reassigned_counts_df.columns if col.startswith('Cluster')])
    
    # Calculate percentage of cells in each cluster for each fold
    fold_percentages = []
    
    for fold in range(1, n_folds + 1):
        fold_data = reassigned_counts_df[reassigned_counts_df['Fold'] == fold]
        cluster_totals = []
        
        for cluster in range(n_clusters):
            cluster_totals.append(fold_data[f'Cluster {cluster}'].sum())
            
        total_cells = sum(cluster_totals)
        cluster_percentages = [100 * total / total_cells for total in cluster_totals]
        
        fold_percentages.append({
            'Fold': fold,
            'Percentages': cluster_percentages
        })
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Percentage of cells in each cluster by fold (stacked bars)
    data_for_stacked = []
    for fold_data in fold_percentages:
        for cluster, percentage in enumerate(fold_data['Percentages']):
            data_for_stacked.append({
                'Fold': f'Fold {fold_data["Fold"]}',
                'Cluster': f'Cluster {cluster}',
                'Percentage': percentage
            })
    
    df_stacked = pd.DataFrame(data_for_stacked)
    fold_pivot = df_stacked.pivot(index='Fold', columns='Cluster', values='Percentage')
    fold_pivot.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
    
    ax1.set_title('Cluster Size Distribution by Fold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Percentage of Cells (%)')
    ax1.legend(title='Cluster', loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Coefficient of variation across folds for each cluster
    cluster_percentages_across_folds = []
    for cluster in range(n_clusters):
        percentages = [fold_data['Percentages'][cluster] for fold_data in fold_percentages]
        mean_percentage = np.mean(percentages)
        std_percentage = np.std(percentages)
        cv = 100 * std_percentage / mean_percentage if mean_percentage > 0 else 0
        
        cluster_percentages_across_folds.append({
            'Cluster': f'Cluster {cluster}',
            'Mean Percentage': mean_percentage,
            'Std Percentage': std_percentage,
            'CV (%)': cv
        })
    
    df_cv = pd.DataFrame(cluster_percentages_across_folds)
    
    # Plot coefficient of variation as a bar chart
    bars = ax2.bar(df_cv['Cluster'], df_cv['CV (%)'], color=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom')
    
    ax2.set_title('Coefficient of Variation in Cluster Sizes Across Folds')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Coefficient of Variation (%)')
    ax2.set_ylim(0, max(df_cv['CV (%)']) * 1.2)  # Add some headroom for labels
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Clustering method selection")
    parser.add_argument(
            '--method', 
            type=str, 
            choices=['kmeans', 'em'],  # Specifies allowed choices for the method
            default='kmeans',  # Default method if none is provided
            help="Choose the clustering method (e.g., 'kmeans', 'em'). Default is 'kmeans'."
        )
    args = parser.parse_args()
    return args

def main():
    # Configuration
    args = parse_args()
    method = args.method
    base_path = method + '_cv_splits'  # Change this to the base directory containing cv_splits
    n_folds = 5

    # Store cluster means from all folds
    all_folds_means = []
    all_cell_counts_dfs = []
    
    # Process each fold - load final means and cell counts
    print("\nLoading data for each fold...")
    for fold_num in range(1, n_folds + 1):
        # Load the cell counts data
        print(f"Processing fold {fold_num}...")
        cell_counts_df = load_cell_counts(base_path, fold_num, method)
        print(f"Loaded cell counts for {len(cell_counts_df)} donors")
        all_cell_counts_dfs.append(cell_counts_df)
        
        # Load the final means
        fold_means_df = load_final_means(base_path, fold_num, method).T
        k = 4
        fold_means_df['cluster_id'] = np.tile([0, 1, 2, 3], len(fold_means_df) // 4 + 1)[:len(fold_means_df)]
        
        # Extract features (assuming all columns except cluster_id are features)
        feature_cols = [col for col in fold_means_df.columns if col != 'cluster_id']
        
        # Convert to numpy array, sorted by cluster_id
        fold_means = fold_means_df.sort_values('cluster_id')[feature_cols].values
        all_folds_means.append(fold_means)
    
    # Reassign cluster labels for consistency across folds
    print("\nReassigning cluster labels for consistency...")
    reassigned_means, reassignment_maps = reassign_clusters(all_folds_means)
    
    # Apply the reassignment to cell counts
    print("Reassigning cell counts based on cluster maps...")
    reassigned_counts_df = reassign_cell_counts(all_cell_counts_dfs, reassignment_maps)
    
    # Validate reassignment to ensure clusters have similar sizes across folds
    validate_reassignment(reassigned_counts_df)
    
    # Calculate stability metrics
    print("\nCalculating stability metrics...")
    stability_results = calculate_fold_stability(reassigned_means)
    print(f"Mean centroid stability: {stability_results['mean_stability']:.4f} ± {stability_results['std_stability']:.4f}")
    
    # Calculate cell count similarity metrics
    print("Calculating cell count similarity metrics...")
    similarity_results = calculate_cell_count_similarity(reassigned_counts_df)
    print(f"Mean cell count similarity: {similarity_results['mean_similarity']:.4f} ± {similarity_results['std_similarity']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12})
    
    # Fold stability visualization
    fig1 = visualize_fold_stability(stability_results)
    fig1.savefig(os.path.join(base_path,'fold_stability.png'), dpi=300, bbox_inches='tight')
    
    # Cell count similarity visualization
    fig2 = visualize_cell_count_similarity(similarity_results)
    fig2.savefig(os.path.join(base_path,'cell_count_similarity.png'), dpi=300, bbox_inches='tight')
    
    # Cluster size visualization
    fig3 = visualize_cluster_sizes(reassigned_counts_df)
    fig3.savefig(os.path.join(base_path,'cluster_sizes.png'), dpi=300, bbox_inches='tight')
    
    # Donor distribution visualization
    fig4 = visualize_donor_distributions(reassigned_counts_df)
    fig4.savefig(os.path.join(base_path,'donor_distributions.png'), dpi=300, bbox_inches='tight')
    
    # Cluster means visualization
    fig5 = visualize_cluster_means(reassigned_means)
    fig5.savefig(os.path.join(base_path,'cluster_means.png'), dpi=300, bbox_inches='tight')
    
    # Cluster consistency visualization
    fig6 = visualize_cluster_consistency(reassigned_counts_df)
    fig6.savefig(os.path.join(base_path,'cluster_consistency.png'), dpi=300, bbox_inches='tight')
    
    # New box plot visualizations
    fig7 = visualize_cluster_boxplots(reassigned_counts_df)
    fig7.savefig(os.path.join(base_path,'cluster_boxplots.png'), dpi=300, bbox_inches='tight')
    
    fig8 = visualize_cluster_boxplots_combined(reassigned_counts_df)
    fig8.savefig(os.path.join(base_path,'cluster_boxplots_combined.png'), dpi=300, bbox_inches='tight')
    
    # Comprehensive summary visualization
    fig9 = visualize_comprehensive_summary(reassigned_means, stability_results, similarity_results)
    fig9.savefig(os.path.join(base_path,'comprehensive_stability_summary.png'), dpi=300, bbox_inches='tight')
    
    print("\nVisualizations saved to disk. Done!")
    
    # Return summary results if needed
    results_summary = {
        'stability_results': stability_results,
        'similarity_results': similarity_results
    }
    
    return results_summary

if __name__ == "__main__":
    main()