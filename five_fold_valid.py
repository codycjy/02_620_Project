import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV
from tqdm import tqdm 

# Create a Kernel Regression class that provides similar interface as the original SegmentedLinearRegression
class KernelRegressionModel:
    def __init__(self, kernel='rbf', alpha=1.0, gamma=None):
        """
        Initialize kernel regression model
        
        Parameters:
        kernel: Kernel type ('rbf', 'linear', 'polynomial', etc.)
        alpha: Regularization strength
        gamma: Kernel coefficient for rbf, polynomial and sigmoid kernels
        """
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
        
    def fit(self, X, y):
        """
        Fit kernel regression model
        X: Input features (PMI values)
        y: Target variable (gene expression)
        """
        # Ensure X and y are properly shaped numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Reshape to ensure proper dimensions
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Optimize hyperparameters with cross-validation if not provided
        if self.gamma is None:
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'gamma': [0.001, 0.01, 0.1, 1.0]
            }
            grid_search = GridSearchCV(
                KernelRidge(kernel=self.kernel),
                param_grid,
                cv=min(5, len(X)),
                scoring='neg_mean_squared_error'
            )
            try:
                grid_search.fit(X, y.ravel())
                self.alpha = grid_search.best_params_['alpha']
                self.gamma = grid_search.best_params_['gamma']
                self.model = grid_search.best_estimator_
            except Exception as e:
                # Fallback to default parameters if grid search fails
                print(f"Grid search failed, using default parameters: {e}")
                self.model = KernelRidge(kernel=self.kernel, alpha=self.alpha, gamma=self.gamma)
                self.model.fit(X, y.ravel())
        else:
            # Use provided parameters
            self.model.fit(X, y.ravel())
        
        return self
    
    def predict(self, X):
        """
        Predict using the kernel regression model
        """
        # Ensure X is a numpy array with proper shape
        X = np.asarray(X)
        X_orig_shape = X.shape
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        return self.model.predict(X)
    
    def get_coefficients(self):
        """
        Return model parameters
        Note: For kernel regression, we return hyperparameters instead of coefficients
        """
        # Return key model parameters - in kernel regression we don't have simple coefficients
        return [self.alpha, self.gamma]

# Function to use the kernel regression on your dataset
def run_kernel_regression_analysis(df, meta_df):
    print("Starting kernel regression analysis...")
 
    if isinstance(df.index, pd.Index) and not 'donor_id' in df.columns:

        df = df.reset_index()
        if df.columns[0] == 'index' or df.columns[0] == 'Unnamed: 0':
            df = df.rename(columns={df.columns[0]: 'donor_id'})
    
    if 'donor_id' not in meta_df.columns and 'Donor ID' in meta_df.columns:
        meta_df = meta_df.rename(columns={"Donor ID": "donor_id"})
    
    df = df.merge(meta_df[['donor_id', 'PMI']], on='donor_id', how='left')
    
    pmi_values = df['PMI'].values.reshape(-1, 1)
   
    gene_cols = [col for col in df.columns if col not in ['donor_id', 'PMI']]
    gene_expression = df[gene_cols].values
    
    valid_indices = ~np.isnan(pmi_values.flatten())
    gene_expression = gene_expression[valid_indices]
    pmi_values = pmi_values[valid_indices]
    
    print(f"Number of valid samples: {len(pmi_values)}")
    print(f"Number of genes: {gene_expression.shape[1]}")
    
    # Initialize lists to store the results
    alphas = []
    gammas = []
    residuals = []
    
    # Process each gene with kernel regression (with progress bar)
    print("Fitting kernel regression for each gene...")
    for gene_idx in tqdm(range(gene_expression.shape[1]), desc="Genes"):
        gene_values = gene_expression[:, gene_idx].reshape(-1, 1)
        
        try:
            # Fit kernel regression
            model = KernelRegressionModel()
            model.fit(pmi_values, gene_values)
            
            # Get model parameters
            alpha, gamma = model.get_coefficients()
            alphas.append(alpha)
            gammas.append(gamma)
            
            # Calculate residuals
            predicted = model.predict(pmi_values)
            residual = gene_values.flatten() - predicted
            residuals.append(residual)
        except Exception as e:
            print(f"Error processing gene {gene_idx}: {e}")
            # Add placeholder values in case of error
            alphas.append(1.0)
            gammas.append(0.1)
            residuals.append(np.zeros_like(pmi_values.flatten()))
    
    # Create dataframe with results
    gene_cols = [col for col in df.columns if col not in ['donor_id', 'PMI']]
    model_params_df = pd.DataFrame({
        'Gene': gene_cols,
        'Alpha': alphas,
        'Gamma': gammas
    })
    
    print("Sample of model parameters:")
    print(model_params_df.head())
    
    return df, pmi_values, gene_expression, model_params_df, residuals

# Function for 5-fold cross-validation analysis
def run_5fold_cv_analysis(pmi_values, gene_expression):
    print("\nPerforming 5-fold cross-validation analysis...")
    
    # Initialize storage for cross-validation results
    cv_alphas = []
    cv_gammas = []
    cv_mse = []
    
    # Set up 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform cross-validation for each gene
    for gene_idx in tqdm(range(gene_expression.shape[1]), desc="5-Fold CV Analysis"):
        gene_values = gene_expression[:, gene_idx].reshape(-1, 1)
        
        # Store results for current gene
        gene_cv_alphas = []
        gene_cv_gammas = []
        gene_cv_mse_values = []
        
        # Execute 5-fold cross-validation for current gene
        for train_idx, test_idx in kf.split(pmi_values):
            try:
                # Split into training and test sets
                X_train, X_test = pmi_values[train_idx], pmi_values[test_idx]
                y_train, y_test = gene_values[train_idx], gene_values[test_idx]
                
                # Fit kernel regression model
                model = KernelRegressionModel()
                model.fit(X_train, y_train)
                
                # Get model parameters
                alpha, gamma = model.get_coefficients()
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                mse = np.mean((y_test.flatten() - y_pred) ** 2)
                
                # Store results
                gene_cv_alphas.append(alpha)
                gene_cv_gammas.append(gamma)
                gene_cv_mse_values.append(mse)
                
            except Exception as e:
                print(f"Error in CV for gene {gene_idx}, fold {len(gene_cv_alphas)+1}: {e}")
                # Add placeholder values on error
                gene_cv_alphas.append(1.0)
                gene_cv_gammas.append(0.1)
                gene_cv_mse_values.append(float('inf'))
        
        # Store results for all folds of current gene
        cv_alphas.append(gene_cv_alphas)
        cv_gammas.append(gene_cv_gammas)
        cv_mse.append(gene_cv_mse_values)
    
    # Convert to DataFrames for analysis
    df_cv_alphas = pd.DataFrame(cv_alphas)
    df_cv_gammas = pd.DataFrame(cv_gammas)
    df_cv_mse = pd.DataFrame(cv_mse)
    
    # Set column names to fold numbers
    df_cv_alphas.columns = [f'Fold_{i+1}' for i in range(5)]
    df_cv_gammas.columns = [f'Fold_{i+1}' for i in range(5)]
    df_cv_mse.columns = [f'Fold_{i+1}' for i in range(5)]
    
    return df_cv_alphas, df_cv_gammas, df_cv_mse

# Analyze cross-validation stability
def analyze_cv_stability(df_cv):
    # Calculate various stability metrics
    stability_metrics = pd.DataFrame({
        'mean': df_cv.mean(axis=1),
        'std': df_cv.std(axis=1),
        'cv': df_cv.std(axis=1) / df_cv.mean(axis=1).abs(),  # Coefficient of variation (absolute value)
        'iqr': df_cv.quantile(0.75, axis=1) - df_cv.quantile(0.25, axis=1),  # Interquartile range
        'range': df_cv.max(axis=1) - df_cv.min(axis=1),  # Range between max and min
        'sign_consistency': df_cv.apply(lambda row: np.mean(np.sign(row) == np.sign(row.mean())), axis=1)  # Sign consistency
    })
    
    # Handle infinite or NaN values in CV (when mean is close to zero)
    stability_metrics['cv'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return stability_metrics

# Function to create visualizations and save results
def create_visualizations_and_save_results(
    df, pmi_values, gene_expression, model_params_df,
    df_cv_alphas, df_cv_gammas, df_cv_mse,
    stability_metrics_alphas, stability_metrics_gammas,
    cv_threshold=0.6, sign_threshold=0.9
):
    """Create visualizations and save results to files"""
    
    # Method 1: Based on coefficient of variation (CV)
    stable_genes_cv_alphas = stability_metrics_alphas[stability_metrics_alphas['cv'] <= cv_threshold]
    stable_genes_cv_gammas = stability_metrics_gammas[stability_metrics_gammas['cv'] <= cv_threshold]
    
    cv_stable_ratio_alphas = len(stable_genes_cv_alphas) / len(stability_metrics_alphas)
    cv_stable_ratio_gammas = len(stable_genes_cv_gammas) / len(stability_metrics_gammas)
    
    # Method 2: Based on sign consistency
    stable_genes_sign_alphas = stability_metrics_alphas[stability_metrics_alphas['sign_consistency'] >= sign_threshold]
    stable_genes_sign_gammas = stability_metrics_gammas[stability_metrics_gammas['sign_consistency'] >= sign_threshold]
    
    sign_stable_ratio_alphas = len(stable_genes_sign_alphas) / len(stability_metrics_alphas)
    sign_stable_ratio_gammas = len(stable_genes_sign_gammas) / len(stability_metrics_gammas)
    
    # Method 3: Combined criteria
    stable_genes_combined_alphas = stability_metrics_alphas[
        (stability_metrics_alphas['cv'] <= cv_threshold) & 
        (stability_metrics_alphas['sign_consistency'] >= sign_threshold)
    ]
    stable_genes_combined_gammas = stability_metrics_gammas[
        (stability_metrics_gammas['cv'] <= cv_threshold) & 
        (stability_metrics_gammas['sign_consistency'] >= sign_threshold)
    ]
    
    combined_stable_ratio_alphas = len(stable_genes_combined_alphas) / len(stability_metrics_alphas)
    combined_stable_ratio_gammas = len(stable_genes_combined_gammas) / len(stability_metrics_gammas)
    
    # Print stability statistics
    print(f"\nStable genes based on CV (CV ≤ {cv_threshold}):")
    print(f"  Alpha parameter: {len(stable_genes_cv_alphas)}, ratio: {cv_stable_ratio_alphas:.2%}")
    print(f"  Gamma parameter: {len(stable_genes_cv_gammas)}, ratio: {cv_stable_ratio_gammas:.2%}")
    
    print(f"\nStable genes based on sign consistency (≥ {sign_threshold}):")
    print(f"  Alpha parameter: {len(stable_genes_sign_alphas)}, ratio: {sign_stable_ratio_alphas:.2%}")
    print(f"  Gamma parameter: {len(stable_genes_sign_gammas)}, ratio: {sign_stable_ratio_gammas:.2%}")
    
    print(f"\nStable genes based on combined criteria:")
    print(f"  Alpha parameter: {len(stable_genes_combined_alphas)}, ratio: {combined_stable_ratio_alphas:.2%}")
    print(f"  Gamma parameter: {len(stable_genes_combined_gammas)}, ratio: {combined_stable_ratio_gammas:.2%}")
    
    # Plot distributions of stability metrics
    plt.figure(figsize=(15, 15))
    
    # 1. CV distribution for Alpha
    plt.subplot(3, 2, 1)
    sns.histplot(stability_metrics_alphas['cv'].dropna(), kde=True)
    plt.axvline(x=cv_threshold, color='r', linestyle='--')
    plt.title('CV Distribution (Alpha parameter)')
    plt.xlabel('Coefficient of Variation (CV)')
    plt.ylabel('Number of Genes')
    
    # 2. CV distribution for Gamma
    plt.subplot(3, 2, 2)
    sns.histplot(stability_metrics_gammas['cv'].dropna(), kde=True)
    plt.axvline(x=cv_threshold, color='r', linestyle='--')
    plt.title('CV Distribution (Gamma parameter)')
    plt.xlabel('Coefficient of Variation (CV)')
    plt.ylabel('Number of Genes')
    
    # 3. Sign consistency distribution for Alpha
    plt.subplot(3, 2, 3)
    sns.histplot(stability_metrics_alphas['sign_consistency'], kde=True)
    plt.axvline(x=sign_threshold, color='r', linestyle='--')
    plt.title('Sign Consistency Distribution (Alpha parameter)')
    plt.xlabel('Sign Consistency')
    plt.ylabel('Number of Genes')
    
    # 4. Sign consistency distribution for Gamma
    plt.subplot(3, 2, 4)
    sns.histplot(stability_metrics_gammas['sign_consistency'], kde=True)
    plt.axvline(x=sign_threshold, color='r', linestyle='--')
    plt.title('Sign Consistency Distribution (Gamma parameter)')
    plt.xlabel('Sign Consistency')
    plt.ylabel('Number of Genes')
    
    # 5. Scatter plot: CV vs sign consistency for Alpha
    plt.subplot(3, 2, 5)
    plt.scatter(stability_metrics_alphas['cv'], stability_metrics_alphas['sign_consistency'], alpha=0.5)
    plt.axhline(y=sign_threshold, color='r', linestyle='--')
    plt.axvline(x=cv_threshold, color='r', linestyle='--')
    plt.title('CV vs Sign Consistency (Alpha parameter)')
    plt.xlabel('Coefficient of Variation (CV)')
    plt.ylabel('Sign Consistency')
    
    # 6. Scatter plot: CV vs sign consistency for Gamma
    plt.subplot(3, 2, 6)
    plt.scatter(stability_metrics_gammas['cv'], stability_metrics_gammas['sign_consistency'], alpha=0.5)
    plt.axhline(y=sign_threshold, color='r', linestyle='--')
    plt.axvline(x=cv_threshold, color='r', linestyle='--')
    plt.title('CV vs Sign Consistency (Gamma parameter)')
    plt.xlabel('Coefficient of Variation (CV)')
    plt.ylabel('Sign Consistency')
    
    plt.tight_layout()
    plt.savefig('parameter_stability_analysis_5fold.png', facecolor='white')
    plt.close()
    
    # Visualize most stable and least stable genes
    try:
        # Sort by CV - Alpha parameter
        sorted_by_cv_alphas = stability_metrics_alphas.sort_values('cv')
        most_stable_alphas = sorted_by_cv_alphas.head(10).index
        least_stable_alphas = sorted_by_cv_alphas.tail(10).index
        
        # Sort by CV - Gamma parameter
        sorted_by_cv_gammas = stability_metrics_gammas.sort_values('cv')
        most_stable_gammas = sorted_by_cv_gammas.head(10).index
        least_stable_gammas = sorted_by_cv_gammas.tail(10).index
        
        # Create figure with 2 rows and 2 columns
        plt.figure(figsize=(16, 12))
        
        # Row 1: Alpha parameter
        plt.subplot(2, 2, 1)
        stable_melted_alphas = df_cv_alphas.loc[most_stable_alphas].reset_index().melt(
            id_vars='index', var_name='Fold', value_name='Alpha')
        stable_melted_alphas.rename(columns={'index': 'Gene_Index'}, inplace=True)
        sns.boxplot(x='Gene_Index', y='Alpha', data=stable_melted_alphas)
        plt.title('10 Most Stable Genes (Alpha parameter)')
        plt.xticks(rotation=90)
        
        plt.subplot(2, 2, 2)
        unstable_melted_alphas = df_cv_alphas.loc[least_stable_alphas].reset_index().melt(
            id_vars='index', var_name='Fold', value_name='Alpha')
        unstable_melted_alphas.rename(columns={'index': 'Gene_Index'}, inplace=True)
        sns.boxplot(x='Gene_Index', y='Alpha', data=unstable_melted_alphas)
        plt.title('10 Least Stable Genes (Alpha parameter)')
        plt.xticks(rotation=90)
        
        # Row 2: Gamma parameter
        plt.subplot(2, 2, 3)
        stable_melted_gammas = df_cv_gammas.loc[most_stable_gammas].reset_index().melt(
            id_vars='index', var_name='Fold', value_name='Gamma')
        stable_melted_gammas.rename(columns={'index': 'Gene_Index'}, inplace=True)
        sns.boxplot(x='Gene_Index', y='Gamma', data=stable_melted_gammas)
        plt.title('10 Most Stable Genes (Gamma parameter)')
        plt.xticks(rotation=90)
        
        plt.subplot(2, 2, 4)
        unstable_melted_gammas = df_cv_gammas.loc[least_stable_gammas].reset_index().melt(
            id_vars='index', var_name='Fold', value_name='Gamma')
        unstable_melted_gammas.rename(columns={'index': 'Gene_Index'}, inplace=True)
        sns.boxplot(x='Gene_Index', y='Gamma', data=unstable_melted_gammas)
        plt.title('10 Least Stable Genes (Gamma parameter)')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig('stable_vs_unstable_genes_5fold.png')
        plt.close()
        
    except Exception as e:
        print(f"Error creating stability gene charts: {e}")
    
    # Export lists of stable genes
    stable_genes_alphas = stable_genes_combined_alphas.index.tolist()
    stable_genes_gammas = stable_genes_combined_gammas.index.tolist()
    
    print(f"\nTop 10 most stable gene indices (Alpha parameter): {stable_genes_alphas[:10]}")
    print(f"Top 10 most stable gene indices (Gamma parameter): {stable_genes_gammas[:10]}")
    
    # Save stability metrics to CSV files
    stability_metrics_alphas.sort_values('cv').to_csv('alpha_stability_metrics_5fold.csv')
    stability_metrics_gammas.sort_values('cv').to_csv('gamma_stability_metrics_5fold.csv')
    
    # Create MSE analysis
    plt.figure(figsize=(10, 6))
    
    # Sort by mean MSE
    mean_mse = df_cv_mse.mean(axis=1)
    sorted_indices = np.argsort(mean_mse)
    
    # Select top 20 and bottom 20 genes by MSE for visualization
    best_genes = sorted_indices[:20]
    worst_genes = sorted_indices[-20:]
    
    # Box plots showing MSE distribution
    plt.subplot(1, 2, 1)
    best_melted = df_cv_mse.iloc[best_genes].reset_index().melt(
        id_vars='index', var_name='Fold', value_name='MSE')
    sns.boxplot(data=best_melted, x='index', y='MSE')
    plt.title('20 Genes with Lowest MSE')
    plt.xticks(rotation=90)
    plt.xlabel('Gene Index')
    
    plt.subplot(1, 2, 2)
    worst_melted = df_cv_mse.iloc[worst_genes].reset_index().melt(
        id_vars='index', var_name='Fold', value_name='MSE')
    sns.boxplot(data=worst_melted, x='index', y='MSE')
    plt.title('20 Genes with Highest MSE')
    plt.xticks(rotation=90)
    plt.xlabel('Gene Index')
    
    plt.tight_layout()
    plt.savefig('model_mse_5fold.png',facecolor='white')
    plt.close()
    
    # Visualize kernel regression fits for specific genes
    try:
        # Select a few genes to visualize
        num_genes = min(5, gene_expression.shape[1])
        genes_to_visualize = np.linspace(0, gene_expression.shape[1]-1, num_genes, dtype=int)
        
        for gene_idx in genes_to_visualize:
            plt.figure(figsize=(15, 10))
            gene_values = gene_expression[:, gene_idx]
            
            # Get current gene's 5-fold CV parameters
            cv_alphas = df_cv_alphas.iloc[gene_idx].values
            cv_gammas = df_cv_gammas.iloc[gene_idx].values
            
            # Scatter plot of original data
            plt.scatter(pmi_values, gene_values, alpha=0.6, label='Data Points')
            
            # Plot regression curves for each fold
            for fold_idx in range(5):
                # Create a fine grid for prediction
                x_grid = np.linspace(min(pmi_values), max(pmi_values), 100).reshape(-1, 1)
                
                # Create model with parameters from this fold
                model = KernelRidge(kernel='rbf', alpha=cv_alphas[fold_idx], gamma=cv_gammas[fold_idx])
                
                try:
                    # Fit on all data (simplification for visualization)
                    model.fit(pmi_values, gene_values)
                    
                    # Make predictions
                    y_pred = model.predict(x_grid)
                    
                    # Plot this fold's regression curve
                    plt.plot(x_grid, y_pred, linestyle='--', alpha=0.7,
                            label=f'Fold {fold_idx+1} (Alpha: {cv_alphas[fold_idx]:.3f}, Gamma: {cv_gammas[fold_idx]:.3f})')
                except Exception as e:
                    print(f"Error plotting fold {fold_idx} for gene {gene_idx}: {e}")
            
            # Fit full data kernel regression model
            try:
                full_model = KernelRegressionModel()
                full_model.fit(pmi_values, gene_values)
                
                # Create a fine grid for prediction
                x_grid = np.linspace(min(pmi_values), max(pmi_values), 100).reshape(-1, 1)
                y_pred = full_model.predict(x_grid)
                
                # Plot full data fit
                plt.plot(x_grid, y_pred, 'r-', linewidth=2,
                        label=f'Full Data Fit (Alpha: {full_model.alpha:.3f}, Gamma: {full_model.gamma:.3f})')
            except Exception as e:
                print(f"Error plotting full model for gene {gene_idx}: {e}")
            
            plt.title(f'Gene: {df.columns[1+gene_idx]} 5-Fold Cross-Validation Kernel Regression')
            plt.xlabel('PMI')
            plt.ylabel('Gene Expression')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'gene_{gene_idx}_5fold_cv.png',facecolor='white')
            plt.close()
            
        # Plot average MSE comparison between folds
        plt.figure(figsize=(10, 6))
        fold_mean_mse = df_cv_mse.mean(axis=0)
        plt.bar(range(5), fold_mean_mse)
        plt.title('Average MSE by Fold')
        plt.xlabel('Fold')
        plt.ylabel('Average MSE')
        plt.xticks(range(5), [f'Fold {i+1}' for i in range(5)])
        plt.tight_layout()
        plt.savefig('fold_comparison_mse.png', facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"Error creating regression example plots: {e}")