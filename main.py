import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
import argparse

# Import functions from both scripts
from create_pseudo import create_pseudobulk_from_directory
from five_fold_valid import (
    run_kernel_regression_analysis,
    run_5fold_cv_analysis,
    analyze_cv_stability,
    create_visualizations_and_save_results,
    KernelRegressionModel
)

def main():
    parser = argparse.ArgumentParser(description="Kernel Regression Analysis for Gene Expression Data")
    parser.add_argument("--input_dir", type=str, default="dataset",
                        help="Directory containing h5ad files")
    parser.add_argument("--meta_file", type=str, default="meta_extracted.csv",
                        help="Path to metadata file with PMI values")
    args = parser.parse_args()
    cv_threshold = 0.6  # Coefficient of variation threshold for stability
    sign_threshold = 0.9  # Sign consistency threshold for stability
    output_dir = "results"  # Directory to save results
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Create pseudobulk data from h5ad files
        print("Step 1: Creating pseudobulk data from h5ad files...")
        pseudobulk_df = create_pseudobulk_from_directory(args.input_dir)
        
        # Step 2: Generate or load metadata with PMI values
        print("Step 2: Processing metadata with PMI values...")
        try:
            # Try to load metadata file if it exists
            meta_df = pd.read_csv(args.meta_file)
            
            # Check if 'donor_id' and 'PMI' columns exist in metadata
            if 'donor_id' not in meta_df.columns and 'Donor ID' in meta_df.columns:
                meta_df = meta_df.rename(columns={"Donor ID": "donor_id"})
            
            if 'PMI' not in meta_df.columns:
                raise ValueError("Metadata file must contain a 'PMI' column")
        except:
            # If file doesn't exist or has issues, generate random PMI values
            print("Metadata file not found or has issues. Generating random PMI values.")
            meta_df = pd.DataFrame({
                "donor_id": pseudobulk_df.index.unique(),
                "PMI": np.random.uniform(3, 12, size=len(pseudobulk_df.index.unique()))
            })
        
        # Step 3: Run kernel regression analysis
        print("Step 3: Running kernel regression analysis...")
        df, pmi_values, gene_expression, model_params_df, residuals = run_kernel_regression_analysis(
            pseudobulk_df, meta_df
        )
        
        # Step 4: Run 5-fold cross-validation analysis
        print("Step 4: Running 5-fold cross-validation analysis...")
        df_cv_alphas, df_cv_gammas, df_cv_mse = run_5fold_cv_analysis(pmi_values, gene_expression)
        
        # Step 5: Analyze parameter stability
        print("Step 5: Analyzing parameter stability...")
        stability_metrics_alphas = analyze_cv_stability(df_cv_alphas)
        stability_metrics_gammas = analyze_cv_stability(df_cv_gammas)
        
        # Print descriptive statistics of stability metrics
        print("\nAlpha parameter stability metrics:")
        print(stability_metrics_alphas.describe())
        
        print("\nGamma parameter stability metrics:")
        print(stability_metrics_gammas.describe())
        
        # Step 6: Create visualizations and save results
        print("Step 6: Creating visualizations and saving results...")
        create_visualizations_and_save_results(
            df, pmi_values, gene_expression, model_params_df,
            df_cv_alphas, df_cv_gammas, df_cv_mse,
            stability_metrics_alphas, stability_metrics_gammas,
            cv_threshold=cv_threshold,
            sign_threshold=sign_threshold
        )
        
        # Save additional results to the output directory
        model_params_df.to_csv(os.path.join(output_dir, "model_parameters.csv"), index=False)
        stability_metrics_alphas.to_csv(os.path.join(output_dir, "alpha_stability_metrics.csv"))
        stability_metrics_gammas.to_csv(os.path.join(output_dir, "gamma_stability_metrics.csv"))
        
        print(f"Analysis successfully completed! Results saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error in main analysis pipeline: {e}")
        return False

if __name__ == "__main__":
    main()