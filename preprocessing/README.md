
# Leveraging SEA-AD for Identifying Gene Signatures and Vulnerable Cell Populations in Alzheimer’s Disease

Alzheimer’s disease affects brain cell types differently, requiring cell-specific
research approaches. Using the Seattle Alzheimer’s Disease Brain Cell Atlas
(SEA-AD) single-cell transcriptomic data, we identified robust disease-associated gene signatures across major brain cell types. 

Our methodology combined (1) data preprocessing, (2) hierarchical cell clustering, and (3) cell type-specific regression analysis.



## Data
Our raw data can be downloaded from the SEA-AD portal: [SEA-AD Brain Cell Atlas Download](https://portal.brain-map.org/explore/seattle-alzheimers-disease/seattle-alzheimers-disease-brain-cell-atlas-download?edit&language=en).

The data we used include the donor meta data [Donor Meta Data Download](https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/b4/c7/b4c727e1-ede1-4c61-b2ee-bf1ae4a3ef68/sea-ad_cohort_donor_metadata_072524.xlsx) and processed 10x snRNAseq Data [snRNAseq Data Download](https://sea-ad-single-cell-profiling.s3.amazonaws.com/index.html#MTG/RNAseq/). It provides gene expression matrices from 1,240,908 nuclei derived from 84 aged donors representing the full AD severity spectrum, plus 5 younger neurotypical donors, by single nucleus RNA seq. 



## Preprocessing

Preprocessing the large dataset. Change the folder to `preprocessing` to run the code.

### High Variance Subset

run `python3 high_var_filter.py ` to get the index set of high variance genes.

### Apply the filter

Usage: `python process.py <input_h5ad_file> <genes_file>`
This will generate a new h5ad file with the high variance subset of genes.

# Data Preprocessing Scripts

This folder contains Python scripts and data files used for preprocessing and analyzing gene expression data, with a focus on examining the impact of post-mortem interval (PMI) and other confounding factors.

## Scripts
* **`confounding_factor_analysis.ipynb`**:
    * This script (`confounding_factor_analysis.ipynb`) analyzes how demographic factors (like sex, age, education) and race relate to whether someone has dementia.
    * It uses logistic regression to model these relationships.
* **`parameter_stability.ipynb`**:
    * This script (`parameter_stability.ipynb`) investigates how gene expression changes over time after death (post-mortem interval or PMI).
    * It uses linear regression to model this relationship and checks how stable the model is using cross-validation.
    * The stability of the model's parameters (slope and intercept) is assessed to find reliable genes.



## Data Files

### Input Data

* **`pseudobulk_data.csv`**: Contains gene expression data, with each row representing a donor and each column a gene.
* **`meta_extracted.csv`**: Contains metadata about the donors, including PMI.
* **`significant_genes.csv`**: A list of genes considered significant in the context of PMI analysis.

### Output Data

* **`linear_params_73genes.csv`**: Contains parameters (slope and intercept) from the linear regression models for each gene.
* **`cv_mse_73genes.csv`**: Contains cross-validation results (mean squared error) for each gene.
* **`cv_intercept_73genes.csv`**: Contains cross-validation results for the intercept of the linear regression models.
* **`cv_slope_73genes.csv`**: Contains cross-validation results for the slope of the linear regression models.

## Purpose

This folder provides scripts and data to:

* Analyze how gene expression is affected by the time after death.
* Examine how factors like demographics and race are related to dementia.

These analyses are important for understanding the complexities of gene expression in post-mortem studies and for identifying potential factors influencing cognitive decline.


### Apply the regress out on PMI factors

Run `python3 regress_out.py` to regress out the pmi factors from the data.
Please make sure to set the correct path for the input and output files in the script.



## Dual-Level Hierarchical Clustering

Run dual-level hierarchical cell clustering.

### Setup

To run the code, the data should be put under the folder `data`, which is under the same parent directory as current `code` folder.

1. Change the corresponding filename to get the code running properly.
2. Install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Coarse-Level Clustering

For coarse-level clustering, the results are contained in `kmeans_cv_splits` and `em_cv_splits` directories for simple analysis.

Run the following to directly implement the overall workflow:
```bash
cd coarse_clustering
bash run_clustering.sh
```

This will generate a `cv_splits` folder, ask you to select the cluster methods and store all corresponding files inside. 

#### Analysis & Visualization

`cell_count_analysis.py` and `coarse_analysis.py` contain the code to visualize the results.

Example usage:
```bash
python3 cell_count_analysis.py --method em
```

### Fine-Level Clustering

Fine clustering can be performed using the interactive `fine_cluster.ipynb` file. The results will be displayed interactively and saved in the designated folder.
