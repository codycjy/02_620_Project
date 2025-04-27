# Data Preprocessing Scripts

This folder contains Python scripts and data files used for preprocessing and analyzing gene expression data, with a focus on examining the impact of post-mortem interval (PMI) and other confounding factors.

## Scripts

* **`parameter_stability.ipynb`**:
    * This script (`parameter_stability.ipynb`) investigates how gene expression changes over time after death (post-mortem interval or PMI).
    * It uses linear regression to model this relationship and checks how stable the model is using cross-validation.
    * The stability of the model's parameters (slope and intercept) is assessed to find reliable genes.

* **`confounding_factor_analysis.ipynb`**:
    * This script (`confounding_factor_analysis.ipynb`) analyzes how demographic factors (like sex, age, education) and race relate to whether someone has dementia.
    * It uses logistic regression to model these relationships.

## Data Files

### Input Data

* **`pseudobulk_data.csv`**: Contains gene expression data, with each row representing a donor and each column a gene[cite: 2].
* **`meta_extracted.csv`**: Contains metadata about the donors, including PMI[cite: 2].
* **`significant_genes.csv`**: A list of genes considered significant in the context of PMI analysis.

### Output Data

* **`linear_params_73genes.csv`**: Contains parameters (slope and intercept) from the linear regression models for each gene[cite: 2].
* **`cv_mse_73genes.csv`**: Contains cross-validation results (mean squared error) for each gene[cite: 3].
* **`cv_intercept_73genes.csv`**: Contains cross-validation results for the intercept of the linear regression models[cite: 4].
* **`cv_slope_73genes.csv`**: Contains cross-validation results for the slope of the linear regression models[cite: 5].

## Purpose

This folder provides scripts and data to:

* Analyze how gene expression is affected by the time after death.
* Examine how factors like demographics and race are related to dementia.

These analyses are important for understanding the complexities of gene expression in post-mortem studies and for identifying potential factors influencing cognitive decline.
