# Cell Clustering Analysis Tool

## Overview

This package provides a computational framework for analyzing single-cell RNA sequencing data by clustering cells based on marker gene expression. It's specifically designed to identify cell population differences across samples from donors with different cognitive statuses (e.g., dementia vs. no dementia). The tool uses k-means clustering with a simulation-based approach to improve robustness.

## Features

- Load and process AnnData (h5ad) files containing single-cell RNA-seq data
- Filter data based on marker genes of interest
- Perform robust clustering through multiple sampling and simulation iterations
- Label datasets with cluster assignments
- Generate comprehensive cell count statistics and summaries
- Evaluate clustering quality and parameter selection
- Optionally perform stratified cross-validation splits by donor CASI scores
- Automatically iterate clustering over all cross-validation folds

## Installation

### Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - scanpy
  - numpy
  - pandas
  - scipy
  - hdf5plugin
  - tqdm
  - matplotlib
  - seaborn

## Usage

### Quick Start

The easiest way to run the tool is using the provided shell script with default parameters:

```bash
./run_clustering.sh --default
```

### Custom Parameters

For more control, you can specify custom parameters:

```bash
./run_clustering.sh \
    --train_file path/to/train.h5ad \
    --test_file path/to/test.h5ad \
    --marker_file path/to/markers.txt \
    --meta_file path/to/metadata.xlsx \
    --n_samples_per_donor 200 \
    --n_simulations 100 \
    --k 15
```

### Generate Cross-Validation Folds

To generate stratified cross-validation splits by donor CASI scores:

```bash
python main.py \
    --generate_cv_splits \
    --combined_file path/to/combined_dataset.h5ad \
    --meta_file path/to/metadata.csv
```

This creates `cv_splits/fold_1`, `fold_2`, ..., each with training and test `.h5ad` files.

### Run Clustering on All Folds

To run clustering on all folds after split generation:

```bash
for fold in cv_splits/fold_*; do
    python main.py \
        --fold_dir "$fold" \
        --marker_file marker_genes/marker_genes_set1.txt \
        --meta_file path/to/metadata.csv
    echo "Finished clustering for $fold"
done
```

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_file` | Path to training data (h5ad file) | Required unless `--fold_dir` used |
| `--test_file` | Path to test data (h5ad file) | Required unless `--fold_dir` used |
| `--marker_file` | Path to marker gene list file | Required |
| `--meta_file` | Path to metadata file with donor information | Required |
| `--n_samples_per_donor` | Number of cells to sample from each donor | 100 |
| `--n_simulations` | Number of sampling and clustering iterations | 10 |
| `--k` | Number of clusters for k-means algorithm | 10 |
| `--random_state` | Random seed for reproducibility | 42 |
| `--max_iterations` | Maximum number of iterations for k-means | 300 |
| `--train_output` | Output file for training data summary | training_cell_count_summary.xlsx |
| `--test_output` | Output file for test data summary | test_cell_count_summary.xlsx |
| `--generate_cv_splits` | Flag to generate CV folds only | False |
| `--combined_file` | Input for CV splitting (h5ad) | Required if `--generate_cv_splits` |
| `--n_folds` | Number of cross-validation folds | 5 |
| `--fold_dir` | Path to a specific fold directory | Optional |

### Marker Gene File Format

The marker gene file should be a text file with the following format:

```
/ Annotation of the marker gene set. Reference.
# Marker type name:
Gene1
Gene2
...

# Another marker type:
GeneA
GeneB
...
```

Lines beginning with `#` indicate marker categories, and lines beginning with `/` are treated as comments.

## Workflow

1. **Data Loading**: Load training and test datasets along with metadata
2. **Gene Filtering**: Filter datasets to focus on specified marker genes
3. **Multiple Simulations**: Run multiple iterations of sampling and clustering
4. **Final Clustering**: Perform final k-means clustering on the collected means
5. **Dataset Labeling**: Label both training and test datasets based on cluster assignments
6. **Output Generation**: Generate comprehensive summaries and statistics

## File Structure

- `run_clustering.sh`: Shell script to easily run the application
- `main.py`: Main entry point for the application
- `cluster_markers.py`: Core clustering functionality
- `process.py`: Data loading and preprocessing utilities
- `my_kmeans.py`: Custom k-means implementation
- `checker.py`: Tools for evaluating clustering results
- `marker_genes/`: Directory containing marker gene list files
- `cv_splits/`: Auto-generated cross-validation fold data

## Output Files

The tool generates several output files:

- **Main cell count summaries**: Excel/CSV files with cluster assignments and cell counts
- **Pivot tables**: Reorganized cell counts for easier visualization
- **Diagnostic information**: Information on cluster quality and distribution

## Example


Run through all folds:

```bash
python3 main.py \
  --generate_cv_splits \
  --meta_file ../../data/sea-ad_cohort_donor_metadata_072524.xlsx \
  --marker_file marker_genes/marker_genes_set1.txt
  
python3 main.py \    
  --fold_dir cv_splits \
  --marker_file marker_genes/marker_genes_set1.txt \
  --meta_file ../../data/sea-ad_cohort_donor_metadata_072524.xlsx \
  --combined_file ../../data/combined_corrected.h5ad

```

Currently implementing checkpointing mechanisms.

