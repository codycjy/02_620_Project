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

<!-- 
### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cell-clustering-tool.git
   cd cell-clustering-tool
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
-->

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
Note that the files should be placed in the appropriate folder.

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_file` | Path to training data (h5ad file) | Required |
| `--test_file` | Path to test data (h5ad file) | Required |
| `--marker_file` | Path to marker gene list file | Required |
| `--meta_file` | Path to metadata file with donor information | Required |
| `--n_samples_per_donor` | Number of cells to sample from each donor | 100 |
| `--n_simulations` | Number of sampling and clustering iterations | 10 |
| `--k` | Number of clusters for k-means algorithm | 10 |
| `--random_state` | Random seed for reproducibility | 42 |
| `--max_iterations` | Maximum number of iterations for k-means | 300 |
| `--train_output` | Output file for training data summary | training_cell_count_summary.xlsx |
| `--test_output` | Output file for test data summary | test_cell_count_summary.xlsx |

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
- `my_kmeans`: Custom k-means implementation
- `checker.py`: Tools for evaluating clustering results
- `marker_genes/`: Directory containing marker gene list files

## Output Files

The tool generates several output files:

- **Main cell count summaries**: CSV files with cluster assignments and cell counts
- **Pivot tables**: Reorganized cell counts for easier visualization
- **Diagnostic information**: Information on cluster quality and distribution

## Example

```bash
./run_clustering.sh \
    --train_file data/combined_train.h5ad \
    --test_file data/combined_test.h5ad \
    --marker_file marker_genes/marker_genes_set1.txt \
    --meta_file data/sea-ad_cohort_donor_metadata_072524.xlsx \
    --n_samples_per_donor 200 \
    --n_simulations 50 \
    --k 7
```

Currently implementing checkpointing mechanisms.


<!-- 
## Methodology

### Robust K-Means Clustering

The tool implements a robust clustering approach:

1. **Multiple Sampling**: For each simulation, cells are sampled from each donor
2. **K-Means Clustering**: Each sampled dataset is clustered
3. **Cluster Means Collection**: Means from all simulations are collected
4. **Final Clustering**: A final k-means is performed on the collected means
5. **Label Assignment**: Cells are assigned to clusters based on similarity to final means

This approach helps mitigate the effects of donor-specific variation and improves the robustness of the clustering.

## Selecting Optimal K

The `checker.py` module includes functionality to help determine the optimal number of clusters:

```python
import checker
import numpy as np

# After running simulations and obtaining means
checker.check_cluster_numbers(means, final_means)
```

This will generate plots for:
- Within-Cluster Sum of Squares (WCSS) vs. K (Elbow Method)
- Silhouette Scores vs. K

## Performance Considerations

- For large datasets, consider using a lower number of simulations
- Memory usage increases with dataset size and number of clusters
- The tool automatically implements batch processing for large datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool was developed for analyzing single-cell RNA sequencing data in neurodegenerative disease research
- Parts of the methodology were inspired by existing approaches in single-cell clustering

## Contact

For questions or support, please contact [your email or contact information].
-->
