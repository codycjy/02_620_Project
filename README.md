# Dual-Level Hierarchical Clustering

This folder contains the code to run dual-level hierarchical clustering.

## Setup

To run the code, the data should be put under the folder `data`, which is under the same parent directory as current `code` folder.

1. Change the corresponding filename to get the code running properly.
2. Install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Coarse-Level Clustering

For coarse-level clustering, the results are contained in `kmeans_cv_splits` and `em_cv_splits` directories for simple analysis.

Run the following to directly implement the overall workflow:
```bash
cd coarse_clustering
bash run_clustering.sh
```

This will generate a `cv_splits` folder, ask you to select the cluster methods and store all corresponding files inside. 

## Analysis & Visualization

`cell_count_analysis.py` and `coarse_analysis.py` contain the code to visualize the results.

Example usage:
```bash
python3 cell_count_analysis.py --method em
```