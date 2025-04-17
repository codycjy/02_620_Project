#!/bin/bash
pip3 install hdf5plugin

# Define file paths
MARKER_FILE="./marker_genes/marker_genes_set1.txt"
META_FILE="../../data/sea-ad_cohort_donor_metadata_072524.xlsx"
COMBINED_FILE="../../data/combined_corrected.h5ad"

# Define clustering hyperparameters
N_SAMPLES_PER_DONOR=100
N_SIMULATIONS=10
K=10
RANDOM_STATE=42
MAX_ITERATIONS=300

# Step 1: Generate CV splits
echo "Generating CV splits..."
python3 main.py \
  --generate_cv_splits \
  --meta_file "$META_FILE" \
  --marker_file "$MARKER_FILE"\
  --n_folds 5 \

if [ $? -ne 0 ]; then
    echo "Error: CV split generation failed!"
    exit 1
fi

# Step 2: Run clustering across all folds
echo "Running clustering across all folds..."
python3 main.py \
  --fold_dir "cv_splits" \
  --marker_file "$MARKER_FILE" \
  --meta_file "$META_FILE" \
  --combined_file "$COMBINED_FILE" \
  --n_samples_per_donor "$N_SAMPLES_PER_DONOR" \
  --n_simulations "$N_SIMULATIONS" \
  --k "$K" \
  --random_state "$RANDOM_STATE" \
  --max_iterations "$MAX_ITERATIONS"

if [ $? -eq 0 ]; then
    echo "Clustering completed successfully!"
else
    echo "Error: Clustering failed!"
    exit 1
fi
