#!/bin/bash
pip3 install hdf5plugin
# Shell script to run the cell clustering Python script
# Choose the marker_file in the `marker_genes` folder
MARKER_FILE="./marker_genes/marker_genes_set1.txt"
# Define default file paths
# change the path accordingly
TRAIN_FILE="../../data/combined_train.h5ad"
TEST_FILE="../../data/combined_test.h5ad"
META_FILE="../../data/sea-ad_cohort_donor_metadata_072524.xlsx"

# Check if we're running with default parameters or if arguments are provided
if [ "$1" == "--default" ]; then
    echo "Running with default parameters..."
    python3 main.py \
        --train_file $TRAIN_FILE \
        --test_file $TEST_FILE \
        --marker_file $MARKER_FILE \
        --meta_file $META_FILE\
    
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "  ./run_clustering.sh --default               # Run with default parameters"
    echo "  ./run_clustering.sh [custom parameters]     # Run with custom parameters"
    echo "  ./run_clustering.sh --help                  # Show this help message"
    echo ""
    echo "Example with custom parameters:"
    echo "  ./run_clustering.sh \\"
    echo "    --train_file path/to/train.h5ad \\"
    echo "    --test_file path/to/test.h5ad \\"
    echo "    --marker_file path/to/markers.txt \\"
    echo "    --meta_file path/to/metadata.csv \\"
    echo "    --n_samples_per_donor 200 \\"
    echo "    --n_simulations 100 \\"
    echo "    --k 15"
    
else
    # Pass all arguments to the Python script
    echo "Running with custom parameters..."
    python3 main.py "$@"
fi

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Clustering completed successfully!"
else
    echo "Error: Clustering script failed with exit code $?"
    exit 1
fi