# Preprocessing
Preprocessing the large dataset.
## High Variance Subset
run `python3 high_var_filter.py ` to get the index set of high variance genes.
## Apply the filter
Usage: `python process.py <input_h5ad_file> <genes_file>`
This will generate a new h5ad file with the high variance subset of genes.

## Apply the regress out on PMI factors
Run `python3 regress_out.py` to regress out the pmi factors from the data.
Please make sure to set the correct path for the input and output files in the script.


