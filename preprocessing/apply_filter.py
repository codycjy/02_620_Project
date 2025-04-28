import sys
import scanpy as sc
import pandas as pd
import os


def process_h5ad(input_file, genes_file):
    with open(genes_file, "r") as f:
        top_genes = [int(line.strip()) for line in f]

    adata = sc.read_h5ad(input_file)

    adata = adata[:, top_genes]

    output_file = input_file.replace(".h5ad", "_2000.h5ad")

    adata.write_h5ad(output_file, compression="gzip", compression_opts=9)

    del adata


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process.py <input_h5ad_file> <genes_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    genes_file = sys.argv[2]
    process_h5ad(input_file, genes_file)
