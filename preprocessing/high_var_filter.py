import scanpy as sc
import numpy as np
import scipy
from tqdm import tqdm

adata = sc.read_h5ad(
    "SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"
)


def calc_var_in_batches(adata, batch_size=5000):
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    gene_var = np.zeros(n_genes)

    mean = np.zeros(n_genes)
    for i in tqdm(range(0, n_cells, batch_size), desc="Computing mean"):
        batch = adata[i : min(i + batch_size, n_cells)].X
        if scipy.sparse.issparse(batch):
            batch = batch.toarray()
        mean += batch.sum(axis=0)
    mean /= n_cells

    for i in tqdm(range(0, n_cells, batch_size), desc="Computing variance"):
        batch = adata[i : min(i + batch_size, n_cells)].X
        if scipy.sparse.issparse(batch):
            batch = batch.toarray()
        gene_var += ((batch - mean) ** 2).sum(axis=0)
    gene_var /= n_cells - 1

    return gene_var


gene_var = calc_var_in_batches(adata)
top_genes_idx = np.argsort(gene_var)[-2000:]
adata = adata[:, top_genes_idx]


top_2000_idx = np.argsort(gene_var)[-2000:].tolist()
top_5000_idx = np.argsort(gene_var)[-5000:].tolist()

top_2000_var = gene_var[top_2000_idx].tolist()
top_5000_var = gene_var[top_5000_idx].tolist()

with open("top_2000_genes.txt", "w") as f:
    for idx, var in zip(top_2000_idx, top_2000_var):
        f.write(f"{idx}\t{var}\n")

with open("top_5000_genes.txt", "w") as f:
    for idx, var in zip(top_5000_idx, top_5000_var):
        f.write(f"{idx}\t{var}\n")
