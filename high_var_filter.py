import scanpy as sc
import numpy as np
import scipy
from tqdm import tqdm

adata = sc.read_h5ad(
    "/kaggle/input/seaad-mtg-rnaseq-final-nuclei/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"
)


def calc_var_in_batches(adata, batch_size=5000):
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    gene_var = np.zeros(n_genes)

    # 分批计算均值
    mean = np.zeros(n_genes)
    for i in tqdm(range(0, n_cells, batch_size), desc="Computing mean"):
        batch = adata[i : min(i + batch_size, n_cells)].X
        if scipy.sparse.issparse(batch):
            batch = batch.toarray()
        mean += batch.sum(axis=0)
    mean /= n_cells

    # 分批计算方差
    for i in tqdm(range(0, n_cells, batch_size), desc="Computing variance"):
        batch = adata[i : min(i + batch_size, n_cells)].X
        if scipy.sparse.issparse(batch):
            batch = batch.toarray()
        gene_var += ((batch - mean) ** 2).sum(axis=0)
    gene_var /= n_cells - 1

    return gene_var


# 使用分批计算
gene_var = calc_var_in_batches(adata)
top_genes_idx = np.argsort(gene_var)[-2000:]
adata = adata[:, top_genes_idx]


# 获取 top 2000 和 5000 的基因索引
top_2000_idx = np.argsort(gene_var)[-2000:].tolist()
top_5000_idx = np.argsort(gene_var)[-5000:].tolist()

# 保存对应的variance值
top_2000_var = gene_var[top_2000_idx].tolist()
top_5000_var = gene_var[top_5000_idx].tolist()

# 将索引和variance写入txt文件
with open("top_2000_genes.txt", "w") as f:
    for idx, var in zip(top_2000_idx, top_2000_var):
        f.write(f"{idx}\t{var}\n")

with open("top_5000_genes.txt", "w") as f:
    for idx, var in zip(top_5000_idx, top_5000_var):
        f.write(f"{idx}\t{var}\n")
