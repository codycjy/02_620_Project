import sys
import scanpy as sc
import pandas as pd
import os


def process_h5ad(input_file, genes_file):
    # 读取top 2000基因列表并转换为整数
    with open(genes_file, "r") as f:
        top_genes = [int(line.strip()) for line in f]

    # 读取h5ad文件
    adata = sc.read_h5ad(input_file)

    # 只保留top 2000基因
    adata = adata[:, top_genes]

    # 构建输出文件名
    output_file = input_file.replace(".h5ad", "_2000.h5ad")

    # 保存文件，使用9级压缩
    adata.write_h5ad(output_file, compression="gzip", compression_opts=9)

    # 清理内存
    del adata


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process.py <input_h5ad_file> <genes_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    genes_file = sys.argv[2]
    process_h5ad(input_file, genes_file)
