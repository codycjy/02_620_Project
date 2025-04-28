import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse


def merge_and_correct_data(input_dir, meta_file, pmi_effects_file, output_file):
    meta_df = pd.read_csv(meta_file)

    if "donor_id" not in meta_df.columns and "Donor ID" in meta_df.columns:
        meta_df = meta_df.rename(columns={"Donor ID": "donor_id"})

    pmi_effects_df = pd.read_csv(pmi_effects_file)
    pmi_effects_df = pmi_effects_df.set_index("Gene")

    pmi_dict = dict(zip(meta_df["donor_id"], meta_df["PMI"]))

    all_adatas = []

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isdir(filepath) or not filename.endswith(".h5ad"):
            continue

        try:
            adata = sc.read_h5ad(filepath)

            donor_id = None
            if "donor_id" in adata.obs.columns:
                donor_id = adata.obs["donor_id"][0]
            else:
                donor_id = filename.split("_")[0]

            if donor_id in pmi_dict:
                pmi_value = pmi_dict[donor_id]
                adata.obs["PMI"] = pmi_value

                for gene, row in pmi_effects_df.iterrows():
                    if gene in adata.var_names:
                        pmi_coef = row["PMI_Coefficient"]
                        gene_idx = adata.var_names.get_loc(gene)
                        if scipy.sparse.issparse(adata.X):
                            adata.X = adata.X.toarray()

                        adata.X[:, gene_idx] = adata.X[:, gene_idx] - (
                            pmi_coef * pmi_value
                        )

                all_adatas.append(adata)
                print(
                    f"Processed file {filename} with donor_id {donor_id} and PMI {pmi_value}."
                )
            else:
                print(f"Cannot find donor_id {donor_id} in metadata.")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    print(f"Combining {len(all_adatas)} datasets...")
    combined_adata = ad.concat(all_adatas, join="outer")

    print(f"Combined data shape: {combined_adata.shape}")
    combined_adata.write(output_file, compression="gzip")


if __name__ == "__main__":
    input_dir = "./donor_objects"
    meta_file = "correction/metadata.csv"
    pmi_effects_file = "correction/significant_genes.csv"
    output_file = "./combined_corrected_data.h5ad"

    merge_and_correct_data(input_dir, meta_file, pmi_effects_file, output_file)
