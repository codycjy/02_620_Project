from collections import defaultdict
import scanpy as sc
import numpy as np
from tqdm import tqdm
import pandas as pd

def parse_marker_gene_file(file_path):
    marker_dict = defaultdict(list)
    current_marker_type = None

    with open(file_path, 'r') as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                # Extract marker type name from comment
                current_marker_type = stripped.lstrip("#").strip().replace(':','')
            elif stripped.startswith('/'):
                continue
            else:
                gene = stripped.split("\t")[0]
                if current_marker_type:
                    marker_dict[current_marker_type].append(gene)
    return dict(marker_dict)

def parse_meta_data(file_path):
    return pd.read_excel(file_path)

def parse_dataset(train_file_path,test_file_path ,meta_data):
    print("Loading training and testing data...")
    train_adata = sc.read_h5ad(train_file_path)
    test_adata = sc.read_h5ad(test_file_path)
    # Extract unique donor IDs from the combined dataset
    train_donor_ids = train_adata.obs['Donor ID'].unique()
    test_donor_ids = test_adata.obs['Donor ID'].unique()
    # Categorize donors based on cognitive status
    train_dementia_donors = train_dementia_donors = meta_data[
            (meta_data["Donor ID"].isin(train_donor_ids)) & 
            (meta_data["Cognitive Status"] == "Dementia")
        ]
    test_dementia_donors = test_dementia_donors = meta_data[
            (meta_data["Donor ID"].isin(test_donor_ids)) & 
            (meta_data["Cognitive Status"] == "Dementia")
        ]
    train_no_dementia_donors = train_no_dementia_donors = meta_data[
            (meta_data["Donor ID"].isin(train_donor_ids)) & 
            (meta_data["Cognitive Status"] == "No dementia")
        ]
    test_no_dementia_donors = test_no_dementia_donors = meta_data[
            (meta_data["Donor ID"].isin(test_donor_ids)) & 
            (meta_data["Cognitive Status"] == "No dementia")
        ]
    
    
    donor_groups = {
        'dementia': {
            'all': train_dementia_donors + test_dementia_donors,
            'training': train_dementia_donors,
            'testing': test_dementia_donors
        },
        'no_dementia': {
            'all': train_no_dementia_donors + test_no_dementia_donors,
            'training': train_no_dementia_donors,
            'testing': test_no_dementia_donors
        }
    }
    
    return train_adata, test_adata, donor_groups
