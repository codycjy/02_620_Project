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

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import StratifiedKFold
import shutil

def stratified_split_by_donor(meta_data, casi_col='CASI', n_folds=5, output_dir='cv_splits', random_state=42):
    """
    Perform stratified donor split based only on metadata (no expression data needed).
    """
    print(f"Starting stratified {n_folds}-fold cross-validation split by donor CASI scores...")

    os.makedirs(output_dir, exist_ok=True)

    # Check if CASI column exists
    if casi_col not in meta_data.columns:
        raise ValueError(f"{casi_col} column not found in meta_data. Available columns: {meta_data.columns.tolist()}")

    # Remove missing CASI
    donor_meta = meta_data.dropna(subset=[casi_col]).copy()
    missing_casi = meta_data[meta_data[casi_col].isna()]
    if len(missing_casi) > 0:
        print(f"Warning: {len(missing_casi)} donors have missing CASI scores and will be excluded from stratification.")
        print(f"Excluded donors: {missing_casi['Donor ID'].tolist()}")

    donor_meta['CASI_bin'] = pd.qcut(donor_meta[casi_col], q=5, labels=False, duplicates='drop')
    has_cognitive_status = 'Cognitive Status' in donor_meta.columns

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_splits = list(skf.split(donor_meta['Donor ID'], donor_meta['CASI_bin']))

    fold_summary = pd.DataFrame()

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        fold_num = fold_idx + 1
        print(f"Processing fold {fold_num}/{n_folds}")

        fold_dir = os.path.join(output_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        train_donors = donor_meta.iloc[train_idx]['Donor ID'].tolist()
        test_donors = donor_meta.iloc[test_idx]['Donor ID'].tolist()

        train_info = donor_meta.iloc[train_idx].copy()
        test_info = donor_meta.iloc[test_idx].copy()
        train_info['fold'] = fold_num
        test_info['fold'] = fold_num
        train_info['set'] = 'train'
        test_info['set'] = 'test'

        fold_donor_info = pd.concat([train_info, test_info])
        fold_summary = pd.concat([fold_summary, fold_donor_info])

        # Report stats
        train_casi_mean = train_info[casi_col].mean()
        test_casi_mean = test_info[casi_col].mean()
        train_donor_count = len(train_donors)
        test_donor_count = len(test_donors)

        cog_status_info = {}
        if has_cognitive_status:
            train_dementia = train_info[train_info['Cognitive Status'] == 'Dementia'].shape[0]
            train_no_dementia = train_info[train_info['Cognitive Status'] == 'No dementia'].shape[0]
            test_dementia = test_info[test_info['Cognitive Status'] == 'Dementia'].shape[0]
            test_no_dementia = test_info[test_info['Cognitive Status'] == 'No dementia'].shape[0]

            cog_status_info.update({
                'train_dementia': train_dementia,
                'train_no_dementia': train_no_dementia,
                'test_dementia': test_dementia,
                'test_no_dementia': test_no_dementia
            })

        # Save fold summary
        fold_info = {
            'fold': fold_num,
            'train_donors': train_donors,
            'test_donors': test_donors,
            'train_donor_count': train_donor_count,
            'test_donor_count': test_donor_count,
            'train_casi_mean': train_casi_mean,
            'test_casi_mean': test_casi_mean,
            **cog_status_info
        }

        with open(os.path.join(fold_dir, 'fold_info.json'), 'w') as f:
            import json
            json.dump(fold_info, f, indent=2)

        train_info.to_csv(os.path.join(fold_dir, 'train_donors.csv'), index=False)
        test_info.to_csv(os.path.join(fold_dir, 'test_donors.csv'), index=False)

        print(f"Fold {fold_num} split completed:")
        print(f"  Train: {train_donor_count} donors, CASI mean: {train_casi_mean:.2f}")
        print(f"  Test: {test_donor_count} donors, CASI mean: {test_casi_mean:.2f}")
        if has_cognitive_status:
            print(f"  Train: {train_dementia} dementia, {train_no_dementia} no dementia")
            print(f"  Test:  {test_dementia} dementia, {test_no_dementia} no dementia")

    fold_summary.to_csv(os.path.join(output_dir, 'all_folds_summary.csv'), index=False)
    print(f"All {n_folds} folds created and saved to {output_dir}")
    return fold_summary


def process_and_split_dataset(meta_data_path, output_dir='cv_splits', n_folds=5, random_state=42):
    print("Loading donor metadata for CV split...")
    meta_data = pd.read_excel(meta_data_path)
    return stratified_split_by_donor(meta_data, casi_col='Last CASI Score', n_folds=n_folds,
                                     output_dir=output_dir, random_state=random_state)