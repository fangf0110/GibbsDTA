import os
import json
import pickle
import numpy as np
import random
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction.text import CountVectorizer

def generate_stratified_folds(dataset_name, data_path='data', seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    Y_path = os.path.join(data_path, dataset_name, 'Y')
    ligands_path = os.path.join(data_path, dataset_name, 'ligands_can.txt')
    proteins_path = os.path.join(data_path, dataset_name, 'proteins.txt')
    
    if not os.path.exists(Y_path):
        print(f"Dataset {dataset_name} not found.")
        return
        
    with open(Y_path, 'rb') as f:
        Y = pickle.load(f, encoding='latin1')
        
    with open(ligands_path, 'r') as f:
        ligands_dict = json.load(f)
        
    with open(proteins_path, 'r') as f:
        proteins_dict = json.load(f)
        
    smiles_list = list(ligands_dict.values())
    proteins_list = list(proteins_dict.values())
    
    num_drugs, num_targets = Y.shape
    assert len(smiles_list) == num_drugs, "Mismatch between Y rows and ligands count"
    assert len(proteins_list) == num_targets, "Mismatch between Y cols and proteins count"
    
    valid_rows, valid_cols = np.where(~np.isnan(Y))
    all_interactions = []
    for r, c in zip(valid_rows, valid_cols):
        all_interactions.append((r, c, int(r * num_targets + c)))
        
    folds_dir = os.path.join(data_path, dataset_name, 'folds')
    os.makedirs(folds_dir, exist_ok=True)
    
    print(f"\n[{dataset_name}] Generating Stratified CV splits for all Settings...")
    
    # ---------------------------------------------------------
    # 1. DRUG CLUSTERING (Morgan Fingerprints)
    # ---------------------------------------------------------
    print("Computing Morgan Fingerprints for Drugs...")
    fps = []
    valid_smiles_indices = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            arr = np.zeros((1,), dtype=int)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_smiles_indices.append(idx)
        else:
            print(f"Warning: RDKit could not parse SMILES at index {idx}")
            
    X_fps = np.zeros((num_drugs, 1024))
    for i, arr in zip(valid_smiles_indices, fps):
        X_fps[i] = arr
        
    n_drug_clusters = 8 if dataset_name == 'davis' else 50
    cluster_model_drug = AgglomerativeClustering(n_clusters=n_drug_clusters, metric='euclidean', linkage='ward')
    drug_cluster_labels = cluster_model_drug.fit_predict(X_fps)
    
    # ---------------------------------------------------------
    # 2. TARGET CLUSTERING (3-mer Sequence Frequency)
    # ---------------------------------------------------------
    print("Computing 3-mer Frequencies for Targets...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X_prot = vectorizer.fit_transform(proteins_list).toarray()
    
    n_target_clusters = 50 if dataset_name == 'davis' else 30
    cluster_model_prot = AgglomerativeClustering(n_clusters=n_target_clusters, metric='euclidean', linkage='ward')
    target_cluster_labels = cluster_model_prot.fit_predict(X_prot)
    
    # ---------------------------------------------------------
    # 3. GENERATE DRUG & TARGET FOLDS
    # ---------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    drug_indices = np.arange(num_drugs)
    target_indices = np.arange(num_targets)
    
    drug_folds = list(skf.split(drug_indices, drug_cluster_labels))
    target_folds = list(skf.split(target_indices, target_cluster_labels))
    
    # ---------------------------------------------------------
    # 4. BUILD SETTINGS 1 TO 4
    # ---------------------------------------------------------
    cv_s1, cv_s2, cv_s3, cv_s4 = [], [], [], []
    
    # Setting 1 (Warm Start): Random Split of all valid interactions
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    all_ints_idx = np.arange(len(all_interactions))
    s1_folds = list(kf.split(all_ints_idx))
    
    for fold_idx in range(5):
        # Retrieve drug and target folds
        drug_train_val_idx, drug_test_idx = drug_folds[fold_idx]
        target_train_val_idx, target_test_idx = target_folds[fold_idx]
        
        # Sub-split train_val into train and val (using 1/8 of train_val as validation)
        # Drugs
        inner_skf_drug = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
        try:
            d_train_inner, d_val_inner = next(inner_skf_drug.split(drug_train_val_idx, drug_cluster_labels[drug_train_val_idx]))
        except ValueError:
            np.random.shuffle(drug_train_val_idx)
            d_val_size = len(drug_train_val_idx) // 8
            d_val_inner = np.arange(d_val_size)
            d_train_inner = np.arange(d_val_size, len(drug_train_val_idx))
            
        drug_train_idx = drug_train_val_idx[d_train_inner]
        drug_val_idx = drug_train_val_idx[d_val_inner]
        
        # Targets
        inner_skf_target = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
        try:
            t_train_inner, t_val_inner = next(inner_skf_target.split(target_train_val_idx, target_cluster_labels[target_train_val_idx]))
        except ValueError:
            np.random.shuffle(target_train_val_idx)
            t_val_size = len(target_train_val_idx) // 8
            t_val_inner = np.arange(t_val_size)
            t_train_inner = np.arange(t_val_size, len(target_train_val_idx))
            
        target_train_idx = target_train_val_idx[t_train_inner]
        target_val_idx = target_train_val_idx[t_val_inner]
        
        # Sets for fast lookup
        D_train, D_val, D_test = set(drug_train_idx), set(drug_val_idx), set(drug_test_idx)
        T_train, T_val, T_test = set(target_train_idx), set(target_val_idx), set(target_test_idx)
        
        # Initialize interaction lists for this fold
        s4_train, s4_val, s4_test = [], [], []
        s2_train, s2_val, s2_test = [], [], []
        s3_train, s3_val, s3_test = [], [], []
        
        for r, c, idx in all_interactions:
            # Setting 4: Both Cold
            if r in D_test and c in T_test: s4_test.append(idx)
            elif r in D_val and c in T_val: s4_val.append(idx)
            elif r in D_train and c in T_train: s4_train.append(idx)
            
            # Setting 2: Cold Drug
            if r in D_test: s2_test.append(idx)
            elif r in D_val: s2_val.append(idx)
            elif r in D_train: s2_train.append(idx)
            
            # Setting 3: Cold Target
            if c in T_test: s3_test.append(idx)
            elif c in T_val: s3_val.append(idx)
            elif c in T_train: s3_train.append(idx)
            
        cv_s4.append({'train': s4_train, 'val': s4_val, 'test': s4_test})
        cv_s2.append({'train': s2_train, 'val': s2_val, 'test': s2_test})
        cv_s3.append({'train': s3_train, 'val': s3_val, 'test': s3_test})
        
        # Setting 1: Warm Start
        s1_train_val_idx, s1_test_idx = s1_folds[fold_idx]
        np.random.shuffle(s1_train_val_idx)
        val_size = len(s1_train_val_idx) // 8
        s1_val_idx = s1_train_val_idx[:val_size]
        s1_train_idx = s1_train_val_idx[val_size:]
        
        cv_s1.append({
            'train': [all_interactions[i][2] for i in s1_train_idx],
            'val': [all_interactions[i][2] for i in s1_val_idx],
            'test': [all_interactions[i][2] for i in s1_test_idx]
        })
        
    # ---------------------------------------------------------
    # 5. SAVE FILES
    # ---------------------------------------------------------
    settings = {1: cv_s1, 2: cv_s2, 3: cv_s3, 4: cv_s4}
    
    for s_idx, cv_data in settings.items():
        json_file = os.path.join(folds_dir, f'cv_folds_setting{s_idx}_stratified.json')
        train_txt = os.path.join(folds_dir, f'train_fold_setting{s_idx}_stratified.txt')
        test_txt = os.path.join(folds_dir, f'test_fold_setting{s_idx}_stratified.txt')
        
        with open(json_file, 'w') as f: json.dump(cv_data, f)
        with open(train_txt, 'w') as f: json.dump([fold['train'] for fold in cv_data], f)
        with open(test_txt, 'w') as f: json.dump([fold['test'] for fold in cv_data], f)
        
    print(f"[{dataset_name}] Successfully generated files for Settings 1-4!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['davis', 'kiba', 'all'])
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generating splits')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        generate_stratified_folds('davis', args.data_path, args.seed)
        generate_stratified_folds('kiba', args.data_path, args.seed)
    else:
        generate_stratified_folds(args.dataset, args.data_path, args.seed)
