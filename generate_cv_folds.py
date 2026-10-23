import os
import json
import pickle
import numpy as np
import random
import argparse

def split_into_k(items, k=5):
    random.shuffle(items)
    fold_size = len(items) // k
    groups = []
    start = 0
    for i in range(k):
        if i == k - 1:
            groups.append(items[start:])
        else:
            groups.append(items[start:start+fold_size])
            start += fold_size
    return groups

def generate_folds(dataset_name, data_path='data'):
    random.seed(42)
    np.random.seed(42)
    
    Y_path = os.path.join(data_path, dataset_name, 'Y')
    if not os.path.exists(Y_path):
        print(f"Dataset {dataset_name} not found at {Y_path}")
        return
        
    with open(Y_path, 'rb') as f:
        Y = pickle.load(f, encoding='latin1')
        
    num_drugs, num_targets = Y.shape
    valid_rows, valid_cols = np.where(~np.isnan(Y))
    
    all_interactions = []
    for r, c in zip(valid_rows, valid_cols):
        all_interactions.append((r, c, int(r * num_targets + c)))
        
    folds_dir = os.path.join(data_path, dataset_name, 'folds')
    os.makedirs(folds_dir, exist_ok=True)
    
    print(f"\n[{dataset_name}] Generating Strict 5-Fold CV splits (7:1:2 ratio)...")
    
    # ------------------
    # Setting 1: Warm Start (Random Interaction Split)
    # ------------------
    int_groups_5 = split_into_k(all_interactions, 5)
    
    cv_s1 = []
    for i in range(5):
        test_ints = int_groups_5[i]
        
        remain_ints = []
        for j in range(5):
            if i != j:
                remain_ints.extend(int_groups_5[j])
                
        # Split remaining 80% into 8 parts (1 part Val [10%], 7 parts Train [70%])
        remain_split = split_into_k(remain_ints, 8)
        val_ints = remain_split[0]
        train_ints = []
        for p in remain_split[1:]:
            train_ints.extend(p)
            
        cv_s1.append({
            'train': [x[2] for x in train_ints],
            'val': [x[2] for x in val_ints],
            'test': [x[2] for x in test_ints]
        })
        
    with open(os.path.join(folds_dir, 'cv_folds_setting1.json'), 'w') as f:
        json.dump(cv_s1, f)
        
    # ------------------
    # Setting 2: Cold Drug (Strict Disjoint Drugs)
    # ------------------
    drug_list = list(range(num_drugs))
    drug_groups_5 = split_into_k(drug_list, 5)
    
    cv_s2 = []
    for i in range(5):
        test_drugs = set(drug_groups_5[i])
        
        remain_drugs = []
        for j in range(5):
            if i != j:
                remain_drugs.extend(drug_groups_5[j])
                
        # Split remaining drugs into 8 parts (1 Val, 7 Train) to maintain strict drug coldness for validation
        remain_drug_split = split_into_k(remain_drugs, 8)
        val_drugs = set(remain_drug_split[0])
        train_drugs = set()
        for p in remain_drug_split[1:]:
            train_drugs.update(p)
            
        train_ints, val_ints, test_ints = [], [], []
        for r, c, idx in all_interactions:
            if r in test_drugs:
                test_ints.append(idx)
            elif r in val_drugs:
                val_ints.append(idx)
            elif r in train_drugs:
                train_ints.append(idx)
                
        cv_s2.append({'train': train_ints, 'val': val_ints, 'test': test_ints})
        
    with open(os.path.join(folds_dir, 'cv_folds_setting2.json'), 'w') as f:
        json.dump(cv_s2, f)
        
    # ------------------
    # Setting 3: Cold Target (Strict Disjoint Targets)
    # ------------------
    target_list = list(range(num_targets))
    target_groups_5 = split_into_k(target_list, 5)
    
    cv_s3 = []
    for i in range(5):
        test_targets = set(target_groups_5[i])
        
        remain_targets = []
        for j in range(5):
            if i != j:
                remain_targets.extend(target_groups_5[j])
                
        remain_target_split = split_into_k(remain_targets, 8)
        val_targets = set(remain_target_split[0])
        train_targets = set()
        for p in remain_target_split[1:]:
            train_targets.update(p)
            
        train_ints, val_ints, test_ints = [], [], []
        for r, c, idx in all_interactions:
            if c in test_targets:
                test_ints.append(idx)
            elif c in val_targets:
                val_ints.append(idx)
            elif c in train_targets:
                train_ints.append(idx)
                
        cv_s3.append({'train': train_ints, 'val': val_ints, 'test': test_ints})
        
    with open(os.path.join(folds_dir, 'cv_folds_setting3.json'), 'w') as f:
        json.dump(cv_s3, f)
        
    # ------------------
    # Setting 4: Cold Both (Strict Intersection)
    # ------------------
    cv_s4 = []
    for i in range(5):
        # Using the same subsets guarantees mathematically rigorous intersections
        test_drugs = set(drug_groups_5[i])
        test_targets = set(target_groups_5[i])
        
        remain_drugs = []
        remain_targets = []
        for j in range(5):
            if i != j:
                remain_drugs.extend(drug_groups_5[j])
                remain_targets.extend(target_groups_5[j])
                
        remain_drug_split = split_into_k(remain_drugs, 8)
        val_drugs = set(remain_drug_split[0])
        train_drugs = set()
        for p in remain_drug_split[1:]:
            train_drugs.update(p)
            
        remain_target_split = split_into_k(remain_targets, 8)
        val_targets = set(remain_target_split[0])
        train_targets = set()
        for p in remain_target_split[1:]:
            train_targets.update(p)
            
        train_ints, val_ints, test_ints = [], [], []
        for r, c, idx in all_interactions:
            if r in test_drugs and c in test_targets:
                test_ints.append(idx)
            elif r in val_drugs and c in val_targets:
                val_ints.append(idx)
            elif r in train_drugs and c in train_targets:
                train_ints.append(idx)
            # Other interactions (e.g. Train Drug x Test Target) are discarded strictly!
                
        cv_s4.append({'train': train_ints, 'val': val_ints, 'test': test_ints})
        
    with open(os.path.join(folds_dir, 'cv_folds_setting4.json'), 'w') as f:
        json.dump(cv_s4, f)
        
    print(f"[{dataset_name}] Fold generation complete.")
    for i in range(5):
        print(f" Fold {i}:")
        print(f"   Set 1 (Warm)  -> Train: {len(cv_s1[i]['train'])}, Val: {len(cv_s1[i]['val'])}, Test: {len(cv_s1[i]['test'])}")
        print(f"   Set 2 (ColdD) -> Train: {len(cv_s2[i]['train'])}, Val: {len(cv_s2[i]['val'])}, Test: {len(cv_s2[i]['test'])}")
        print(f"   Set 3 (ColdT) -> Train: {len(cv_s3[i]['train'])}, Val: {len(cv_s3[i]['val'])}, Test: {len(cv_s3[i]['test'])}")
        print(f"   Set 4 (Both)  -> Train: {len(cv_s4[i]['train'])}, Val: {len(cv_s4[i]['val'])}, Test: {len(cv_s4[i]['test'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['davis', 'kiba', 'all'])
    parser.add_argument('--data_path', type=str, default='data')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        generate_folds('davis', args.data_path)
        generate_folds('kiba', args.data_path)
    else:
        generate_folds(args.dataset, args.data_path)
