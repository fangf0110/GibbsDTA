import os
import json
import pickle
import numpy as np
import random
import argparse

def generate_cold_folds(dataset_name, data_path='data'):
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
    
    # Helper to divide into 6 groups (1 test, 5 train)
    def split_into_6(items):
        random.shuffle(items)
        fold_size = len(items) // 6
        groups = []
        start = 0
        for i in range(6):
            if i == 5:
                groups.append(set(items[start:]))
            else:
                groups.append(set(items[start:start+fold_size]))
                start += fold_size
        return groups
        
    drug_groups = split_into_6(list(range(num_drugs)))
    target_groups = split_into_6(list(range(num_targets)))
    
    folds_dir = os.path.join(data_path, dataset_name, 'folds')
    os.makedirs(folds_dir, exist_ok=True)
    
    # ------------------
    # Setting 2: Cold Drug
    # ------------------
    test_s2 = []
    train_s2 = [[] for _ in range(5)]
    
    for r, c in zip(valid_rows, valid_cols):
        idx = int(r * num_targets + c)
        if r in drug_groups[0]: # test
            test_s2.append(idx)
        else:
            for i in range(1, 6):
                if r in drug_groups[i]:
                    train_s2[i-1].append(idx)
                    break
                    
    with open(os.path.join(folds_dir, 'test_fold_setting2.txt'), 'w') as f:
        json.dump(test_s2, f)
    with open(os.path.join(folds_dir, 'train_fold_setting2.txt'), 'w') as f:
        json.dump(train_s2, f)

    # ------------------
    # Setting 3: Cold Target
    # ------------------
    test_s3 = []
    train_s3 = [[] for _ in range(5)]
    
    for r, c in zip(valid_rows, valid_cols):
        idx = int(r * num_targets + c)
        if c in target_groups[0]: # test
            test_s3.append(idx)
        else:
            for i in range(1, 6):
                if c in target_groups[i]:
                    train_s3[i-1].append(idx)
                    break
                    
    with open(os.path.join(folds_dir, 'test_fold_setting3.txt'), 'w') as f:
        json.dump(test_s3, f)
    with open(os.path.join(folds_dir, 'train_fold_setting3.txt'), 'w') as f:
        json.dump(train_s3, f)
        
    # ------------------
    # Setting 4: Cold Both
    # ------------------
    test_s4 = []
    train_s4 = [[] for _ in range(5)]
    
    for r, c in zip(valid_rows, valid_cols):
        idx = int(r * num_targets + c)
        if r in drug_groups[0] and c in target_groups[0]:
            test_s4.append(idx)
        else:
            for i in range(1, 6):
                if r in drug_groups[i] and c in target_groups[i]:
                    train_s4[i-1].append(idx)
                    break
                    
    with open(os.path.join(folds_dir, 'test_fold_setting4.txt'), 'w') as f:
        json.dump(test_s4, f)
    with open(os.path.join(folds_dir, 'train_fold_setting4.txt'), 'w') as f:
        json.dump(train_s4, f)

    print(f"Generated folds for Settings 2, 3, and 4 for dataset '{dataset_name}' in {folds_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kiba')
    args = parser.parse_args()
    generate_cold_folds(args.dataset)
