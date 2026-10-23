import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PhyNetDataset(Dataset):
    def __init__(self, data_path, dataset_name, fold_type, val_fold=0, smi_encoder='chemberta', setting=1, use_stratified=False, use_txt_folds=False):
        super(PhyNetDataset, self).__init__()
        # Load ligands
        with open(os.path.join(data_path, dataset_name, 'ligands_can.txt'), 'r') as f:
            ligands = json.load(f)
        
        # Load proteins
        with open(os.path.join(data_path, dataset_name, 'proteins.txt'), 'r') as f:
            proteins = json.load(f)
        
        # Load Y matrix
        with open(os.path.join(data_path, dataset_name, 'Y'), 'rb') as f:
            Y = pickle.load(f, encoding='latin1')
            
        # Load folds
        if use_txt_folds:
            suffix = "_stratified" if use_stratified else ""
            train_file = os.path.join(data_path, dataset_name, 'folds', f'train_fold_setting{setting}{suffix}.txt')
            test_file = os.path.join(data_path, dataset_name, 'folds', f'test_fold_setting{setting}{suffix}.txt')
            with open(train_file, 'r') as f:
                train_folds_list = json.load(f)
            with open(test_file, 'r') as f:
                test_folds_list = json.load(f)
                
            cv_folds = []
            for i in range(len(train_folds_list)):
                train_val_idx = list(train_folds_list[i])
                test_idx = list(test_folds_list[i])
                
                # Split train_val into train and val deterministically
                np.random.seed(42 + i)
                np.random.shuffle(train_val_idx)
                val_size = len(train_val_idx) // 8
                val_idx = train_val_idx[:val_size]
                train_idx = train_val_idx[val_size:]
                
                cv_folds.append({'train': train_idx, 'val': val_idx, 'test': test_idx})
        else:
            if use_stratified:
                fold_file = os.path.join(data_path, dataset_name, 'folds', f'cv_folds_setting{setting}_stratified.json')
            else:
                fold_file = os.path.join(data_path, dataset_name, 'folds', f'cv_folds_setting{setting}.json')
                
            with open(fold_file, 'r') as f:
                cv_folds = json.load(f)
            
        indices = cv_folds[val_fold][fold_type]
            
        self.ligand_keys = list(ligands.keys())
        self.protein_keys = list(proteins.keys())
        self.Y = Y
        
        self.smi_encoder = smi_encoder
        if self.smi_encoder == 'unimol':
            self.ligand_feat_dir = os.path.join(data_path, dataset_name, 'features', 'ligands_unimol')
        else:
            self.ligand_feat_dir = os.path.join(data_path, dataset_name, 'features', 'ligands')
            
        self.protein_feat_dir = os.path.join(data_path, dataset_name, 'features', 'proteins')
        
        self.num_proteins = Y.shape[1]
        
        self.samples = []
        for idx in indices:
            row = idx // self.num_proteins
            col = idx % self.num_proteins
            if np.isnan(self.Y[row, col]):
                continue
            self.samples.append((row, col, self.Y[row, col]))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        l_idx, p_idx, affinity = self.samples[idx]
        
        l_id = self.ligand_keys[l_idx]
        p_id = self.protein_keys[p_idx]
        
        # Load precomputed physical features
        smi_feat = torch.load(os.path.join(self.ligand_feat_dir, f"{l_id}.pt"), weights_only=True)
        prot_feat = torch.load(os.path.join(self.protein_feat_dir, f"{p_id}.pt"), weights_only=True)
        
        return smi_feat, prot_feat, torch.FloatTensor([affinity])

def collate_fn(batch):
    smi_feats = []
    prot_feats = []
    y_trues = []
    
    for smi_feat, prot_feat, y_true in batch:
        smi_feats.append(smi_feat)
        prot_feats.append(prot_feat)
        y_trues.append(y_true)
        
    # Pad sequences to max length in the batch
    smi_padded = pad_sequence(smi_feats, batch_first=True, padding_value=0.0)
    prot_padded = pad_sequence(prot_feats, batch_first=True, padding_value=0.0)
    
    # Create attention masks (True means masked/ignored)
    smi_mask = (smi_padded.sum(dim=-1) == 0.0)
    prot_mask = (prot_padded.sum(dim=-1) == 0.0)
    
    y_trues = torch.stack(y_trues)
    
    return smi_padded, prot_padded, y_trues, smi_mask, prot_mask

def get_dataloaders(data_path, dataset_name, batch_size=64, val_fold=0, smi_encoder='chemberta', setting=1, use_stratified=False, use_txt_folds=False):
    train_dataset = PhyNetDataset(data_path, dataset_name, 'train', val_fold=val_fold, smi_encoder=smi_encoder, setting=setting, use_stratified=use_stratified, use_txt_folds=use_txt_folds)
    val_dataset = PhyNetDataset(data_path, dataset_name, 'val', val_fold=val_fold, smi_encoder=smi_encoder, setting=setting, use_stratified=use_stratified, use_txt_folds=use_txt_folds)
    test_dataset = PhyNetDataset(data_path, dataset_name, 'test', val_fold=val_fold, smi_encoder=smi_encoder, setting=setting, use_stratified=use_stratified, use_txt_folds=use_txt_folds)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader, test_loader
