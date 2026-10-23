import os
import json
import torch
import numpy as np
try:
    from unimol_tools import UniMolRepr
except ImportError:
    print("Please install Uni-Mol first: pip install unimol_tools rdkit")
    exit(1)

def extract_unimol_features(data_path='data', dataset_name='kiba'):
    """
    Extracts 3D structural embeddings for molecules using DP Technology's Uni-Mol.
    Uni-Mol automatically handles the RDKit 3D conformation generation under the hood.
    """
    ligand_feat_dir = os.path.join(data_path, dataset_name, 'features', 'ligands_unimol')
    os.makedirs(ligand_feat_dir, exist_ok=True)
    
    # Load raw data
    with open(os.path.join(data_path, dataset_name, 'ligands_can.txt'), 'r') as f:
        ligands = json.load(f)
        
    smi_keys = list(ligands.keys())
    smi_list = list(ligands.values())
    
    print(f"Loaded {len(smi_list)} SMILES strings.")
    print("Initializing UniMolRepr (This will load the Uni-Mol weights)...")
    
    # Initialize the UniMol representation model
    # data_type='molecule' loads the pre-trained weights for small molecules
    repr_model = UniMolRepr(data_type='molecule', remove_hs=False)
    
    print("Generating 3D conformations and extracting Uni-Mol embeddings...")
    print("Note: RDKit will use ETKDG to calculate 3D coordinates, this might take a while.")
    
    # get_repr automatically processes SMILES -> 3D Coords -> Uni-Mol Vectors
    # return_atomic_reprs=True ensures we get the (seq_len, 512) feature matrix for Cross-Attention
    repr_output = repr_model.get_repr(smi_list, return_atomic_reprs=True)
    
    atomic_reprs = repr_output['atomic_reprs']
    
    print("Saving features to disk...")
    for i, key in enumerate(smi_keys):
        feat_path = os.path.join(ligand_feat_dir, f"{key}.pt")
        
        # Convert the numpy array to a PyTorch tensor
        # Shape should be (num_atoms, 512)
        atom_feat = torch.FloatTensor(atomic_reprs[i])
        
        torch.save(atom_feat, feat_path)
        
    print(f"Successfully extracted Uni-Mol features for {len(smi_keys)} molecules!")
    print(f"Features saved in: {ligand_feat_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='kiba')
    args = parser.parse_args()
    
    extract_unimol_features(args.data_path, args.dataset)
