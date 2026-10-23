import os
import json
import torch
from transformers import AutoTokenizer, EsmModel, AutoModel
from tqdm import tqdm

def extract_features(data_path, dataset_name, device='cuda'):
    # Setup directories
    ligand_feat_dir = os.path.join(data_path, dataset_name, 'features', 'ligands')
    protein_feat_dir = os.path.join(data_path, dataset_name, 'features', 'proteins')
    os.makedirs(ligand_feat_dir, exist_ok=True)
    os.makedirs(protein_feat_dir, exist_ok=True)
    
    # Load raw data
    with open(os.path.join(data_path, dataset_name, 'ligands_can.txt'), 'r') as f:
        ligands = json.load(f)
        
    with open(os.path.join(data_path, dataset_name, 'proteins.txt'), 'r') as f:
        proteins = json.load(f)
        
    print("Loading ChemBERTa model...")
    chem_model_path = os.path.join("premodel", "ChemBERTa-77M-MTR")
    chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_path)
    chem_model = AutoModel.from_pretrained(chem_model_path).to(device)
    chem_model.eval()

    print("Loading ESM-2 model...")
    esm_model_path = os.path.join("premodel", "esm2_t12_35M_UR50D")
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_path)
    esm_model = EsmModel.from_pretrained(esm_model_path).to(device)
    esm_model.eval()

    print("Extracting Ligand Features (ChemBERTa)...")
    with torch.no_grad():
        for l_id, smi in tqdm(ligands.items()):
            feat_path = os.path.join(ligand_feat_dir, f"{l_id}.pt")
            if os.path.exists(feat_path): continue
            
            inputs = chem_tokenizer(smi, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = chem_model(**inputs)
            # Sequence of hidden states: (seq_len, hidden_dim)
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu() 
            torch.save(hidden_states, feat_path)

    print("Extracting Protein Features (ESM-2)...")
    with torch.no_grad():
        for p_id, seq in tqdm(proteins.items()):
            feat_path = os.path.join(protein_feat_dir, f"{p_id}.pt")
            if os.path.exists(feat_path): continue
            
            inputs = esm_tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = esm_model(**inputs)
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu()
            torch.save(hidden_states, feat_path)
            
    print("Feature extraction complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='kiba')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    extract_features(args.data_path, args.dataset, args.device)
