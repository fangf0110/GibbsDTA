import os
from transformers import AutoTokenizer, AutoModel, EsmModel

def download_models(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    chem_dir = os.path.join(save_dir, "ChemBERTa-77M-MTR")
    esm_dir = os.path.join(save_dir, "esm2_t12_35M_UR50D")
    
    print(f"Downloading ChemBERTa-77M-MTR to {chem_dir}...")
    chem_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    chem_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    chem_tokenizer.save_pretrained(chem_dir)
    chem_model.save_pretrained(chem_dir)
    
    print(f"Downloading ESM-2 to {esm_dir}...")
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    esm_tokenizer.save_pretrained(esm_dir)
    esm_model.save_pretrained(esm_dir)
    
    print("Download complete! Models are now stored locally.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='premodel')
    args = parser.parse_args()
    download_models(args.save_dir)
