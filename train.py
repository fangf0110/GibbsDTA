import torch
import torch.nn as nn
from tqdm import tqdm

def mse_loss(pred, target):
    """
    Standard Mean Squared Error (MSE) Loss
    """
    return ((pred.view(-1) - target.view(-1)) ** 2).mean()

def train_epoch(model, dataloader, optimizer, lambda_phy=0.1, temperature=1.0, device='cuda', dataset_name='kiba'):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_phy_loss = 0
    
    # We now use the unified Joint MSE-Ranking Loss for ALL datasets
    
    for smi_feat, prot_feat, y_true, smi_mask, prot_mask in tqdm(dataloader, desc="Training", leave=False):
        smi_feat = smi_feat.to(device)
        prot_feat = prot_feat.to(device)
        y_true = y_true.to(device)
        smi_mask = smi_mask.to(device)
        prot_mask = prot_mask.to(device)
        
        optimizer.zero_grad()
        
        y_main, h_pred, s_pred, _, _ = model(smi_feat, prot_feat, smi_mask, prot_mask)
        
        # Main Loss (Data-driven)
        main_loss = mse_loss(y_main, y_true)
        
        # Physics-Informed Loss
        # Formulation: Y_pred_phy = H - T * S
        # Since KIBA score represents affinity (higher is stronger binding, similar to pKd), 
        # and standard Delta G = H - TS. Here we match the KIBA magnitude.
        y_phy = h_pred - temperature * s_pred
        phy_loss = mse_loss(y_phy, y_true)
        
        loss = main_loss + lambda_phy * phy_loss
        
        loss.backward()
        # Clip gradients to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_phy_loss += phy_loss.item()
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_main_loss / num_batches, total_phy_loss / num_batches
