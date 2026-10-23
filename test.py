import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from tqdm import tqdm


def get_cindex(Y, P):
    Y = np.asarray(Y)
    P = np.asarray(P)
    
    summ = 0.0
    pair = 0
    
    # Process in a vectorized way to avoid O(N^2) slow Python loops
    for i in range(1, len(Y)):
        Y_j = Y[0:i]
        P_j = P[0:i]
        
        y_diff = Y[i] - Y_j
        p_diff = P[i] - P_j
        
        # Where Y[i] > Y[j]
        mask_gt = (y_diff > 0)
        pair += mask_gt.sum()
        summ += (p_diff[mask_gt] > 0).sum() + 0.5 * (p_diff[mask_gt] == 0).sum()
        
        # Where Y[i] < Y[j]
        mask_lt = (y_diff < 0)
        pair += mask_lt.sum()
        summ += (p_diff[mask_lt] < 0).sum() + 0.5 * (p_diff[mask_lt] == 0).sum()
        
    if pair != 0:
        return summ / pair
    else:
        return 0

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    y_pred_mean = [np.mean(y_pred) for _ in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    
    if y_obs_sq == 0 or y_pred_sq == 0:
        return 0

    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    denom = sum(y_pred * y_pred)
    if denom == 0:
        return 0
    return sum(y_obs * y_pred) / float(denom)

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down= sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    if down == 0:
        return 0
    return 1 - (upp / float(down))

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    
    # 修正了 DeepDTA 原始代码中的平方 bug: r2 本身已经是平方了，不应再 r2**2
    return r2 * (1 - np.sqrt(np.absolute(r2 - r02)))

def test_epoch(model, dataloader, dataset_name='kiba', device='cuda', return_preds=False):
    model.eval()
    total_loss = 0
    
    mse_criterion = nn.MSELoss()
    
    y_preds = []
    y_trues = []
    
    with torch.no_grad():
        for smi_feat, prot_feat, y_true, smi_mask, prot_mask in tqdm(dataloader, desc="Testing", leave=False):
            smi_feat = smi_feat.to(device)
            prot_feat = prot_feat.to(device)
            y_true = y_true.to(device)
            smi_mask = smi_mask.to(device)
            prot_mask = prot_mask.to(device)
            
            y_main, h_pred, s_pred, _, _ = model(smi_feat, prot_feat, smi_mask, prot_mask)
            
            loss = mse_criterion(y_main, y_true)
            total_loss += loss.item()
            
            y_preds.extend(y_main.squeeze(-1).cpu().numpy().tolist())
            y_trues.extend(y_true.squeeze(-1).cpu().numpy().tolist())
            
    num_batches = len(dataloader)
    if num_batches == 0:
        if return_preds:
            return 0, 0, 0, 0, [], []
        return 0, 0, 0, 0
    avg_loss = total_loss / num_batches
    
    # Calculate metrics
    y_preds_np = np.array(y_preds)
    y_trues_np = np.array(y_trues)
    
    mse = np.mean((y_preds_np - y_trues_np)**2)
    rmse = sqrt(mse)
    ci = get_cindex(y_trues_np, y_preds_np)
    rm2 = get_rm2(y_trues_np, y_preds_np)
    
    # Calculate AUPR based on dataset threshold
    threshold = 7.0 if dataset_name.lower() == 'davis' else 12.1
    y_trues_bin = (y_trues_np >= threshold).astype(int)
    try:
        from sklearn.metrics import average_precision_score
        aupr = average_precision_score(y_trues_bin, y_preds_np)
    except Exception as e:
        print(f"AUPR Error: {e}")
        aupr = 0.0
    
    if return_preds:
        return avg_loss, mse, rmse, ci, rm2, aupr, y_trues, y_preds
    return avg_loss, mse, rmse, ci, rm2, aupr
