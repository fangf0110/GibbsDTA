import argparse
import torch
import torch.optim as optim
from dataset import get_dataloaders
from model import PhyNetDTA
from train import train_epoch
from test import test_epoch
import os
import csv

def main():
    parser = argparse.ArgumentParser(description='PhyNet-DTA Training')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='kiba', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lambda_phy', type=float, default=0.1, help='Weight for physics loss')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature factor T')
    parser.add_argument('--val_fold', type=int, default=0, help='Validation fold index (0-4)')
    parser.add_argument('--smi_encoder', type=str, default='chemberta', choices=['chemberta', 'unimol'], help='SMILES encoder type')
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2, 3, 4], help='Cold start setting (1-4)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--no_cross_attention', action='store_true', help='Disable cross-attention mechanism')
    parser.add_argument('--use_stratified', action='store_true', help='Use stratified clustering drug splits instead of random')
    parser.add_argument('--use_txt_folds', action='store_true', help='Read .txt format folds (e.g. train_fold_setting2.txt) instead of .json')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Setup results directory
    train_out_dir = os.path.join("results", args.dataset, f"setting_{args.setting}", "train", f"fold_{args.val_fold}")
    test_out_dir = os.path.join("results", args.dataset, f"setting_{args.setting}", "test", f"fold_{args.val_fold}")
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)
    
    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_path, args.dataset, batch_size=args.batch_size, 
        val_fold=args.val_fold, smi_encoder=args.smi_encoder, setting=args.setting,
        use_stratified=args.use_stratified,
        use_txt_folds=args.use_txt_folds
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Init model
    smi_dim = 512 if args.smi_encoder == 'unimol' else 384
    model = PhyNetDTA(smi_feat_dim=smi_dim, use_cross_attention=not args.no_cross_attention).to(args.device)
    
    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    best_val_score = -float('inf')
    start_epoch = 1
    best_model_path = os.path.join(train_out_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(train_out_dir, "latest_checkpoint.pth")
    log_path = os.path.join(train_out_dir, "training_log.csv")
    
    # Check if we can resume training
    if os.path.exists(latest_checkpoint_path):
        print(f"Found checkpoint at {latest_checkpoint_path}. Resuming training...")
        checkpoint = torch.load(latest_checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_score = checkpoint.get('best_val_score', -float('inf'))
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        file_mode = 'a'
    else:
        file_mode = 'w'
    
    print("Starting training...")
    with open(log_path, file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':
            writer.writerow(['Epoch', 'TrainLoss', 'ValMSE', 'ValRMSE', 'ValCI', 'ValRM2', 'ValAUPR'])
        f.flush()
        
        for epoch in range(start_epoch, args.epochs + 1):
            # Train
            train_loss, train_main_loss, train_phy_loss = train_epoch(
                model, train_loader, optimizer, 
                lambda_phy=args.lambda_phy, 
                temperature=args.temperature,
                device=args.device,
                dataset_name=args.dataset
            )
            
            # Validate
            val_loss, mse, rmse, ci, rm2, aupr = test_epoch(model, val_loader, dataset_name=args.dataset, device=args.device)
            
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Val MSE: {mse:.4f} | Val RMSE: {rmse:.4f} | Val CI: {ci:.4f} | Val RM2: {rm2:.4f} | Val AUPR: {aupr:.4f}")
                  
            writer.writerow([epoch, train_loss, mse, rmse, ci, rm2, aupr])
            f.flush()
            
            val_score = ci - 0.1 * mse
            scheduler.step()
                  
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), best_model_path)
                print(f">>> Saved best model based on Compound Score (CI - 0.1*MSE) = {val_score:.4f}!")
                
            # Save latest checkpoint for resuming
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_score': best_val_score,
            }, latest_checkpoint_path)

    print("\n--- Evaluating Best Model on Validation Set ---")
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()
    
    val_loss, val_mse, val_rmse, val_ci, val_rm2, val_aupr, val_trues, val_preds = test_epoch(model, val_loader, dataset_name=args.dataset, device=args.device, return_preds=True)
    
    # Save val metrics
    with open(os.path.join(train_out_dir, "val_metrics.txt"), 'w') as f:
        f.write(f"Val MSE: {val_mse:.4f}\n")
        f.write(f"Val RMSE: {val_rmse:.4f}\n")
        f.write(f"Val CI: {val_ci:.4f}\n")
        f.write(f"Val RM2: {val_rm2:.4f}\n")
        f.write(f"Val AUPR: {val_aupr:.4f}\n")
        
    # Save val predictions
    with open(os.path.join(train_out_dir, "val_predictions.csv"), 'w', newline='') as f:
        writer_val = csv.writer(f)
        writer_val.writerow(['y_true', 'y_pred'])
        for t, p in zip(val_trues, val_preds):
            writer_val.writerow([t, p])

    print("\n--- Final Evaluation on Unseen Test Set ---")
    test_loss, test_mse, test_rmse, test_ci, test_rm2, test_aupr, test_trues, test_preds = test_epoch(model, test_loader, dataset_name=args.dataset, device=args.device, return_preds=True)
    
    print(f"Final Test MSE: {test_mse:.4f}")
    print(f"Final Test RMSE: {test_rmse:.4f}")
    print(f"Final Test CI: {test_ci:.4f}")
    print(f"Final Test RM2: {test_rm2:.4f}")
    print(f"Final Test AUPR: {test_aupr:.4f}")
    
    # Save final metrics
    with open(os.path.join(test_out_dir, "metrics.txt"), 'w') as f:
        f.write(f"Test MSE: {test_mse:.4f}\n")
        f.write(f"Test RMSE: {test_rmse:.4f}\n")
        f.write(f"Test CI: {test_ci:.4f}\n")
        f.write(f"Test RM2: {test_rm2:.4f}\n")
        f.write(f"Test AUPR: {test_aupr:.4f}\n")
        
    # Save test predictions
    with open(os.path.join(test_out_dir, "test_predictions.csv"), 'w', newline='') as f:
        writer_test = csv.writer(f)
        writer_test.writerow(['y_true', 'y_pred'])
        for t, p in zip(test_trues, test_preds):
            writer_test.writerow([t, p])
        
if __name__ == '__main__':
    main()
