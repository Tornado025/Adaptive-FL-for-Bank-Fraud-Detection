import os
import sys
import torch
import matplotlib.pyplot as plt

# Add root to sys.path so we can import dataloaders
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

from data.src.dataloaders import get_dataloader, FEATURE_DIM
from client.src.model import FraudDetectionMLP
from client.src.train import train_local, TrainingConfig

def main():
    print("="*60)
    print("  Starting Local Training (Client Models)")
    print("="*60)
    
    os.makedirs(os.path.join(REPO_ROOT, "client", "models"), exist_ok=True)
    
    banks = ["bank_a", "bank_b", "bank_c", "bank_d"]
    
    for bank in banks:
        print(f"\n--- Training {bank} ---")
        db_path = os.path.join(REPO_ROOT, "data", "databases", f"{bank}.db")
        
        # Load data (dataloaders initializes FEATURE_DIM dynamically)
        train_loader = get_dataloader(db_path, batch_size=256, split="train")
        val_loader = get_dataloader(db_path, batch_size=256, split="val")
        
        # Import FEATURE_DIM after first get_dataloader call sets it
        from data.src.dataloaders import FEATURE_DIM
        
        # Initialize model
        model = FraudDetectionMLP(input_dim=FEATURE_DIM)
        
        # Config (5 epochs to show loss going down)
        config = TrainingConfig(bank_id=bank, local_epochs=5)
        
        # Train
        history = train_local(model, train_loader, val_loader, config)
        
        # Plot and save loss curve
        plt.figure()
        plt.plot(history.train_loss, label='Train Loss')
        plt.plot(history.val_loss, label='Val Loss')
        plt.title(f"{bank} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        plt.savefig(os.path.join(REPO_ROOT, "client", "models", f"loss_curve_{bank}.png"))
        plt.close()
        
        # Save model
        torch.save(model.state_dict(), os.path.join(REPO_ROOT, "client", "models", f"{bank}_model.pt"))
        print(f"       Saved {bank}_model.pt and loss curve.")
        
    print("\n[OK] Local training complete.")

if __name__ == "__main__":
    main()
