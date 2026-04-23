import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Add root to sys.path so we can import dataloaders
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

from data.src.dataloaders import get_dataloader
from client.src.model import FraudDetectionMLP

def main():
    print("="*60)
    print("  Generating Professor Evaluation Report")
    print("="*60)
    
    banks = ["bank_a", "bank_b", "bank_c", "bank_d"]
    device = torch.device("cpu")
    
    # Setup plotting grid (4 banks x 2 plots: ROC and Confusion Matrix)
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    fig.suptitle('Local Model Evaluation per Bank (Validation Set)', fontsize=16, y=0.98)
    
    report_text = ["=== NEURAL NETWORK ARCHITECTURE ===", ""]
    
    for idx, bank in enumerate(banks):
        print(f"Evaluating {bank}...")
        
        # Load validation data first to initialize FEATURE_DIM
        db_path = os.path.join(REPO_ROOT, "data", "databases", f"{bank}.db")
        val_loader = get_dataloader(db_path, batch_size=256, split="val")
        
        from data.src.dataloaders import FEATURE_DIM
        
        # Load model
        model = FraudDetectionMLP(input_dim=FEATURE_DIM)
        model.load_state_dict(torch.load(os.path.join(REPO_ROOT, "client", "models", f"{bank}_model.pt"), weights_only=True))
        model.eval()
        
        # Capture architecture for text report only once
        if idx == 0:
            report_text.append(str(model))
            total_params = sum(p.numel() for p in model.parameters())
            report_text.append(f"\nTotal Trainable Parameters: {total_params:,}")
            report_text.append("\n=== VALIDATION RESULTS ===")
            

        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.numpy())
                all_preds.extend(preds.numpy())
                
        # Calculate metrics
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(all_labels, all_preds)
        
        report_text.append(f"\n[{bank.upper()}]")
        report_text.append(f"Validation AUC-ROC: {roc_auc:.4f}")
        report_text.append(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Plot ROC Curve
        ax_roc = axes[idx, 0]
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'{bank.upper()} ROC Curve')
        ax_roc.legend(loc="lower right")
        
        # Plot Confusion Matrix
        ax_cm = axes[idx, 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_xlabel('Predicted Label (0=Legit, 1=Fraud)')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title(f'{bank.upper()} Confusion Matrix')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(REPO_ROOT, "client", "models", "professor_evaluation_report.png")
    text_path = os.path.join(REPO_ROOT, "client", "models", "model_architecture_report.txt")
    
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    with open(text_path, "w") as f:
        f.write("\n".join(report_text))
        
    print(f"\n[OK] Visual report saved to {plot_path}")
    print(f"[OK] Text report saved to {text_path}")

if __name__ == "__main__":
    main()
