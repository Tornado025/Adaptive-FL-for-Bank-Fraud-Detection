import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

@dataclass
class TrainingConfig:
    local_epochs: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4
    device: str = "cpu"
    bank_id: str = "bank_a"

class TrainingHistory:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_auc = []

def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> TrainingHistory:
    """Runs local training for config.local_epochs epochs."""
    
    device = torch.device(config.device)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Calculate positive class weight
    num_positive = 0
    num_negative = 0
    for _, labels in train_loader:
        num_positive += int(labels.sum().item())
        num_negative += int((labels == 0).sum().item())
    
    pos_weight_val = num_negative / max(1, num_positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    
    history = TrainingHistory()
    
    for epoch in range(config.local_epochs):
        model.train()
        train_losses = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())
                
        avg_val_loss = sum(val_losses) / max(1, len(val_losses))
        
        # Metrics
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5 # If only one class is present
            
        history.train_loss.append(avg_train_loss)
        history.val_loss.append(avg_val_loss)
        history.val_auc.append(auc)
        
        print(f"       [{config.bank_id}] Epoch {epoch+1}/{config.local_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {auc:.4f}")
              
    return history


def train_local_fedprox(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    global_params: dict,
    mu: float = 0.01,
) -> tuple:
    """
    FedProx-aware local training.

    Minimises:
        L_k(w) = BCE(w) + (mu/2) * ||w_base - w_global_base||²

    The proximal term is computed only over base_layers parameters to match
    the Track 3 server contract (only base weights are shared / aggregated).

    Args:
        model:         FraudDetectionMLP instance (base + top layers).
        train_loader:  DataLoader for the bank's training split.
        val_loader:    DataLoader for the bank's validation split.
        config:        TrainingConfig (same as train_local).
        global_params: Dict {layer_name: torch.Tensor} of the current global
                       base-layer weights (snapshot taken BEFORE training starts).
        mu:            FedProx proximal coefficient. Typical: 0.01.

    Returns:
        (history, final_proximal_term)
        history            — TrainingHistory with per-epoch metrics.
        final_proximal_term — ||w_final_base - w_global_base||² (scalar float).
                              This is what Track 3 expects in metadata["proximal_term"].
    """
    device = torch.device(config.device)
    model = model.to(device)

    # Move global snapshot to the same device
    global_tensors = {k: v.clone().to(device) for k, v in global_params.items()}

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Class-imbalance weight for BCE
    num_positive = 0
    num_negative = 0
    for _, labels in train_loader:
        num_positive += int(labels.sum().item())
        num_negative += int((labels == 0).sum().item())

    pos_weight_val = num_negative / max(1, num_positive)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val]).to(device)
    )

    history = TrainingHistory()

    for epoch in range(config.local_epochs):
        model.train()
        train_losses = []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            local_loss = criterion(outputs, labels)

            # FedProx proximal term — base layers only
            prox = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if name in global_tensors:
                    prox = prox + torch.sum((param - global_tensors[name]) ** 2)

            loss = local_loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()

            # Record only the local BCE loss so the metric is interpretable
            train_losses.append(local_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation (identical to train_local)
        model.eval()
        val_losses = []
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss.item())

                probs = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        avg_val_loss = sum(val_losses) / max(1, len(val_losses))

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5

        history.train_loss.append(avg_train_loss)
        history.val_loss.append(avg_val_loss)
        history.val_auc.append(auc)

        print(
            f"       [{config.bank_id}] Epoch {epoch+1}/{config.local_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC: {auc:.4f} | Prox: {(mu/2)*prox.item():.6f}"
        )

    # Compute final proximal term ||w_final - w_global||²
    final_proximal_term = 0.0
    model.eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in global_tensors:
                final_proximal_term += torch.sum(
                    (param - global_tensors[name]) ** 2
                ).item()

    return history, final_proximal_term
