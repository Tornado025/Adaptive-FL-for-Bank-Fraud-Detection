import os
import sys
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

from client.src.model import FraudDetectionMLP
from client.src.weight_extractor import extract_base_weights, load_base_weights


class ModelManager:
    def __init__(self, repo_root: str, feature_dim: int, device: str = "cpu"):
        self.repo_root = repo_root
        self.device = torch.device(device)
        self.models_dir = os.path.join(repo_root, "server", "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.global_model = FraudDetectionMLP(input_dim=feature_dim)
        self.global_model = self.global_model.to(self.device)

    def initialize_global_model(self) -> None:
        for module in self.global_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        total_params = sum(
            p.numel() for p in self.global_model.parameters() if p.requires_grad
        )
        print(
            f"[MODEL] Initialized global model with {total_params:,} trainable parameters."
        )

    def save_checkpoint(self, round_num: int) -> str:
        path = os.path.join(self.models_dir, f"global_model_r{round_num:03d}.pt")
        torch.save(self.global_model.state_dict(), path)
        print(f"[MODEL] Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, round_num: int) -> None:
        path = os.path.join(self.models_dir, f"global_model_r{round_num:03d}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Global model checkpoint not found for round {round_num}: {path}"
            )
        state_dict = torch.load(path, weights_only=True, map_location=self.device)
        self.global_model.load_state_dict(state_dict)
        print(f"[MODEL] Loaded checkpoint from {path}")

    def get_global_weight_dict(self, round_num: int, num_samples: int = 0) -> dict:
        return extract_base_weights(
            self.global_model,
            bank_id="server",
            round_num=round_num,
            num_samples=num_samples,
        )

    def apply_aggregated_weights(self, weight_dict: dict) -> None:
        load_base_weights(self.global_model, weight_dict)
