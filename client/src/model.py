import torch
import torch.nn as nn

class FraudDetectionMLP(nn.Module):
    def __init__(self, input_dim: int, base_hidden_dims: list[int] = [256, 128, 64],
                 top_hidden_dims: list[int] = [32], dropout_rate: float = 0.3):
        super().__init__()

        # Construct base layers
        base_layers = []
        in_dim = input_dim
        for h_dim in base_hidden_dims:
            base_layers.append(nn.Linear(in_dim, h_dim))
            base_layers.append(nn.BatchNorm1d(h_dim))
            base_layers.append(nn.ReLU())
            base_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        
        self.base_layers = nn.Sequential(*base_layers)

        # Construct top layers
        top_layers = []
        for h_dim in top_hidden_dims:
            top_layers.append(nn.Linear(in_dim, h_dim))
            top_layers.append(nn.ReLU())
            in_dim = h_dim
        
        # Output layer (no sigmoid since we use BCEWithLogitsLoss)
        top_layers.append(nn.Linear(in_dim, 1))
        
        self.top_layers = nn.Sequential(*top_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.base_layers(x)
        output = self.top_layers(representation)
        # Squeeze output to shape (batch,)
        return output.squeeze(-1)

    def get_base_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 64-dim embedding before the top layers. Useful for debugging."""
        return self.base_layers(x)
