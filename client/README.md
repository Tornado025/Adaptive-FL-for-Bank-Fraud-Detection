# Track 2 — Deep Learning (Client-Side)

**Role: The Local ML Engineer**

You build the brain that lives inside each bank. Your code runs entirely within the bank's environment — it never touches the network, never sees another bank's data, and never communicates with the server directly. You produce a trained model and a clean weight extraction function. That's your complete output.

Think of your work as a black box with two interfaces:
- **Input**: A DataLoader from Track 1
- **Output**: A dictionary of base layer weights, ready for Track 4 to transmit

Everything in between — architecture, training loop, optimization — is your domain.

---

## File Structure

```
track2_deep_learning/
├── src/
│   ├── model.py               # Phase 1: MLP architecture with explicit layer separation
│   ├── train.py               # Phase 2: Local training loop (forward, loss, backprop)
│   ├── weight_extractor.py    # Phase 3: Extracts only shared base layer weights
│   ├── hyperparameter_tuner.py # Phase 4: Grid/random search for optimal hyperparameters
│   └── evaluate.py            # Metrics: accuracy, F1, AUC-ROC, precision/recall
├── models/                    # Saved .pt checkpoint files (gitignored for large files)
├── tests/
│   ├── test_model.py          # Architecture shape tests, forward pass smoke tests
│   ├── test_training.py       # Loss reduction test, overfitting check on small batch
│   └── test_weight_extractor.py # Verify extracted weights match base layers exactly
└── requirements.txt
```

---

## Phase 1 — Architecture Design

**File: `src/model.py`**

Build a Multi-Layer Perceptron (MLP) using PyTorch. The **critical architectural requirement** is that the model is explicitly divided into two named sections with a clean separation boundary. This is not just a coding style choice — it is the mechanism that makes federated learning possible.

```
INPUT (FEATURE_DIM features)
        │
┌───────▼────────────────────────────────┐
│         SHARED BASE LAYERS             │  ← These weights travel to the server
│  Linear(FEATURE_DIM → 256) + BN + ReLU │
│  Linear(256 → 128) + BN + ReLU         │
│  Dropout(0.3)                           │
│  Linear(128 → 64) + BN + ReLU          │
└───────┬────────────────────────────────┘
        │  64-dimensional representation
┌───────▼────────────────────────────────┐
│       PERSONALIZED TOP LAYERS          │  ← These weights NEVER leave the bank
│  Linear(64 → 32) + ReLU               │
│  Linear(32 → 1) + Sigmoid             │
└───────┬────────────────────────────────┘
        │
      OUTPUT (fraud probability 0–1)
```

**Implementation:**

```python
class FraudDetectionMLP(nn.Module):
    def __init__(self, input_dim: int, base_hidden_dims: list[int] = [256, 128, 64],
                 top_hidden_dims: list[int] = [32], dropout_rate: float = 0.3):
        super().__init__()

        # SHARED BASE: Must be a named nn.Sequential for clean weight extraction
        self.base_layers = nn.Sequential(...)

        # PERSONALIZED TOP: Also a named nn.Sequential, separate from base
        self.top_layers = nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.base_layers(x)
        output = self.top_layers(representation)
        return output

    def get_base_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 64-dim embedding before the top layers. Useful for debugging."""
        return self.base_layers(x)
```

**Requirements:**
- Use `nn.BatchNorm1d` in base layers for training stability with imbalanced data.
- The `base_layers` and `top_layers` attribute names are part of the public contract with Track 3. Do not rename them.
- The model must be serializable with `torch.save(model.state_dict(), path)`.

---

## Phase 2 — Local Training Loop

**File: `src/train.py`**

Write a complete training loop for a single bank. This must be self-contained and runnable from the command line for testing purposes.

**Loss Function:** Use `nn.BCELoss()`. However, because fraud detection has severe class imbalance (~3.5% positive), you must apply **positive class weighting**:

```python
# Calculate weight for the positive (fraud) class
pos_weight = (num_negative_samples / num_positive_samples)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
# Note: BCEWithLogitsLoss combines sigmoid + BCE, so remove sigmoid from model output
# when using this loss. Adjust model.top_layers accordingly.
```

**Training loop structure:**

```python
def train_local(
    model: FraudDetectionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> TrainingHistory:
    """
    Runs local training for `config.local_epochs` epochs.
    Returns a history object with per-epoch train_loss, val_loss, val_auc.
    This function is called by Track 4's bank node container at the start of each FL round.
    """
```

**The `TrainingConfig` dataclass:**
```python
@dataclass
class TrainingConfig:
    local_epochs: int = 3         # Number of local epochs per FL round (keep low — FL standard)
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4    # L2 regularization
    device: str = "cpu"           # Banks may not have GPUs
    bank_id: str = "bank_a"       # For logging
```

**Validation metrics to log after each epoch:**
- Binary cross-entropy loss
- Accuracy
- AUC-ROC (use `sklearn.metrics.roc_auc_score`)
- F1 score at threshold 0.5
- Precision and Recall

**Proof of learning:** You must demonstrate that the model's validation loss decreases monotonically for at least the first 5 epochs on a clean bank dataset. Save a loss curve plot to `models/loss_curve_{bank_id}.png`. This is your Phase 2 deliverable.

---

## Phase 3 — Weight Extraction

**File: `src/weight_extractor.py`**

This is the file that defines the privacy boundary. It must extract exactly and only the `base_layers` weights — nothing from `top_layers`.

```python
def extract_base_weights(model: FraudDetectionMLP) -> dict[str, list]:
    """
    Extracts the shared base layer weights from a trained model.

    Returns a JSON-serializable dictionary:
    {
        "bank_id": "bank_a",
        "round": 3,
        "num_samples": 82341,         # Used by Track 3 for weighted aggregation
        "weights": {
            "base_layers.0.weight": [[...], ...],
            "base_layers.0.bias": [...],
            ...
        },
        "metadata": {
            "val_loss": 0.312,
            "val_auc": 0.934,
            "local_epochs_trained": 3
        }
    }

    IMPORTANT: Only keys starting with 'base_layers' are included.
    Any 'top_layers' key is a bug — raise ValueError if detected.
    """
```

```python
def load_base_weights(model: FraudDetectionMLP, weight_dict: dict) -> FraudDetectionMLP:
    """
    Applies a received global weight dictionary back into the model's base_layers.
    Top layers are untouched.
    Called at the start of each FL round to receive the server's aggregated update.
    """
```

**Serialization contract with Track 4:**
- All tensors must be converted to Python lists (JSON serializable)
- Include `num_samples` — Track 3 uses this for sample-weighted aggregation
- Include `val_auc` in metadata — Track 3 uses this for reliability scoring
- The payload must be transmittable as a JSON body in an HTTP POST request

---

## Phase 4 — Baseline Tuning

**File: `src/hyperparameter_tuner.py`**

Optimize the local hyperparameters to maximize AUC-ROC on the validation set of each bank independently. The goal is to establish the best possible local baseline before federated rounds begin.

**Search space:**
```python
search_space = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "batch_size": [128, 256, 512],
    "weight_decay": [0, 1e-5, 1e-4, 1e-3],
    "dropout_rate": [0.2, 0.3, 0.4],
    "base_hidden_dims": [[256, 128, 64], [512, 256, 128], [128, 64, 32]],
}
```

Use random search (sample 30 combinations). For each combination, train for 10 epochs on the bank's training set and evaluate AUC-ROC on validation. Save results to `models/tuning_results_{bank_id}.csv`.

**Target benchmarks** (aim to exceed these before FL begins):

| Bank | Target AUC-ROC | Target F1 |
|------|---------------|-----------|
| A (Domestic Retail) | ≥ 0.90 | ≥ 0.55 |
| B (International Corporate) | ≥ 0.88 | ≥ 0.50 |
| C (High-Fraud) | ≥ 0.92 | ≥ 0.70 |
| D (E-commerce) | ≥ 0.89 | ≥ 0.52 |

These benchmarks represent the standalone local performance. After federated learning rounds (Track 3), the global model should exceed these on every bank — that's the proof that federation adds value.

---

## Integration Contract with Track 1

Track 1 provides the DataLoaders. You consume them. Never read from SQLite directly in this track.

```python
# This is the only import you need from Track 1
import sys; sys.path.append('../track1_data_engineering')
from src.dataloaders import get_dataloader, FEATURE_DIM

train_loader = get_dataloader('../../track1_data_engineering/data/databases/bank_a.db',
                               batch_size=256, split='train')
val_loader = get_dataloader('...', batch_size=256, split='val')

# Validate the interface before writing any training code
features, labels = next(iter(train_loader))
assert features.dtype == torch.float32
assert labels.dtype == torch.float32
assert features.shape[1] == FEATURE_DIM
assert labels.shape == (256,) or labels.shape == (256, 1)
```

Raise an issue with Track 1 immediately if any assertion fails.

---

## Integration Contract with Track 4

Track 4 calls your training code from inside a Docker container. Your `train_local()` function and `extract_base_weights()` function are the public API. They must:

- Accept `db_path` as an absolute path (Docker volumes use `/data/bank_x.db`)
- Have no interactive prompts or blocking `input()` calls
- Log to `stdout` in a structured format (Track 4 captures this via Docker logs)
- Complete a single FL round (3 local epochs) in under 5 minutes on CPU

---

## Dependencies

```
torch>=2.0
scikit-learn>=1.3
numpy>=1.24
matplotlib>=3.7
pandas>=2.0  # for loading tuning results
```

Install: `pip install -r requirements.txt`
