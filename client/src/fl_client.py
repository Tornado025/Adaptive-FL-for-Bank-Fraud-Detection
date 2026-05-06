"""
FL Client — Bank-Side Federated Learning Participant
=====================================================
Wraps the existing local training pipeline (Track 2) into a stateful FL client
that participates in round-by-round federated learning.

Each bank has one FLClient instance. Across rounds:
  - Top layers persist on the client (personalised, never shared)
  - Base layers are replaced by the aggregated global weights at the start of
    each round, then updated via FedProx local training
  - The resulting base weights + metadata are packaged and returned to the server

Usage (by fl_runner.py):
    client = FLClient(bank_id="bank_a", db_path="data/databases/bank_a.db")
    package = client.local_train_and_package(global_weights, round_num=1)
    # package is a full Track 3 weight_package dict, ready for aggregate()
"""

from __future__ import annotations

import os
import sys
import copy

import torch

# ── Path setup — allows imports from repo root regardless of working directory ──
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from client.src.model import FraudDetectionMLP
from client.src.train import TrainingConfig, train_local_fedprox
from client.src.weight_extractor import extract_base_weights, load_base_weights
from data.src.dataloaders import get_dataloader


class FLClient:
    """
    A stateful FL client representing one bank node.

    State that persists across rounds:
        self.model — the full FraudDetectionMLP; base layers are overwritten
                     each round from global weights, top layers accumulate
                     personalisation from all local training runs.

    Attributes:
        bank_id      — "bank_a", "bank_b", "bank_c", or "bank_d"
        db_path      — Absolute path to the bank's SQLite database
        mu           — FedProx proximal coefficient (default 0.01)
        local_epochs — Epochs per FL round (default 3)
        device       — "cpu" or "cuda" (default "cpu")
    """

    def __init__(
        self,
        bank_id: str,
        db_path: str,
        mu: float = 0.01,
        local_epochs: int = 3,
        device: str = "cpu",
    ):
        self.bank_id = bank_id
        self.db_path = db_path
        self.mu = mu
        self.local_epochs = local_epochs
        self.device = device
        self.model: FraudDetectionMLP | None = None
        self._feature_dim: int | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_initialised(self) -> None:
        """Lazily creates the model on the first round."""
        if self.model is not None:
            return

        # A single dataset access is enough to determine FEATURE_DIM
        get_dataloader(self.db_path, batch_size=1, split="val")
        from data.src.dataloaders import FEATURE_DIM  # set by the call above
        self._feature_dim = FEATURE_DIM
        self.model = FraudDetectionMLP(input_dim=FEATURE_DIM)

    def _snapshot_base_params(self) -> dict[str, torch.Tensor]:
        """
        Returns a frozen copy of the current base_layer parameter tensors.
        Used as the proximal anchor point: snapshot is taken AFTER loading
        global weights, BEFORE local optimisation begins.
        """
        snapshot = {}
        for name, param in self.model.named_parameters():
            if name.startswith("base_layers"):
                snapshot[name] = param.detach().clone()
        return snapshot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def local_train_and_package(
        self,
        global_weights: dict[str, list],
        round_num: int,
    ) -> dict:
        """
        Core FL round method. Called once per round by fl_runner.py.

        Sequence:
            1. Initialise model (first round only)
            2. Load global base weights into the model → synchronise with server
            3. Snapshot the base params as the proximal anchor (w_global)
            4. Run FedProx local training (local_epochs epochs)
            5. Extract updated base weights + full metadata
            6. Return a complete Track 3 weight_package dict

        Args:
            global_weights: Dict mapping layer name → nested list (from server).
                            Contains only base_layers.* keys.
            round_num:      Current FL round number (1-indexed).

        Returns:
            weight_package dict ready for Track 3's aggregate():
            {
                "bank_id": ...,
                "round": ...,
                "num_samples": ...,
                "weights": { base_layers.*: [...] },
                "metadata": {
                    "val_loss": float,
                    "val_auc": float,
                    "local_epochs_trained": int,
                    "mu": float,
                    "proximal_term": float,   ← ||w_final - w_global||²
                }
            }
        """
        self._ensure_model_initialised()

        # ── Step 2: Load global base weights ─────────────────────────
        load_base_weights(self.model, {"weights": global_weights})

        # ── Step 3: Snapshot for proximal anchor ──────────────────────
        global_params_snapshot = self._snapshot_base_params()

        # ── Step 4: Build dataloaders and train ───────────────────────
        train_loader = get_dataloader(self.db_path, batch_size=256, split="train")
        val_loader   = get_dataloader(self.db_path, batch_size=256, split="val")

        config = TrainingConfig(
            bank_id=self.bank_id,
            local_epochs=self.local_epochs,
            device=self.device,
        )

        history, proximal_term = train_local_fedprox(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            global_params=global_params_snapshot,
            mu=self.mu,
        )

        # ── Step 5: Assemble metadata ─────────────────────────────────
        num_samples = len(train_loader.dataset)

        metadata = {
            "val_loss":             round(history.val_loss[-1], 6),
            "val_auc":              round(history.val_auc[-1], 6),
            "local_epochs_trained": self.local_epochs,
            "mu":                   self.mu,
            "proximal_term":        round(proximal_term, 8),
        }

        # ── Step 6: Extract and return weight package ─────────────────
        return extract_base_weights(
            model=self.model,
            bank_id=self.bank_id,
            round_num=round_num,
            num_samples=num_samples,
            metadata=metadata,
        )
