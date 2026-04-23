"""
Phase 3 — PyTorch DataLoaders
===============================
Provides dataset and dataloader classes that read directly from per-bank
SQLite databases. This is the primary interface for Track 2.

Usage (from Track 2):
    from data.src.dataloaders import get_dataloader, FEATURE_DIM
    loader = get_dataloader("data/databases/bank_a.db", batch_size=256, split="train")
"""

import os
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# This constant MUST be consistent across all 4 banks.
# It is set dynamically the first time a dataset is created, then frozen.
# Track 2 should import this after the first dataloader init.
# ---------------------------------------------------------------------------
FEATURE_DIM: int | None = None


class BankTransactionDataset(Dataset):
    """
    Reads directly from a SQLite .db file.
    Returns (features_tensor, label_tensor) pairs.
    """

    def __init__(self, db_path: str, split: str = "train",
                 train_ratio: float = 0.8):
        """
        Parameters
        ----------
        db_path : str
            Path to the bank SQLite database.
        split : str
            "train" or "val".
        train_ratio : float
            Fraction of rows used for the training split.
        """
        global FEATURE_DIM

        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT * FROM transactions")
        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        conn.close()

        data = np.array(rows, dtype=np.float32)

        # Deterministic split
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(data))
        split_idx = int(len(data) * train_ratio)

        if split == "train":
            data = data[indices[:split_idx]]
        else:
            data = data[indices[split_idx:]]

        # Separate features and label
        # Columns: txn_id, ..., isFraud, ...
        # isFraud position determined by column names
        fraud_idx = col_names.index("isFraud")
        txn_id_idx = col_names.index("txn_id")

        # Remove txn_id and isFraud from features
        drop_indices = sorted([fraud_idx, txn_id_idx])
        self.labels = torch.tensor(data[:, fraud_idx], dtype=torch.float32)
        self.features = torch.tensor(
            np.delete(data, drop_indices, axis=1), dtype=torch.float32
        )

        # Set / verify FEATURE_DIM
        if FEATURE_DIM is None:
            FEATURE_DIM = self.features.shape[1]
        else:
            assert self.features.shape[1] == FEATURE_DIM, (
                f"Feature dimension mismatch: expected {FEATURE_DIM}, "
                f"got {self.features.shape[1]}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def get_dataloader(db_path: str, batch_size: int = 256,
                   split: str = "train") -> DataLoader:
    """
    Factory function — primary interface for Track 2.

    Returns a DataLoader yielding (features, label) batches:
      features : torch.float32, shape (batch, FEATURE_DIM)
      label    : torch.float32, shape (batch,)
    """
    ds = BankTransactionDataset(db_path, split=split)
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0)
