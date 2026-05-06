"""
Phase 3 — Conflict-Aware Weight Aggregation
============================================
Banks with Non-IID data (e.g. Bank C with high-fraud distribution vs Bank A
with low-fraud) produce local updates that push the global model in opposing
directions. Naive FedAvg silently cancels these out. This module detects and
moderates explicit opposition using cosine similarity.

**Important note on intent:**
A penalised client is *not* necessarily wrong — it may simply reflect a
legitimately different fraud distribution. The conflict penalty *modulates*
influence; it does not silence a client. No client's effective weight may fall
below the global minimum floor of 0.05 (enforced in aggregator.py).

**Algorithm (compute_conflict_penalty):**
    1. Compute mean weight vector across all clients → consensus direction.
    2. For each client: cosine_similarity(client_vector, consensus_vector).
    3. penalty_k = max(0, -cosine_similarity_k)
       Only negative cosine similarity (direct opposition) incurs a penalty.
    4. normalised_penalty_k = penalty_k / (max(all penalties) + ε)
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10  # Small constant to prevent division by zero


def weights_to_vector(weights: dict[str, list]) -> np.ndarray:
    """
    Flattens all weight tensors in a dict into a single 1-D numpy array.

    The order follows ``weights.keys()`` iteration order, which is insertion-
    order in Python 3.7+. Callers must ensure the same model architecture is
    used across all clients so the flattened vectors are comparable.

    Args:
        weights: Dict mapping layer name → nested Python list (as produced by
                 Track 2 and returned by the aggregation pipeline).

    Returns:
        1-D numpy float64 array of all concatenated weight values.
    """
    parts = [np.array(v, dtype=np.float64).flatten() for v in weights.values()]
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns the cosine similarity between two 1-D vectors.

    Returns 0.0 if either vector has zero norm to avoid NaN propagation.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _EPS or norm_b < _EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_pairwise_conflict(weight_packages: list[dict]) -> np.ndarray:
    """
    Returns an N×N matrix of cosine similarities between all client weight vectors.

    Interpretation:
        +1.0  → full agreement (identical gradient direction)
         0.0  → orthogonal (no relationship)
        -1.0  → direct conflict (opposing gradient direction)

    Args:
        weight_packages: List of client weight packages.

    Returns:
        numpy array of shape (N, N) with float cosine similarities.
    """
    n = len(weight_packages)
    vectors = [weights_to_vector(pkg["weights"]) for pkg in weight_packages]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = _cosine_similarity(vectors[i], vectors[j])
    return matrix


def compute_conflict_penalty(weight_packages: list[dict]) -> dict[str, float]:
    """
    For each client, computes how much it conflicts with the majority direction.

    Algorithm:
        1. Compute mean weight vector across all clients (consensus direction).
        2. For each client: sim_k = cosine_similarity(client_vec, consensus_vec)
        3. raw_penalty_k = max(0, -sim_k)   ← only penalise true opposition
        4. normalised_penalty_k = raw_penalty_k / (max(all raw_penalties) + ε)

    A client that *agrees* with the consensus receives penalty = 0.0.
    A client that is the *mirror opposite* of the consensus receives penalty = 1.0.
    Intermediate conflicts receive proportional values in (0, 1).

    Args:
        weight_packages: List of client weight packages.

    Returns:
        Dict mapping bank_id → normalised penalty in [0.0, 1.0].
        Example: {'bank_a': 0.0, 'bank_b': 0.12, 'bank_c': 0.0, 'bank_d': 0.73}
    """
    vectors = {pkg["bank_id"]: weights_to_vector(pkg["weights"]) for pkg in weight_packages}

    # Consensus direction = unweighted mean of all client vectors
    stacked = np.stack(list(vectors.values()), axis=0)
    consensus = stacked.mean(axis=0)

    # Raw penalties: only negative cosine similarity counts
    raw_penalties: dict[str, float] = {}
    for bank_id, vec in vectors.items():
        sim = _cosine_similarity(vec, consensus)
        raw_penalties[bank_id] = max(0.0, -sim)

    max_penalty = max(raw_penalties.values()) if raw_penalties else 0.0

    normalised: dict[str, float] = {
        bank_id: raw / (max_penalty + _EPS)
        for bank_id, raw in raw_penalties.items()
    }

    # Clamp to [0, 1] to guard against floating-point edge cases
    return {bank_id: float(np.clip(v, 0.0, 1.0)) for bank_id, v in normalised.items()}
