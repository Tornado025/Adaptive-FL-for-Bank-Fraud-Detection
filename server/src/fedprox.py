"""
Phase 5 — FedProx-Aware Aggregation
=====================================
FedProx (Li et al., 2020) handles heterogeneous (Non-IID) clients by adding a
proximal regularisation term to local training. On the server side, this module
aggregates with awareness of how much each client *drifted* from the global model
during local training — clients that drifted more are down-weighted.

**Background:**
During local training each bank minimises:
    F_k(w) = local_loss(w) + (μ/2) × ||w - w_global||²

The ``proximal_term`` in the weight package is ||w_final - w_global||² measured
at the end of local training. A high value indicates significant drift.

**Drift penalty formula:**
    drift_penalty_k = proximal_term_k / (proximal_term_k + 1 / μ_k)
    fedprox_weight_k = 1.0 - drift_penalty_k

When ``proximal_term = 0``: drift_penalty = 0, fedprox_weight = 1.0 (full trust).
When ``proximal_term → ∞``: drift_penalty → 1.0, fedprox_weight → 0.0.

The formula naturally lives in [0, 1] without requiring additional clamping
(as long as proximal_term ≥ 0 and μ > 0, which are both guaranteed by the
FedProx training objective in Track 2).
"""

from __future__ import annotations

import numpy as np
from .fed_avg import federated_average_with_custom_weights


def compute_fedprox_weights(
    weight_packages: list[dict],
    mu_default: float = 0.01,
) -> dict[str, float]:
    """
    Computes per-client FedProx weights based on proximal drift.

    Algorithm:
        1. Extract proximal_term from metadata["proximal_term"].
           If missing, default to 0.0 (no penalty applied).
        2. mu_k = metadata["mu"] if present, else mu_default.
        3. drift_penalty_k = proximal_term_k / (proximal_term_k + 1 / mu_k)
        4. fedprox_weight_k = 1.0 - drift_penalty_k
        5. Normalise so all fedprox_weights sum to 1.0.

    Args:
        weight_packages: List of client weight packages.
        mu_default:       Fallback proximal coefficient when not in metadata.

    Returns:
        Dict mapping bank_id → normalised FedProx weight in (0, 1].
        Example: {'bank_a': 0.31, 'bank_b': 0.28, 'bank_c': 0.22, 'bank_d': 0.19}
    """
    raw_weights: dict[str, float] = {}
    for pkg in weight_packages:
        bank_id: str = pkg["bank_id"]
        meta: dict = pkg.get("metadata", {})

        proximal_term: float = float(meta.get("proximal_term", 0.0))
        mu: float = float(meta.get("mu", mu_default))
        if mu <= 0:
            mu = mu_default

        # Drift penalty ∈ [0, 1)
        drift_penalty = proximal_term / (proximal_term + (1.0 / mu))
        raw_weights[bank_id] = 1.0 - drift_penalty  # ∈ (0, 1]

    total = sum(raw_weights.values())
    if total == 0.0:
        # Edge case: every client has drift → all weights → 0; fall back to uniform
        n = len(weight_packages)
        return {pkg["bank_id"]: 1.0 / n for pkg in weight_packages}

    return {bank_id: w / total for bank_id, w in raw_weights.items()}


def fedprox_aggregate(
    weight_packages: list[dict],
    mu_default: float = 0.01,
) -> dict[str, list]:
    """
    Aggregates weights combining FedProx drift weighting with sample-size weighting.

    Combined per-client weight:
        raw_k        = (n_k / N) × fedprox_weight_k
        effective_k  = raw_k / Σ raw_j     (normalised to sum = 1)

    global_weights[key] = Σ_k (effective_k × client_k_weights[key])

    Args:
        weight_packages: List of client weight packages.
        mu_default:       Fallback μ for clients that omit it from metadata.

    Returns:
        JSON-serialisable weight dict — same keys as clients, values are lists.
    """
    fedprox_weights = compute_fedprox_weights(weight_packages, mu_default=mu_default)

    total_samples: int = sum(pkg["num_samples"] for pkg in weight_packages)
    if total_samples == 0:
        raise ValueError("Total sample count across all clients is zero.")

    # Combine sample fraction with FedProx drift weight
    raw: dict[str, float] = {}
    for pkg in weight_packages:
        bank_id = pkg["bank_id"]
        sample_fraction = pkg["num_samples"] / total_samples
        raw[bank_id] = sample_fraction * fedprox_weights[bank_id]

    # Normalise to sum = 1
    total_raw = sum(raw.values())
    if total_raw == 0.0:
        n = len(weight_packages)
        effective: dict[str, float] = {pkg["bank_id"]: 1.0 / n for pkg in weight_packages}
    else:
        effective = {bank_id: r / total_raw for bank_id, r in raw.items()}

    return federated_average_with_custom_weights(weight_packages, effective)
