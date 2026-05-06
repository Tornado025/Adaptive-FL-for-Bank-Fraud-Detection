"""
Phase 1 — Standard FedAvg Baseline
===================================
Implements the canonical Federated Averaging algorithm (McMahan et al., 2017).

This is the control group: every subsequent method in the pipeline must outperform
FedAvg on the final benchmark. It performs a sample-weighted mean of all client
weight tensors, treating every client's update as equally trustworthy.

Algorithm:
    global_weight[key] = Σ_k (n_k / N_total) * client_weight_k[key]
"""

from __future__ import annotations

import numpy as np


def federated_average(weight_packages: list[dict]) -> dict[str, list]:
    """
    Standard FedAvg. Sample-weighted mean of all base layer tensors.

    Args:
        weight_packages: List of weight package dicts from Track 2. Each must contain:
            - "num_samples" (int): number of local training samples
            - "weights" (dict[str, list]): the model weight tensors as Python lists

    Returns:
        A single weights dict with the same keys as the inputs, containing the
        sample-weighted averaged tensors as Python lists (JSON-serializable).

    Raises:
        ValueError: If weight_packages is empty or if clients have mismatched weight keys.
    """
    if not weight_packages:
        raise ValueError("weight_packages must be non-empty.")

    # Validate that all packages share the same weight keys
    reference_keys = set(weight_packages[0]["weights"].keys())
    for pkg in weight_packages[1:]:
        if set(pkg["weights"].keys()) != reference_keys:
            raise ValueError(
                f"Mismatched weight keys between clients. "
                f"Expected {reference_keys}, got {set(pkg['weights'].keys())}."
            )

    # Compute total sample count
    total_samples: int = sum(pkg["num_samples"] for pkg in weight_packages)
    if total_samples == 0:
        raise ValueError("Total sample count across all clients is zero.")

    # Initialise the aggregated weight dict with zeros matching each tensor's shape
    aggregated: dict[str, np.ndarray] = {}
    for key in reference_keys:
        first_tensor = np.array(weight_packages[0]["weights"][key], dtype=np.float64)
        aggregated[key] = np.zeros_like(first_tensor)

    # Accumulate the weighted contributions
    for pkg in weight_packages:
        fraction: float = pkg["num_samples"] / total_samples
        for key in reference_keys:
            tensor = np.array(pkg["weights"][key], dtype=np.float64)
            aggregated[key] += fraction * tensor

    # Convert back to nested Python lists for JSON-serialisability
    return {key: aggregated[key].tolist() for key in reference_keys}


def federated_average_with_custom_weights(
    weight_packages: list[dict],
    effective_weights: dict[str, float],
) -> dict[str, list]:
    """
    Generalised weighted average using caller-supplied per-client weights.

    Used internally by Phase 6 (aggregator.py) which computes the full
    combined effective weight formula. The ``effective_weights`` dict must
    already be normalised (i.e. they must sum to 1.0).

    Args:
        weight_packages: Same format as ``federated_average``.
        effective_weights: Mapping from ``bank_id`` to normalised weight in [0, 1].

    Returns:
        JSON-serialisable weight dict (same structure as ``federated_average``).

    Raises:
        ValueError: If weight_packages is empty or if a bank_id in weight_packages
                    is not found in effective_weights.
    """
    if not weight_packages:
        raise ValueError("weight_packages must be non-empty.")

    reference_keys = set(weight_packages[0]["weights"].keys())

    aggregated: dict[str, np.ndarray] = {}
    for key in reference_keys:
        first_tensor = np.array(weight_packages[0]["weights"][key], dtype=np.float64)
        aggregated[key] = np.zeros_like(first_tensor)

    for pkg in weight_packages:
        bank_id = pkg["bank_id"]
        if bank_id not in effective_weights:
            raise ValueError(f"No effective weight provided for bank '{bank_id}'.")
        w = effective_weights[bank_id]
        for key in reference_keys:
            tensor = np.array(pkg["weights"][key], dtype=np.float64)
            aggregated[key] += w * tensor

    return {key: aggregated[key].tolist() for key in reference_keys}
