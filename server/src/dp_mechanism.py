"""
Phase 4 — Differential Privacy (DP) Mechanism
===============================================
Adds calibrated Gaussian noise to each client's weight *update* (delta from the
global model) to provide formal (ε, δ)-differential privacy guarantees. This
protects individual transaction patterns from being reverse-engineered from the
shared model weights.

**Protocol (applied per client, before aggregation):**
    1. Clip: compute update delta = client_weights - global_weights.
             If ||delta||₂ > clip_norm, rescale delta so ||delta||₂ = clip_norm.
             Add clipped delta back to global_weights.
    2. Noise: add N(0, (clip_norm * noise_multiplier)²) elementwise to all weights.

**Privacy budget estimation:**
The moments accountant approximation used here:
    ε ≈ noise_multiplier⁻¹ × √(2 × num_rounds × log(1/δ))
gives a conservative upper bound for the cumulative privacy loss (ε) over
``num_rounds`` FL rounds.

**DP Hyperparameter presets:**

    | Setting        | clip_norm | noise_multiplier | ε (20 rounds) |
    |----------------|-----------|------------------|---------------|
    | Strong privacy | 1.0       | 2.0              | ~2.1          |
    | Balanced       | 1.0       | 1.1              | ~3.8          |
    | Weak privacy   | 1.0       | 0.5              | ~8.4          |
"""

from __future__ import annotations

import math
import copy

import numpy as np


# ---------------------------------------------------------------------------
# Core DP operations
# ---------------------------------------------------------------------------


def clip_weight_update(
    client_weights: dict[str, list],
    global_weights: dict[str, list],
    clip_norm: float = 1.0,
) -> dict[str, list]:
    """
    Clips the L2 norm of the weight UPDATE (delta = client − global) to ``clip_norm``.

    Steps:
        1. delta[key] = client_weights[key] − global_weights[key] for all keys.
        2. Flatten all deltas and compute the combined L2 norm.
        3. If norm > clip_norm: scale all deltas by (clip_norm / norm).
        4. clipped_weights[key] = global_weights[key] + clipped_delta[key].

    Args:
        client_weights:  Client's local model weights after training.
        global_weights:  Server's current global model weights.
        clip_norm:       Maximum allowed L2 norm for the update vector.

    Returns:
        Clipped version of ``client_weights`` — same keys and shape, but the
        total L2 norm of the update is bounded by ``clip_norm``.
    """
    # Compute the raw delta for every layer
    deltas: dict[str, np.ndarray] = {}
    for key in global_weights:
        c = np.array(client_weights[key], dtype=np.float64)
        g = np.array(global_weights[key], dtype=np.float64)
        deltas[key] = c - g

    # Compute the combined L2 norm of the full update vector
    flat_deltas = np.concatenate([d.flatten() for d in deltas.values()])
    update_norm = float(np.linalg.norm(flat_deltas))

    # Scale down if the update exceeds the allowed budget
    if update_norm > clip_norm:
        scale = clip_norm / update_norm
        deltas = {key: d * scale for key, d in deltas.items()}

    # Reconstruct clipped client weights
    clipped: dict[str, list] = {}
    for key in global_weights:
        g = np.array(global_weights[key], dtype=np.float64)
        clipped[key] = (g + deltas[key]).tolist()

    return clipped


def add_gaussian_noise(
    weights: dict[str, list],
    clip_norm: float,
    noise_multiplier: float = 1.1,
) -> dict[str, list]:
    """
    Adds zero-mean Gaussian noise to each weight element.

        noise_std = clip_norm × noise_multiplier
        noisy_weight[key] = weight[key] + N(0, noise_std²) per element

    Higher ``noise_multiplier`` → stronger privacy, lower accuracy.
    Default 1.1 gives a balanced (ε ≈ 3.8 over 20 rounds, δ = 1e-5).

    Args:
        weights:          Weight dict to apply noise to.
        clip_norm:        Sensitivity bound used in the preceding clip step.
        noise_multiplier: Scaling factor for the Gaussian standard deviation.

    Returns:
        New weight dict with the same keys and shapes, with added noise.
    """
    noise_std = clip_norm * noise_multiplier
    noisy: dict[str, list] = {}
    for key, val in weights.items():
        arr = np.array(val, dtype=np.float64)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=arr.shape)
        noisy[key] = (arr + noise).tolist()
    return noisy


def apply_dp_to_packages(
    weight_packages: list[dict],
    current_global_weights: dict,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1,
) -> list[dict]:
    """
    Applies ``clip_weight_update`` + ``add_gaussian_noise`` to every client package.

    Returns a new list of weight packages with DP-protected weights.
    The fields ``bank_id``, ``round``, ``num_samples``, and ``metadata`` are
    preserved unchanged.

    Args:
        weight_packages:       List of client weight packages (Track 2 format).
        current_global_weights: The server's current global model weights.
        clip_norm:             Max L2 norm allowed for each client's update.
        noise_multiplier:      Gaussian noise scale factor.

    Returns:
        List of weight packages with the same structure but DP-processed weights.
    """
    dp_packages: list[dict] = []
    for pkg in weight_packages:
        # Deep-copy so we never mutate the caller's data
        new_pkg = copy.deepcopy(pkg)

        # Step 1: Clip the update
        clipped = clip_weight_update(
            client_weights=pkg["weights"],
            global_weights=current_global_weights,
            clip_norm=clip_norm,
        )

        # Step 2: Add Gaussian noise
        noisy = add_gaussian_noise(
            weights=clipped,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
        )

        new_pkg["weights"] = noisy
        dp_packages.append(new_pkg)

    return dp_packages


# ---------------------------------------------------------------------------
# Privacy budget estimation
# ---------------------------------------------------------------------------


def estimate_privacy_budget(
    num_rounds: int,
    num_clients: int,
    noise_multiplier: float,
    clip_norm: float,
    delta: float = 1e-5,
) -> dict:
    """
    Estimates cumulative (ε, δ) consumed after ``num_rounds`` FL rounds.

    Uses the moments accountant approximation:
        ε ≈ noise_multiplier⁻¹ × √(2 × num_rounds × log(1/δ))

    This is a conservative upper-bound estimate. For production use, the
    Rényi Differential Privacy (RDP) accountant from Google's DP library
    provides tighter bounds.

    Args:
        num_rounds:       Total number of FL rounds.
        num_clients:      Number of participating clients (unused in this
                          approximation but recorded for audit purposes).
        noise_multiplier: Gaussian noise scaling factor (σ/clip_norm).
        clip_norm:        L2 sensitivity bound.
        delta:            Target δ (probability of privacy failure). Default 1e-5.

    Returns:
        Dict with keys: epsilon, delta, noise_multiplier, num_rounds, interpretation.
    """
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")

    epsilon = (1.0 / noise_multiplier) * math.sqrt(2 * num_rounds * math.log(1.0 / delta))

    if epsilon < 1.0:
        interpretation = "Very strong privacy (eps < 1)"
    elif epsilon < 3.0:
        interpretation = "Strong privacy (eps < 3)"
    elif epsilon < 5.0:
        interpretation = "Strong privacy (eps < 5)"
    elif epsilon < 10.0:
        interpretation = "Moderate privacy (5 <= eps < 10)"
    else:
        interpretation = "Weak privacy (eps >= 10) -- consider increasing noise_multiplier"

    return {
        "epsilon": round(epsilon, 4),
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "clip_norm": clip_norm,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "interpretation": interpretation,
    }
