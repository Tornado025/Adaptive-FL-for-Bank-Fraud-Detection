"""
Phase 6 — Master Aggregation Pipeline
======================================
Combines all phases into a single ``aggregate()`` entry point. This is the
function that Track 4 calls once per FL round.

**Execution order for ``method="custom"`` (full pipeline):**
    1. Apply DP clipping + Gaussian noise to all packages (Phase 4)
    2. Score client reliability from val_auc metadata (Phase 2)
    3. Compute conflict penalties via cosine similarity (Phase 3)
    4. Compute FedProx drift weights (Phase 5)
    5. Merge all signals into effective_weight_k (formula below)
    6. Weighted average of weight tensors (Phase 1 math)

**Full combined weight formula:**
    raw_k           = (n_k / N) × reliability_k × (1 - conflict_penalty_k) × fedprox_weight_k
    effective_k     = raw_k / Σ raw_j          # normalised to sum = 1
    global[key]     = Σ_k (effective_k × client_k_weights[key])

**Minimum weight floor:**
No client's ``effective_weight`` may fall below ``MIN_WEIGHT_FLOOR`` (default 0.05)
after normalisation. This prevents a single round of coincidentally poor metrics
from silencing any bank entirely.

**Track 4 import contract:**
    from track3_federated_algorithms.src.aggregator import aggregate, AggregationResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fed_avg import federated_average, federated_average_with_custom_weights
from .reliability_scorer import score_all_clients
from .conflict_resolver import compute_conflict_penalty
from .dp_mechanism import apply_dp_to_packages, estimate_privacy_budget
from .fedprox import compute_fedprox_weights


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_WEIGHT_FLOOR: float = 0.05
"""No client may receive an effective weight below this value."""

VALID_METHODS = frozenset(["fedavg", "fedprox", "dp_fedavg", "custom"])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AggregationResult:
    """
    Output of a single aggregation round.

    Attributes:
        new_global_weights: The merged weight dict to replace the server state.
        round_diagnostics:  Rich metadata dict broadcast to the dashboard via
                            WebSocket by Track 4.
    """

    new_global_weights: dict[str, list]
    round_diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_weight_floor(
    weights: dict[str, float],
    floor: float = MIN_WEIGHT_FLOOR,
) -> dict[str, float]:
    """
    Ensures every client weight is at least ``floor``, then re-normalises.

    Uses an iterative approach: clients above the floor donate weight to those
    below it, proportionally reducing their own weight while guaranteeing all
    clients land at or above the floor. Converges in at most N iterations where
    N is the number of clients.

    Returns:
        Dict with same keys, values in [floor, 1] summing to 1.0.
    """
    result = dict(weights)
    n = len(result)

    # Guard: if floor * n >= 1, set everything to uniform (floor too large)
    if floor * n >= 1.0:
        uniform = 1.0 / n
        return {k: uniform for k in result}

    # Iteratively push below-floor clients up to exactly floor,
    # subtracting the shortfall proportionally from above-floor clients.
    for _ in range(n + 1):
        below = {k for k, v in result.items() if v < floor - 1e-12}
        if not below:
            break
        shortfall = sum(floor - result[k] for k in below)
        for k in below:
            result[k] = floor
        above = {k: v for k, v in result.items() if k not in below}
        total_above = sum(above.values())
        if total_above <= 0:
            break
        scale = (total_above - shortfall) / total_above
        for k in above:
            result[k] = result[k] * scale

    # Final normalisation (guards against floating-point drift)
    total = sum(result.values())
    return {k: v / total for k, v in result.items()}


def _build_per_client_diagnostics(
    weight_packages: list[dict],
    reliability_scores: dict[str, float],
    conflict_penalties: dict[str, float],
    fedprox_weights: dict[str, float],
    effective_weights: dict[str, float],
    total_samples: int,
    dp_applied: bool,
) -> dict:
    """Assembles the per-client diagnostics sub-dict for round_diagnostics."""
    per_client = {}
    for pkg in weight_packages:
        bid = pkg["bank_id"]
        per_client[bid] = {
            "val_auc": pkg["metadata"].get("val_auc", None),
            "reliability_score": reliability_scores.get(bid, 1.0),
            "conflict_penalty": conflict_penalties.get(bid, 0.0),
            "fedprox_weight": fedprox_weights.get(bid, 1.0 / len(weight_packages)),
            "sample_fraction": pkg["num_samples"] / total_samples if total_samples > 0 else 0.0,
            "effective_weight": effective_weights.get(bid, 0.0),
            "dp_applied": dp_applied,
        }
    return per_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate(
    weight_packages: list[dict],
    current_global_weights: dict,
    method: str = "custom",
    dp_config: Optional[dict] = None,
    fedprox_mu: float = 0.01,
    num_rounds: int = 20,
) -> AggregationResult:
    """
    Master aggregation entry point. Called once per FL round by Track 4.

    Args:
        weight_packages:        List of client weight packages from Track 2.
        current_global_weights: The server's current global model weights.
        method:                 Aggregation strategy. One of:
                                  "fedavg"    → Phase 1 only
                                  "fedprox"   → Phases 1+5
                                  "dp_fedavg" → Phases 1+4
                                  "custom"    → Phases 1+2+3+4+5 (full pipeline)
        dp_config:              Required when method includes DP:
                                  {"clip_norm": float, "noise_multiplier": float}
        fedprox_mu:             Fallback proximal coefficient when not in metadata.

    Returns:
        AggregationResult with new_global_weights and round_diagnostics.

    Raises:
        ValueError: If method is unknown or weight_packages is empty.
    """
    if not weight_packages:
        raise ValueError("weight_packages must be non-empty.")
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {sorted(VALID_METHODS)}.")

    n_clients = len(weight_packages)
    total_samples = sum(pkg["num_samples"] for pkg in weight_packages)

    # Initialise diagnostic placeholders
    reliability_scores: dict[str, float] = {pkg["bank_id"]: 1.0 for pkg in weight_packages}
    conflict_penalties: dict[str, float] = {pkg["bank_id"]: 0.0 for pkg in weight_packages}
    fedprox_wts: dict[str, float] = {pkg["bank_id"]: 1.0 / n_clients for pkg in weight_packages}
    dp_applied: bool = False
    privacy_budget: Optional[dict] = None

    # DP config defaults
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1
    if dp_config:
        clip_norm = float(dp_config.get("clip_norm", 1.0))
        noise_multiplier = float(dp_config.get("noise_multiplier", 1.1))

    # ------------------------------------------------------------------
    # Select and execute aggregation pipeline
    # ------------------------------------------------------------------

    if method == "fedavg":
        # ── Phase 1 only ──────────────────────────────────────────────
        new_weights = federated_average(weight_packages)

        # Build sample-fraction weights, then apply floor for diagnostics
        pre_floor: dict[str, float] = {
            pkg["bank_id"]: pkg["num_samples"] / total_samples
            for pkg in weight_packages
        }
        effective: dict[str, float] = _apply_weight_floor(pre_floor)

    elif method == "fedprox":
        # ── Phases 1 + 5 ─────────────────────────────────────────────
        fedprox_wts = compute_fedprox_weights(weight_packages, mu_default=fedprox_mu)

        raw: dict[str, float] = {}
        for pkg in weight_packages:
            bid = pkg["bank_id"]
            raw[bid] = (pkg["num_samples"] / total_samples) * fedprox_wts[bid]

        total_raw = sum(raw.values()) or 1.0
        effective = {bid: r / total_raw for bid, r in raw.items()}
        effective = _apply_weight_floor(effective)

        new_weights = federated_average_with_custom_weights(weight_packages, effective)

    elif method == "dp_fedavg":
        # ── Phases 1 + 4 ─────────────────────────────────────────────
        dp_packages = apply_dp_to_packages(
            weight_packages, current_global_weights, clip_norm, noise_multiplier
        )
        dp_applied = True
        new_weights = federated_average(dp_packages)

        # Build effective weights with floor applied
        pre_floor = {
            pkg["bank_id"]: pkg["num_samples"] / total_samples
            for pkg in dp_packages
        }
        effective = _apply_weight_floor(pre_floor)
        privacy_budget = estimate_privacy_budget(
            num_rounds=num_rounds,
            num_clients=n_clients,
            noise_multiplier=noise_multiplier,
            clip_norm=clip_norm,
        )
        # Swap original packages so diagnostics reference DP-processed weights
        weight_packages = dp_packages

    else:
        # ── method == "custom": full pipeline ─────────────────────────
        # Step 1 — Apply DP (Phase 4)
        dp_packages = apply_dp_to_packages(
            weight_packages, current_global_weights, clip_norm, noise_multiplier
        )
        dp_applied = True
        privacy_budget = estimate_privacy_budget(
            num_rounds=num_rounds,
            num_clients=n_clients,
            noise_multiplier=noise_multiplier,
            clip_norm=clip_norm,
        )

        # Use DP-processed packages for all subsequent signal computation
        active_packages = dp_packages

        # Step 2 — Reliability scoring (Phase 2)
        # Note: we score from the *original* metadata (DP does not touch metadata)
        reliability_scores = score_all_clients(weight_packages)

        # Step 3 — Conflict penalties (Phase 3)
        conflict_penalties = compute_conflict_penalty(active_packages)

        # Step 4 — FedProx drift weights (Phase 5)
        fedprox_wts = compute_fedprox_weights(weight_packages, mu_default=fedprox_mu)

        # Step 5 — Merge all signals
        raw = {}
        for pkg in active_packages:
            bid = pkg["bank_id"]
            sample_frac = pkg["num_samples"] / total_samples
            rel = reliability_scores.get(bid, 1.0)
            conf = conflict_penalties.get(bid, 0.0)
            fp = fedprox_wts.get(bid, 1.0 / n_clients)
            raw[bid] = sample_frac * rel * (1.0 - conf) * fp

        total_raw = sum(raw.values()) or 1.0
        effective_pre_floor = {bid: r / total_raw for bid, r in raw.items()}
        effective = _apply_weight_floor(effective_pre_floor)

        # Step 6 — Weighted average (Phase 1 math)
        new_weights = federated_average_with_custom_weights(active_packages, effective)

        # For diagnostics, use original packages (metadata preserved)
        weight_packages = weight_packages

    # ------------------------------------------------------------------
    # Build diagnostics
    # ------------------------------------------------------------------

    per_client = _build_per_client_diagnostics(
        weight_packages=weight_packages,
        reliability_scores=reliability_scores,
        conflict_penalties=conflict_penalties,
        fedprox_weights=fedprox_wts,
        effective_weights=effective,
        total_samples=total_samples,
        dp_applied=dp_applied,
    )

    round_diagnostics = {
        "per_client": per_client,
        "method": method,
        "privacy_budget_used": privacy_budget,
    }

    return AggregationResult(
        new_global_weights=new_weights,
        round_diagnostics=round_diagnostics,
    )
