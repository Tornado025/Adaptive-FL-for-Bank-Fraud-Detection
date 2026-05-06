"""
Phase 2 — Metadata-Based Reliability Scoring
=============================================
Assigns a trust score in [0, 1] to each FL client based solely on its reported
validation AUC (``val_auc``) from the local held-out set.

**Design rationale — no proxy dataset:**
With four well-defined, fully-participating banks in a controlled simulation, a
separate server-side proxy dataset adds infrastructure cost without clear benefit.
The ``val_auc`` each client reports on its *own* local validation split is a
simpler, more honest signal — it directly captures how well the local update
improved model quality on that client's fraud distribution.

**Sigmoid-shaped mapping:**
    score = sigmoid((val_auc - threshold) * sensitivity_factor)

- ``threshold`` (default 0.85): The "neutral" AUC. A client at exactly this AUC
  receives a reliability score of 0.5 — neither rewarded nor penalised.
- ``sensitivity_factor`` (default 10): Controls the steepness of the sigmoid.
  Higher values create sharper differentiation between good and excellent clients.

**Tuning guidance:**
If all four banks consistently exceed 0.90 AUC, raise the threshold to 0.90 so
the scorer differentiates between *good* and *excellent* updates rather than
treating all as equally reliable.

**Score examples:**
    val_auc = 0.85  →  score ≈ 0.50   (neutral)
    val_auc = 0.93  →  score ≈ 0.88   (high trust)
    val_auc = 0.70  →  score ≈ 0.18   (low trust)
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Module-level defaults — change here to globally recalibrate the scorer.
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD: float = 0.85
DEFAULT_SENSITIVITY: float = 10.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Equivalent but avoids overflow for large negative x
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def score_client_reliability(
    weight_package: dict,
    threshold: float = DEFAULT_THRESHOLD,
    sensitivity_factor: float = DEFAULT_SENSITIVITY,
) -> float:
    """
    Returns a reliability score in [0.0, 1.0] using only client metadata.

    Algorithm:
        1. Extract val_auc from weight_package["metadata"]["val_auc"]
        2. score = sigmoid((val_auc - threshold) * sensitivity_factor)

    Args:
        weight_package: A client weight package dict (Track 2 format).
        threshold:         AUC value mapped to score = 0.5. Default 0.85.
        sensitivity_factor: Sigmoid steepness. Default 10.

    Returns:
        Reliability score in [0.0, 1.0].

    Raises:
        KeyError: If "metadata" or "val_auc" keys are absent from the package.
            (No silent fallback — a missing val_auc indicates a malformed package
            from Track 2 that should be caught early rather than silently trusted.)
    """
    val_auc: float = weight_package["metadata"]["val_auc"]
    return _sigmoid((val_auc - threshold) * sensitivity_factor)


def score_all_clients(
    weight_packages: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
    sensitivity_factor: float = DEFAULT_SENSITIVITY,
) -> dict[str, float]:
    """
    Returns reliability scores for all clients keyed by bank_id.

    Example return value:
        {'bank_a': 0.88, 'bank_b': 0.73, 'bank_c': 0.91, 'bank_d': 0.52}

    Args:
        weight_packages: List of client weight packages.
        threshold:         Forwarded to ``score_client_reliability``.
        sensitivity_factor: Forwarded to ``score_client_reliability``.

    Returns:
        Dict mapping bank_id → reliability score in [0.0, 1.0].

    Raises:
        KeyError: If any package is missing "metadata" or "val_auc".
    """
    return {
        pkg["bank_id"]: score_client_reliability(pkg, threshold, sensitivity_factor)
        for pkg in weight_packages
    }
