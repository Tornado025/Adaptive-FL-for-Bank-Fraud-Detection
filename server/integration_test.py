"""
Integration Test — Single Simulated FL Round
=============================================
Run this manually (not via pytest) to validate the full aggregation pipeline
end-to-end with mock weight packages representing all four banks.

Usage:
    cd track3_federated_algorithms/
    python integration_test.py

Expected output:
    Integration test passed.
    Diagnostics: { ... }
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.aggregator import aggregate, AggregationResult

# ---------------------------------------------------------------------------
# Mock weight packages for all four banks
# ---------------------------------------------------------------------------

mock_global_weights = {
    "layer.weight": [[0.0, 0.0], [0.0, 0.0]],
    "layer.bias":   [0.0, 0.0],
}

mock_packages = [
    {
        "bank_id": "bank_a",
        "round": 1,
        "num_samples": 80000,
        "weights": {
            "layer.weight": [[0.10, 0.20], [0.30, 0.40]],
            "layer.bias":   [0.50, 0.60],
        },
        "metadata": {
            "val_loss": 0.30, "val_auc": 0.92, "local_epochs_trained": 3,
            "mu": 0.01, "proximal_term": 0.002,
        },
    },
    {
        "bank_id": "bank_b",
        "round": 1,
        "num_samples": 65000,
        "weights": {
            "layer.weight": [[0.15, 0.25], [0.35, 0.45]],
            "layer.bias":   [0.55, 0.65],
        },
        "metadata": {
            "val_loss": 0.32, "val_auc": 0.89, "local_epochs_trained": 3,
            "mu": 0.01, "proximal_term": 0.005,
        },
    },
    {
        "bank_id": "bank_c",
        "round": 1,
        "num_samples": 95000,
        "weights": {
            "layer.weight": [[0.05, 0.10], [0.20, 0.25]],
            "layer.bias":   [0.40, 0.45],
        },
        "metadata": {
            "val_loss": 0.38, "val_auc": 0.86, "local_epochs_trained": 3,
            "mu": 0.01, "proximal_term": 0.012,
        },
    },
    {
        "bank_id": "bank_d",
        "round": 1,
        "num_samples": 50000,
        "weights": {
            "layer.weight": [[0.20, 0.30], [0.40, 0.50]],
            "layer.bias":   [0.60, 0.70],
        },
        "metadata": {
            "val_loss": 0.28, "val_auc": 0.94, "local_epochs_trained": 3,
            "mu": 0.01, "proximal_term": 0.001,
        },
    },
]

# ---------------------------------------------------------------------------
# Run integration test
# ---------------------------------------------------------------------------

def main():
    result = aggregate(
        weight_packages=mock_packages,
        current_global_weights=mock_global_weights,
        method="custom",
        dp_config={"clip_norm": 1.0, "noise_multiplier": 1.1},
        fedprox_mu=0.01,
    )

    # ── Structural assertions ─────────────────────────────────────────
    assert isinstance(result, AggregationResult), "Result must be an AggregationResult"
    assert isinstance(result.new_global_weights, dict), "new_global_weights must be a dict"
    assert set(result.new_global_weights.keys()) == set(mock_global_weights.keys()), \
        "new_global_weights must have same keys as global_weights"

    assert "per_client" in result.round_diagnostics, "round_diagnostics must have 'per_client'"
    assert "method" in result.round_diagnostics, "round_diagnostics must have 'method'"

    # ── Per-client assertions ─────────────────────────────────────────
    required_keys = {
        "val_auc", "reliability_score", "conflict_penalty",
        "fedprox_weight", "sample_fraction", "effective_weight", "dp_applied",
    }

    for bank_id in ["bank_a", "bank_b", "bank_c", "bank_d"]:
        assert bank_id in result.round_diagnostics["per_client"], \
            f"Missing bank '{bank_id}' in diagnostics"
        diag = result.round_diagnostics["per_client"][bank_id]

        missing = required_keys - set(diag.keys())
        assert not missing, f"bank '{bank_id}' diagnostics missing keys: {missing}"

        eff = diag["effective_weight"]
        assert 0.0 <= eff <= 1.0, f"{bank_id}: effective_weight {eff:.4f} out of [0,1]"
        assert eff >= 0.05, f"{bank_id}: effective_weight {eff:.4f} below floor 0.05"
        assert diag["dp_applied"] is True, f"{bank_id}: dp_applied should be True"

    # ── Sum to 1 assertion ────────────────────────────────────────────
    eff_weights = [
        result.round_diagnostics["per_client"][bid]["effective_weight"]
        for bid in ["bank_a", "bank_b", "bank_c", "bank_d"]
    ]
    total = sum(eff_weights)
    assert abs(total - 1.0) < 1e-6, f"Effective weights sum to {total:.8f}, expected 1.0"

    print("Integration test passed.")
    print("\nDiagnostics:")
    for bank_id, diag in result.round_diagnostics["per_client"].items():
        print(f"  {bank_id}:")
        for k, v in diag.items():
            if isinstance(v, float):
                print(f"    {k:25s}: {v:.4f}")
            else:
                print(f"    {k:25s}: {v}")
    print(f"\n  method:               {result.round_diagnostics['method']}")
    budget = result.round_diagnostics.get("privacy_budget_used")
    if budget:
        print(f"  privacy_budget_used:  ε={budget['epsilon']:.4f}, δ={budget['delta']}")


if __name__ == "__main__":
    main()
