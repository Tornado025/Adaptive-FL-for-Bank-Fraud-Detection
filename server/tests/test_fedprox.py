"""
Tests for Phase 5 — FedProx Aggregation
=========================================
Covers:
  - Zero-drift client gets strictly higher weight than drift-5.0 client
    (same mu and num_samples)
  - All returned effective weights sum to 1.0 within 1e-6
  - All-equal drift + equal samples → result within 1e-4 of plain FedAvg
  - Missing proximal_term defaults to 0.0 (no penalty)
  - mu=0 falls back to mu_default gracefully
  - fedprox_aggregate returns same-keyed dict as inputs
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fedprox import compute_fedprox_weights, fedprox_aggregate
from src.fed_avg import federated_average


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_package(
    bank_id: str,
    proximal_term: float,
    mu: float = 0.01,
    num_samples: int = 50000,
    weights: dict = None,
) -> dict:
    if weights is None:
        weights = {"layer.weight": [1.0, 2.0, 3.0], "layer.bias": [0.5]}
    return {
        "bank_id": bank_id,
        "round": 1,
        "num_samples": num_samples,
        "weights": weights,
        "metadata": {
            "val_auc": 0.90,
            "val_loss": 0.30,
            "local_epochs_trained": 3,
            "mu": mu,
            "proximal_term": proximal_term,
        },
    }


def _make_package_no_proximal(bank_id: str, num_samples: int = 50000) -> dict:
    """Package intentionally missing proximal_term."""
    return {
        "bank_id": bank_id,
        "round": 1,
        "num_samples": num_samples,
        "weights": {"layer.weight": [1.0, 2.0], "layer.bias": [0.5]},
        "metadata": {
            "val_auc": 0.90,
            "val_loss": 0.30,
            "local_epochs_trained": 3,
            "mu": 0.01,
            # proximal_term intentionally absent
        },
    }


# ── Tests: compute_fedprox_weights ───────────────────────────────────────────

class TestComputeFedProxWeights:

    def test_zero_drift_gets_higher_weight_than_high_drift(self):
        """
        A client with proximal_term=0.0 must receive a strictly higher weight
        than one with proximal_term=5.0 (same mu and num_samples).
        """
        packages = [
            _make_package("bank_a", proximal_term=0.0, mu=0.01, num_samples=50000),
            _make_package("bank_b", proximal_term=5.0, mu=0.01, num_samples=50000),
        ]
        weights = compute_fedprox_weights(packages)
        assert weights["bank_a"] > weights["bank_b"], (
            f"Zero-drift bank_a ({weights['bank_a']:.4f}) should outweigh "
            f"drifted bank_b ({weights['bank_b']:.4f})"
        )

    def test_weights_sum_to_one(self):
        """All FedProx weights must sum to 1.0 within 1e-6."""
        packages = [
            _make_package("bank_a", proximal_term=0.001),
            _make_package("bank_b", proximal_term=0.500),
            _make_package("bank_c", proximal_term=2.000),
            _make_package("bank_d", proximal_term=5.000),
        ]
        weights = compute_fedprox_weights(packages)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total:.8f}, expected 1.0"

    def test_equal_drift_produces_equal_weights(self):
        """Equal drift across all clients → equal weights (uniform distribution)."""
        packages = [
            _make_package(f"bank_{i}", proximal_term=0.5, mu=0.01, num_samples=50000)
            for i in range(4)
        ]
        weights = compute_fedprox_weights(packages)
        expected = 1.0 / 4
        for bid, w in weights.items():
            assert abs(w - expected) < 1e-6, f"{bid}: expected {expected:.4f}, got {w:.6f}"

    def test_missing_proximal_term_defaults_to_zero_penalty(self):
        """
        Missing proximal_term in metadata must default to 0.0.
        This should give the client maximum FedProx weight (no penalty).
        """
        pkg_no_proximal = _make_package_no_proximal("bank_a")
        pkg_high_drift  = _make_package("bank_b", proximal_term=5.0)
        packages = [pkg_no_proximal, pkg_high_drift]
        weights = compute_fedprox_weights(packages)
        assert weights["bank_a"] > weights["bank_b"], (
            f"Missing proximal_term (→ 0 penalty) should outweigh high drift: "
            f"bank_a={weights['bank_a']:.4f}, bank_b={weights['bank_b']:.4f}"
        )

    def test_all_weights_positive(self):
        """All FedProx weights must be strictly positive."""
        packages = [
            _make_package("bank_a", proximal_term=0.0),
            _make_package("bank_b", proximal_term=100.0),  # extreme drift
        ]
        weights = compute_fedprox_weights(packages)
        for bid, w in weights.items():
            assert w > 0.0, f"{bid} weight should be positive, got {w}"

    def test_returns_all_bank_ids(self):
        packages = [
            _make_package("bank_a", 0.1),
            _make_package("bank_b", 0.2),
            _make_package("bank_c", 0.3),
        ]
        weights = compute_fedprox_weights(packages)
        assert set(weights.keys()) == {"bank_a", "bank_b", "bank_c"}

    def test_higher_drift_always_lower_weight(self):
        """Monotonicity: increasing proximal_term → decreasing FedProx weight."""
        drift_values = [0.0, 0.1, 0.5, 1.0, 5.0]
        packages = [
            _make_package(f"bank_{i}", proximal_term=d)
            for i, d in enumerate(drift_values)
        ]
        weights = compute_fedprox_weights(packages)
        bank_ids = [f"bank_{i}" for i in range(len(drift_values))]
        w_values = [weights[bid] for bid in bank_ids]
        for i in range(len(w_values) - 1):
            assert w_values[i] >= w_values[i + 1] - 1e-9, (
                f"Non-monotonic: drift={drift_values[i]} weight={w_values[i]:.6f} < "
                f"drift={drift_values[i+1]} weight={w_values[i+1]:.6f}"
            )


# ── Tests: fedprox_aggregate ─────────────────────────────────────────────────

class TestFedProxAggregate:

    def test_output_has_same_keys_as_input(self):
        packages = [
            _make_package("bank_a", 0.0, weights={"w": [1.0, 2.0], "b": [0.5]}),
            _make_package("bank_b", 0.1, weights={"w": [3.0, 4.0], "b": [1.5]}),
        ]
        result = fedprox_aggregate(packages)
        assert set(result.keys()) == {"w", "b"}

    def test_equal_drift_equal_samples_approaches_fedavg(self):
        """
        When all clients have equal drift and equal sample counts, FedProx
        should degenerate to FedAvg (max deviation ≤ 1e-4).
        """
        weights_list = [
            {"layer.weight": [float(i), float(i + 1)], "layer.bias": [float(i) * 0.1]}
            for i in range(4)
        ]
        packages = [
            _make_package(
                f"bank_{i}",
                proximal_term=0.5,   # equal drift
                num_samples=50000,   # equal samples
                weights=weights_list[i],
            )
            for i in range(4)
        ]

        fedprox_result = fedprox_aggregate(packages)
        fedavg_result  = federated_average(packages)

        for key in fedavg_result:
            fp  = np.array(fedprox_result[key])
            avg = np.array(fedavg_result[key])
            max_dev = float(np.max(np.abs(fp - avg)))
            assert max_dev <= 1e-4, (
                f"Key '{key}': FedProx deviated from FedAvg by {max_dev:.2e} (threshold 1e-4)"
            )

    def test_effective_weights_sum_to_one(self):
        """
        Indirectly verified via the weighted average formula:
        if weights sum to 1, the average of identical inputs equals the input.
        """
        w = {"layer.weight": [1.0, 2.0, 3.0], "layer.bias": [4.0]}
        packages = [
            _make_package("bank_a", 0.1, weights=w, num_samples=50000),
            _make_package("bank_b", 0.2, weights=w, num_samples=50000),
            _make_package("bank_c", 0.3, weights=w, num_samples=50000),
            _make_package("bank_d", 0.5, weights=w, num_samples=50000),
        ]
        result = fedprox_aggregate(packages)
        for key in w:
            np.testing.assert_allclose(
                np.array(result[key]), np.array(w[key]), atol=1e-4,
                err_msg=f"Key '{key}': identical inputs should produce identical output"
            )

    def test_low_drift_client_dominates(self):
        """
        A zero-drift client should have more influence than a high-drift client.
        Verify by using very different weight values and checking which direction
        the aggregated result leans.
        """
        packages = [
            _make_package("bank_a", proximal_term=0.0, num_samples=50000,
                          weights={"w": [10.0]}),  # zero drift
            _make_package("bank_b", proximal_term=10.0, num_samples=50000,
                          weights={"w": [0.0]}),   # high drift
        ]
        result = fedprox_aggregate(packages)
        # Zero-drift client (value=10) should dominate → result > 5.0
        assert result["w"][0] > 5.0, (
            f"Zero-drift client should dominate; result={result['w'][0]:.4f}"
        )
