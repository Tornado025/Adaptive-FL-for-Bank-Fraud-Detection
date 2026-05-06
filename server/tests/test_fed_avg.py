"""
Tests for Phase 1 — Standard FedAvg
=====================================
Covers:
  - Identical weights from all clients → output must match input within 1e-6
  - One client with 2× the samples → must contribute proportionally more
  - Single-client case → output equals that client's weights exactly
  - Empty input → raises ValueError
  - Mismatched weight keys → raises ValueError
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fed_avg import federated_average, federated_average_with_custom_weights


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_package(bank_id: str, num_samples: int, weights: dict) -> dict:
    return {
        "bank_id": bank_id,
        "round": 1,
        "num_samples": num_samples,
        "weights": weights,
        "metadata": {"val_auc": 0.90, "val_loss": 0.3, "local_epochs_trained": 3,
                     "mu": 0.01, "proximal_term": 0.001},
    }


def _assert_weights_close(result: dict, expected: dict, tol: float = 1e-6):
    assert set(result.keys()) == set(expected.keys()), "Key mismatch"
    for key in expected:
        r = np.array(result[key])
        e = np.array(expected[key])
        assert r.shape == e.shape, f"Shape mismatch for key '{key}'"
        assert np.allclose(r, e, atol=tol), (
            f"Key '{key}': max deviation = {np.max(np.abs(r - e)):.2e}"
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFedAvgIdenticalWeights:
    """With identical weights, the average must equal the input exactly."""

    def test_two_clients_identical(self):
        w = {"layer.weight": [[0.1, 0.2], [0.3, 0.4]], "layer.bias": [0.5, 0.6]}
        packages = [
            _make_package("bank_a", 1000, w),
            _make_package("bank_b", 1000, w),
        ]
        result = federated_average(packages)
        _assert_weights_close(result, w)

    def test_four_clients_identical(self):
        w = {
            "base_layers.0.weight": [[0.1, -0.2, 0.3]] * 4,
            "base_layers.0.bias":   [0.05, -0.05, 0.10, 0.20],
        }
        packages = [_make_package(f"bank_{i}", 5000, w) for i in range(4)]
        result = federated_average(packages)
        _assert_weights_close(result, w)

    def test_non_uniform_sample_counts_identical_weights(self):
        """Even with different sample sizes, identical weights must average to themselves."""
        w = {"layer.weight": [1.0, 2.0, 3.0], "layer.bias": [-1.0]}
        packages = [
            _make_package("bank_a", 10000, w),
            _make_package("bank_b", 50000, w),
            _make_package("bank_c", 3000,  w),
        ]
        result = federated_average(packages)
        _assert_weights_close(result, w)


class TestFedAvgProportionalContribution:
    """A client with 2× samples must contribute proportionally more."""

    def test_double_samples_double_contribution(self):
        # bank_a: weight = [1.0], 1000 samples  → contribution = 1/3
        # bank_b: weight = [4.0], 2000 samples  → contribution = 2/3
        # Expected average = 1/3 * 1.0 + 2/3 * 4.0 = 3.0
        packages = [
            _make_package("bank_a", 1000, {"w": [1.0]}),
            _make_package("bank_b", 2000, {"w": [4.0]}),
        ]
        result = federated_average(packages)
        expected = 1000 / 3000 * 1.0 + 2000 / 3000 * 4.0
        assert abs(result["w"][0] - expected) < 1e-6

    def test_proportional_four_clients(self):
        """Verify the weighted sum formula across four clients."""
        samples = {"bank_a": 80000, "bank_b": 65000, "bank_c": 95000, "bank_d": 50000}
        weights = {"bank_a": [1.0], "bank_b": [2.0], "bank_c": [3.0], "bank_d": [4.0]}
        total = sum(samples.values())
        expected = sum(samples[b] / total * weights[b][0] for b in samples)

        packages = [_make_package(b, samples[b], {"w": weights[b]}) for b in samples]
        result = federated_average(packages)
        assert abs(result["w"][0] - expected) < 1e-6


class TestFedAvgSingleClient:
    """Single client → output must equal that client's weights exactly."""

    def test_single_client(self):
        w = {"layer.weight": [[0.5, 0.6], [-0.1, 0.2]], "layer.bias": [0.3, -0.4]}
        packages = [_make_package("bank_a", 50000, w)]
        result = federated_average(packages)
        _assert_weights_close(result, w)


class TestFedAvgEdgeCases:
    """Error handling for malformed inputs."""

    def test_empty_packages_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            federated_average([])

    def test_mismatched_keys_raises(self):
        packages = [
            _make_package("bank_a", 1000, {"w1": [1.0], "w2": [2.0]}),
            _make_package("bank_b", 1000, {"w1": [1.0], "w3": [3.0]}),  # different key
        ]
        with pytest.raises(ValueError, match="Mismatched weight keys"):
            federated_average(packages)


class TestFedAvgWithCustomWeights:
    """Tests for the generalised weighted average helper."""

    def test_uniform_effective_weights_matches_fedavg(self):
        w = {"layer.weight": [[1.0, 2.0], [3.0, 4.0]], "layer.bias": [0.5]}
        packages = [
            _make_package("bank_a", 500, w),
            _make_package("bank_b", 500, w),
        ]
        eff = {"bank_a": 0.5, "bank_b": 0.5}
        result = federated_average_with_custom_weights(packages, eff)
        _assert_weights_close(result, w)

    def test_custom_weights_applied_correctly(self):
        packages = [
            _make_package("bank_a", 100, {"w": [0.0]}),
            _make_package("bank_b", 100, {"w": [10.0]}),
        ]
        # Give bank_b 100% weight
        result = federated_average_with_custom_weights(packages, {"bank_a": 0.0, "bank_b": 1.0})
        assert abs(result["w"][0] - 10.0) < 1e-6
