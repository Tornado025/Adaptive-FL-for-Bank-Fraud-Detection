"""
Tests for Phase 6 — Master Aggregator
=======================================
Covers:
  - method="fedavg" → result matches federated_average() directly within 1e-6
  - method="custom" → returns AggregationResult with both fields populated
  - round_diagnostics contains all four bank IDs with required sub-keys
  - dp_config provided → dp_applied=True for all clients in diagnostics
  - No client's effective_weight falls below MIN_WEIGHT_FLOOR (0.05)
  - All effective_weights sum to 1.0 within 1e-6
  - Unknown method raises ValueError
  - Empty weight_packages raises ValueError
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aggregator import aggregate, AggregationResult, MIN_WEIGHT_FLOOR
from src.fed_avg import federated_average

BANK_IDS = ["bank_a", "bank_b", "bank_c", "bank_d"]
REQUIRED_DIAG_KEYS = {
    "val_auc", "reliability_score", "conflict_penalty",
    "fedprox_weight", "sample_fraction", "effective_weight", "dp_applied",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_package(
    bank_id: str,
    val_auc: float = 0.90,
    proximal_term: float = 0.002,
    num_samples: int = 60000,
    weights: dict = None,
) -> dict:
    if weights is None:
        weights = {
            "layer.weight": [[0.1, 0.2], [0.3, 0.4]],
            "layer.bias":   [0.5, 0.6],
        }
    return {
        "bank_id": bank_id,
        "round": 1,
        "num_samples": num_samples,
        "weights": weights,
        "metadata": {
            "val_loss": 0.30,
            "val_auc": val_auc,
            "local_epochs_trained": 3,
            "mu": 0.01,
            "proximal_term": proximal_term,
        },
    }


def _make_four_packages(**kwargs) -> list[dict]:
    return [_make_package(bid, **kwargs) for bid in BANK_IDS]


_GLOBAL_WEIGHTS = {
    "layer.weight": [[0.0, 0.0], [0.0, 0.0]],
    "layer.bias":   [0.0, 0.0],
}

_DP_CONFIG = {"clip_norm": 1.0, "noise_multiplier": 1.1}


# ── Tests: method="fedavg" ────────────────────────────────────────────────────

class TestAggregateMethodFedAvg:
    """FedAvg method must produce results identical to federated_average()."""

    def test_matches_federated_average_identical_weights(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="fedavg")
        direct = federated_average(packages)

        assert isinstance(result, AggregationResult)
        for key in direct:
            np.testing.assert_allclose(
                np.array(result.new_global_weights[key]),
                np.array(direct[key]),
                atol=1e-6,
                err_msg=f"FedAvg mismatch on key '{key}'",
            )

    def test_matches_federated_average_different_weights(self):
        packages = [
            _make_package("bank_a", weights={"w": [1.0, 2.0]}, num_samples=80000),
            _make_package("bank_b", weights={"w": [3.0, 4.0]}, num_samples=40000),
            _make_package("bank_c", weights={"w": [5.0, 6.0]}, num_samples=60000),
            _make_package("bank_d", weights={"w": [7.0, 8.0]}, num_samples=20000),
        ]
        global_w = {"w": [0.0, 0.0]}
        result = aggregate(packages, global_w, method="fedavg")
        direct = federated_average(packages)
        np.testing.assert_allclose(
            np.array(result.new_global_weights["w"]),
            np.array(direct["w"]),
            atol=1e-6,
        )

    def test_fedavg_no_dp_applied(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="fedavg")
        for bid in BANK_IDS:
            assert result.round_diagnostics["per_client"][bid]["dp_applied"] is False


# ── Tests: method="custom" ───────────────────────────────────────────────────

class TestAggregateMethodCustom:

    def test_returns_aggregation_result(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="custom", dp_config=_DP_CONFIG)
        assert isinstance(result, AggregationResult)

    def test_new_global_weights_populated(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="custom", dp_config=_DP_CONFIG)
        assert isinstance(result.new_global_weights, dict)
        assert set(result.new_global_weights.keys()) == {"layer.weight", "layer.bias"}

    def test_round_diagnostics_populated(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="custom", dp_config=_DP_CONFIG)
        assert isinstance(result.round_diagnostics, dict)
        assert "per_client" in result.round_diagnostics
        assert "method" in result.round_diagnostics

    def test_all_bank_ids_in_diagnostics(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="custom", dp_config=_DP_CONFIG)
        per_client = result.round_diagnostics["per_client"]
        for bid in BANK_IDS:
            assert bid in per_client, f"Missing bank '{bid}' in round_diagnostics"

    def test_all_required_diagnostic_keys_present(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="custom", dp_config=_DP_CONFIG)
        per_client = result.round_diagnostics["per_client"]
        for bid in BANK_IDS:
            missing = REQUIRED_DIAG_KEYS - set(per_client[bid].keys())
            assert not missing, f"bank '{bid}' diagnostics missing keys: {missing}"

    def test_dp_applied_true_when_dp_config_provided(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="custom", dp_config=_DP_CONFIG)
        per_client = result.round_diagnostics["per_client"]
        for bid in BANK_IDS:
            assert per_client[bid]["dp_applied"] is True, (
                f"{bid}: dp_applied should be True when dp_config is provided"
            )

    def test_dp_applied_false_for_fedavg(self):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="fedavg")
        per_client = result.round_diagnostics["per_client"]
        for bid in BANK_IDS:
            assert per_client[bid]["dp_applied"] is False


# ── Tests: weight floor ────────────────────────────────────────────────────────

class TestWeightFloor:
    """No client's effective_weight may fall below MIN_WEIGHT_FLOOR."""

    def test_floor_enforced_custom_method(self):
        """Even a very poorly-scored client must get ≥ MIN_WEIGHT_FLOOR."""
        packages = [
            _make_package("bank_a", val_auc=0.99, proximal_term=0.0, num_samples=200000),
            _make_package("bank_b", val_auc=0.99, proximal_term=0.0, num_samples=200000),
            _make_package("bank_c", val_auc=0.99, proximal_term=0.0, num_samples=200000),
            _make_package("bank_d", val_auc=0.40, proximal_term=50.0, num_samples=100),  # terrible
        ]
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="custom", dp_config=_DP_CONFIG)
        per_client = result.round_diagnostics["per_client"]
        for bid in BANK_IDS:
            eff = per_client[bid]["effective_weight"]
            assert eff >= MIN_WEIGHT_FLOOR - 1e-9, (
                f"{bid}: effective_weight {eff:.6f} < floor {MIN_WEIGHT_FLOOR}"
            )

    def test_floor_enforced_fedprox_method(self):
        packages = [
            _make_package("bank_a", proximal_term=0.0, num_samples=500000),
            _make_package("bank_b", proximal_term=0.0, num_samples=500000),
            _make_package("bank_c", proximal_term=0.0, num_samples=500000),
            _make_package("bank_d", proximal_term=1000.0, num_samples=10),
        ]
        result = aggregate(packages, _GLOBAL_WEIGHTS, method="fedprox")
        per_client = result.round_diagnostics["per_client"]
        for bid in BANK_IDS:
            eff = per_client[bid]["effective_weight"]
            assert eff >= MIN_WEIGHT_FLOOR - 1e-9, (
                f"{bid}: effective_weight {eff:.6f} < floor {MIN_WEIGHT_FLOOR}"
            )


# ── Tests: effective weights sum ──────────────────────────────────────────────

class TestEffectiveWeightSum:
    """Effective weights must sum to 1.0 within 1e-6 for all methods."""

    @pytest.mark.parametrize("method,dp", [
        ("fedavg",    None),
        ("fedprox",   None),
        ("dp_fedavg", _DP_CONFIG),
        ("custom",    _DP_CONFIG),
    ])
    def test_weights_sum_to_one(self, method, dp):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method=method, dp_config=dp)
        per_client = result.round_diagnostics["per_client"]
        total = sum(per_client[bid]["effective_weight"] for bid in BANK_IDS)
        assert abs(total - 1.0) < 1e-6, (
            f"method={method}: effective weights sum to {total:.8f}, expected 1.0"
        )


# ── Tests: effective weight bounds ────────────────────────────────────────────

class TestEffectiveWeightBounds:
    """All effective weights must be in [0.0, 1.0]."""

    @pytest.mark.parametrize("method,dp", [
        ("fedavg",    None),
        ("fedprox",   None),
        ("dp_fedavg", _DP_CONFIG),
        ("custom",    _DP_CONFIG),
    ])
    def test_weights_in_unit_interval(self, method, dp):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method=method, dp_config=dp)
        per_client = result.round_diagnostics["per_client"]
        for bid in BANK_IDS:
            eff = per_client[bid]["effective_weight"]
            assert 0.0 <= eff <= 1.0 + 1e-9, (
                f"method={method}, {bid}: effective_weight {eff:.6f} out of [0,1]"
            )


# ── Tests: error handling ─────────────────────────────────────────────────────

class TestAggregateErrorHandling:

    def test_unknown_method_raises_value_error(self):
        packages = _make_four_packages()
        with pytest.raises(ValueError, match="Unknown method"):
            aggregate(packages, _GLOBAL_WEIGHTS, method="nonexistent_method")

    def test_empty_packages_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            aggregate([], _GLOBAL_WEIGHTS, method="fedavg")


# ── Tests: method diagnostics field ──────────────────────────────────────────

class TestDiagnosticsMethod:

    @pytest.mark.parametrize("method,dp", [
        ("fedavg",    None),
        ("fedprox",   None),
        ("dp_fedavg", _DP_CONFIG),
        ("custom",    _DP_CONFIG),
    ])
    def test_method_recorded_in_diagnostics(self, method, dp):
        packages = _make_four_packages()
        result = aggregate(packages, _GLOBAL_WEIGHTS, method=method, dp_config=dp)
        assert result.round_diagnostics["method"] == method
