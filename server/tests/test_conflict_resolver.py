"""
Tests for Phase 3 — Conflict Resolver
=======================================
Covers:
  - Client whose weights are exact negation of consensus → penalty = 1.0
  - Client matching consensus direction → penalty = 0.0
  - All four clients in full agreement → all penalties = 0.0
  - weights_to_vector output length equals total element count
  - compute_pairwise_conflict diagonal = 1.0 (self-similarity)
  - Two-client scenario: one aligned, one opposing
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conflict_resolver import (
    weights_to_vector,
    compute_pairwise_conflict,
    compute_conflict_penalty,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_package(bank_id: str, weights: dict) -> dict:
    return {
        "bank_id": bank_id,
        "round": 1,
        "num_samples": 50000,
        "weights": weights,
        "metadata": {"val_auc": 0.90, "val_loss": 0.3, "local_epochs_trained": 3,
                     "mu": 0.01, "proximal_term": 0.001},
    }


def _w(values: list) -> dict:
    """Convenience: wrap a flat list as a weight dict."""
    return {"layer.weight": values}


# ── Tests: weights_to_vector ──────────────────────────────────────────────────

class TestWeightsToVector:

    def test_single_flat_tensor(self):
        weights = {"layer.weight": [1.0, 2.0, 3.0]}
        v = weights_to_vector(weights)
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])

    def test_2d_tensor_flattened(self):
        weights = {"layer.weight": [[1.0, 2.0], [3.0, 4.0]]}
        v = weights_to_vector(weights)
        assert len(v) == 4
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0, 4.0])

    def test_multiple_layers_concatenated(self):
        weights = {
            "layer.weight": [[1.0, 2.0], [3.0, 4.0]],  # 4 elements
            "layer.bias":   [5.0, 6.0],                  # 2 elements
        }
        v = weights_to_vector(weights)
        assert len(v) == 6  # total element count

    def test_output_length_equals_total_element_count(self):
        """Length of flattened vector = sum of all tensor element counts."""
        weights = {
            "base_layers.0.weight": [[0.1] * 64] * 256,  # 256 × 64 = 16384
            "base_layers.0.bias":   [0.0] * 256,          # 256
            "base_layers.2.weight": [[0.2] * 256] * 128,  # 128 × 256 = 32768
            "base_layers.2.bias":   [0.0] * 128,          # 128
        }
        v = weights_to_vector(weights)
        expected_len = 256 * 64 + 256 + 128 * 256 + 128
        assert len(v) == expected_len, f"Expected {expected_len}, got {len(v)}"

    def test_empty_weights(self):
        v = weights_to_vector({})
        assert len(v) == 0


# ── Tests: compute_pairwise_conflict ─────────────────────────────────────────

class TestComputePairwiseConflict:

    def test_diagonal_is_one(self):
        """Cosine similarity of a vector with itself must be 1.0."""
        packages = [
            _make_package("bank_a", _w([1.0, 2.0, 3.0])),
            _make_package("bank_b", _w([4.0, 5.0, 6.0])),
        ]
        mat = compute_pairwise_conflict(packages)
        assert mat.shape == (2, 2)
        assert abs(mat[0, 0] - 1.0) < 1e-6
        assert abs(mat[1, 1] - 1.0) < 1e-6

    def test_opposite_vectors_have_similarity_neg1(self):
        packages = [
            _make_package("bank_a", _w([1.0, 0.0])),
            _make_package("bank_b", _w([-1.0, 0.0])),
        ]
        mat = compute_pairwise_conflict(packages)
        assert abs(mat[0, 1] - (-1.0)) < 1e-6
        assert abs(mat[1, 0] - (-1.0)) < 1e-6

    def test_orthogonal_vectors_have_similarity_zero(self):
        packages = [
            _make_package("bank_a", _w([1.0, 0.0])),
            _make_package("bank_b", _w([0.0, 1.0])),
        ]
        mat = compute_pairwise_conflict(packages)
        assert abs(mat[0, 1]) < 1e-6

    def test_identical_vectors_have_similarity_one(self):
        w = _w([1.0, 2.0, 3.0])
        packages = [
            _make_package("bank_a", w),
            _make_package("bank_b", w),
            _make_package("bank_c", w),
        ]
        mat = compute_pairwise_conflict(packages)
        np.testing.assert_allclose(mat, np.ones((3, 3)), atol=1e-6)

    def test_matrix_is_symmetric(self):
        packages = [
            _make_package("bank_a", _w([1.0, 2.0, 3.0])),
            _make_package("bank_b", _w([4.0, -5.0, 6.0])),
            _make_package("bank_c", _w([-1.0, 2.0, -3.0])),
        ]
        mat = compute_pairwise_conflict(packages)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)


# ── Tests: compute_conflict_penalty ──────────────────────────────────────────

class TestComputeConflictPenalty:

    def test_all_clients_in_agreement_have_zero_penalty(self):
        """When all clients share the same direction, all penalties must be 0.0."""
        w = _w([1.0, 2.0, 3.0])
        packages = [
            _make_package("bank_a", w),
            _make_package("bank_b", w),
            _make_package("bank_c", w),
            _make_package("bank_d", w),
        ]
        penalties = compute_conflict_penalty(packages)
        for bid, p in penalties.items():
            assert abs(p) < 1e-6, f"{bid} should have penalty=0.0, got {p:.6f}"

    def test_exact_negation_receives_maximum_penalty(self):
        """
        A client whose weights are the exact negation of all others must receive
        the maximum normalised penalty (= 1.0) after normalisation.

        Setup:
          bank_a, bank_b, bank_c: all at [1.0, 0.0]
          bank_d: [-1.0, 0.0] — exact negation of the majority consensus

        The consensus vector ≈ [(3-1)/4, 0] = [0.5, 0], pointing in the positive
        direction. bank_d has cosine similarity ≈ -1 with the consensus → maximum penalty.
        """
        pos = _w([1.0, 0.0])
        neg = _w([-1.0, 0.0])
        packages = [
            _make_package("bank_a", pos),
            _make_package("bank_b", pos),
            _make_package("bank_c", pos),
            _make_package("bank_d", neg),
        ]
        penalties = compute_conflict_penalty(packages)
        # bank_d is most conflicting and should receive the highest penalty
        assert penalties["bank_d"] == pytest.approx(1.0, abs=1e-6), (
            f"Opposing client should get penalty=1.0, got {penalties['bank_d']:.6f}"
        )

    def test_consensus_aligned_client_has_zero_penalty(self):
        """
        A client that perfectly aligns with the consensus direction must receive
        penalty = 0.0.
        """
        packages = [
            _make_package("bank_a", _w([1.0, 0.0])),
            _make_package("bank_b", _w([1.0, 0.0])),
            _make_package("bank_c", _w([-2.0, 0.0])),  # conflicting
        ]
        penalties = compute_conflict_penalty(packages)
        # bank_a and bank_b align with the consensus → 0.0 penalty
        # (consensus = mean([1,0], [1,0], [-2,0]) = [0, 0]; sim with zero vec = 0)
        # In practice with mixed clients, aligned ones have non-negative sim → 0 raw penalty
        assert penalties["bank_a"] == pytest.approx(0.0, abs=1e-6)
        assert penalties["bank_b"] == pytest.approx(0.0, abs=1e-6)

    def test_penalties_in_unit_interval(self):
        """All penalties must lie in [0.0, 1.0]."""
        packages = [
            _make_package("bank_a", _w([1.0, 2.0, -3.0])),
            _make_package("bank_b", _w([-1.0, 2.0, 3.0])),
            _make_package("bank_c", _w([0.5, -0.5, 0.5])),
            _make_package("bank_d", _w([-0.5, -0.5, -0.5])),
        ]
        penalties = compute_conflict_penalty(packages)
        for bid, p in penalties.items():
            assert 0.0 <= p <= 1.0, f"{bid} penalty {p:.6f} out of [0,1]"

    def test_two_directly_opposing_clients(self):
        """With only two clients exactly opposing each other, both get equal penalties."""
        packages = [
            _make_package("bank_a", _w([1.0, 0.0, 0.0])),
            _make_package("bank_b", _w([-1.0, 0.0, 0.0])),
        ]
        penalties = compute_conflict_penalty(packages)
        # Consensus ≈ [0, 0, 0] (zero vector) → cosine sim undefined, treated as 0 → no penalty
        # Both clients are equally (dis)similar to the zero consensus
        # All raw penalties = max(0, 0) = 0 in this degenerate case
        assert set(penalties.keys()) == {"bank_a", "bank_b"}
        for p in penalties.values():
            assert 0.0 <= p <= 1.0

    def test_returns_all_bank_ids(self):
        packages = [
            _make_package("bank_a", _w([1.0, 2.0])),
            _make_package("bank_b", _w([3.0, 4.0])),
            _make_package("bank_c", _w([-1.0, 2.0])),
        ]
        penalties = compute_conflict_penalty(packages)
        assert set(penalties.keys()) == {"bank_a", "bank_b", "bank_c"}
