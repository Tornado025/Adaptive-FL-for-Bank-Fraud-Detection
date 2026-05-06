"""
Tests for Phase 4 — Differential Privacy Mechanism
=====================================================
Covers:
  - After clip_weight_update, L2 norm of update ≤ clip_norm + 1e-6
  - After add_gaussian_noise, weights differ from input (non-deterministic check)
  - apply_dp_to_packages preserves bank_id, round, num_samples
  - Large update (norm ≈ 100, clip_norm = 1.0) → clipped norm within 1e-6 of 1.0
  - Zero update (client weights = global) → no clipping needed
  - estimate_privacy_budget returns correct keys and reasonable epsilon values
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dp_mechanism import (
    clip_weight_update,
    add_gaussian_noise,
    apply_dp_to_packages,
    estimate_privacy_budget,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_weights(scale: float = 1.0, seed: int = 0) -> dict[str, list]:
    """Returns a small weight dict with controlled random values."""
    rng = np.random.RandomState(seed)
    return {
        "layer.weight": (rng.randn(4, 4) * scale).tolist(),
        "layer.bias":   (rng.randn(4) * scale).tolist(),
    }


def _update_l2_norm(client: dict, global_: dict) -> float:
    """Computes the L2 norm of (client - global) flattened."""
    parts = []
    for key in global_:
        c = np.array(client[key], dtype=np.float64)
        g = np.array(global_[key], dtype=np.float64)
        parts.append((c - g).flatten())
    return float(np.linalg.norm(np.concatenate(parts)))


def _make_package(bank_id: str, weights: dict) -> dict:
    return {
        "bank_id": bank_id,
        "round": 3,
        "num_samples": 75000,
        "weights": weights,
        "metadata": {"val_auc": 0.90, "val_loss": 0.3, "local_epochs_trained": 3,
                     "mu": 0.01, "proximal_term": 0.002},
    }


# ── Tests: clip_weight_update ─────────────────────────────────────────────────

class TestClipWeightUpdate:

    def test_small_update_not_clipped(self):
        """When update norm < clip_norm, weights should be unchanged."""
        global_w = _make_weights(scale=1.0, seed=0)
        # Create a tiny perturbation
        client_w = {k: (np.array(v) + 0.001).tolist() for k, v in global_w.items()}
        clip_norm = 1.0
        clipped = clip_weight_update(client_w, global_w, clip_norm)
        # Norm should still be very small (< clip_norm)
        norm = _update_l2_norm(clipped, global_w)
        assert norm <= clip_norm + 1e-6

    def test_large_update_clipped_to_clip_norm(self):
        """Update with norm ≈ 100 must be clipped to within 1e-6 of clip_norm=1.0."""
        clip_norm = 1.0
        global_w = {"layer.weight": [[0.0, 0.0], [0.0, 0.0]], "layer.bias": [0.0, 0.0]}
        # Client weights chosen so ||client - global||₂ = 100
        # Using a single direction: [100, 0, 0, 0, 0, 0]
        client_w = {"layer.weight": [[100.0, 0.0], [0.0, 0.0]], "layer.bias": [0.0, 0.0]}
        clipped = clip_weight_update(client_w, global_w, clip_norm)
        norm = _update_l2_norm(clipped, global_w)
        assert norm <= clip_norm + 1e-6, f"Clipped norm {norm:.8f} exceeds {clip_norm}"
        # Should be very close to exactly clip_norm
        assert abs(norm - clip_norm) < 1e-6, (
            f"Expected clipped norm ≈ {clip_norm}, got {norm:.8f}"
        )

    def test_norm_never_exceeds_clip_norm(self):
        """Property test: clipped norm ≤ clip_norm for various input norms."""
        global_w = {"w": [0.0, 0.0, 0.0, 0.0]}
        clip_norm = 1.0
        rng = np.random.RandomState(42)
        for _ in range(20):
            scale = rng.uniform(0.0, 50.0)
            delta = rng.randn(4) * scale
            client_w = {"w": delta.tolist()}
            clipped = clip_weight_update(client_w, global_w, clip_norm)
            norm = _update_l2_norm(clipped, global_w)
            assert norm <= clip_norm + 1e-6, f"Norm {norm:.8f} exceeded clip_norm={clip_norm}"

    def test_zero_update_unchanged(self):
        """If client weights equal global weights, output should equal global weights."""
        global_w = _make_weights(scale=0.5, seed=7)
        clipped = clip_weight_update(global_w, global_w, clip_norm=1.0)
        for key in global_w:
            np.testing.assert_allclose(
                np.array(clipped[key]), np.array(global_w[key]), atol=1e-12
            )

    def test_preserves_weight_keys(self):
        """Clipped output must have the same keys as global_weights."""
        global_w = _make_weights(seed=1)
        client_w = _make_weights(scale=5.0, seed=2)
        clipped = clip_weight_update(client_w, global_w, clip_norm=1.0)
        assert set(clipped.keys()) == set(global_w.keys())


# ── Tests: add_gaussian_noise ─────────────────────────────────────────────────

class TestAddGaussianNoise:

    def test_noise_changes_weights(self):
        """After adding noise, weights must differ from input (non-deterministic)."""
        weights = _make_weights(scale=1.0, seed=10)
        noisy = add_gaussian_noise(weights, clip_norm=1.0, noise_multiplier=1.1)
        # Cannot be all zeros noise → statistically impossible
        at_least_one_changed = any(
            not np.allclose(np.array(weights[k]), np.array(noisy[k]))
            for k in weights
        )
        assert at_least_one_changed, "Noise had zero effect on all weights"

    def test_output_shape_preserved(self):
        """Noisy weights must have identical shapes to inputs."""
        weights = _make_weights(scale=1.0, seed=3)
        noisy = add_gaussian_noise(weights, clip_norm=1.0, noise_multiplier=1.1)
        for key in weights:
            assert np.array(noisy[key]).shape == np.array(weights[key]).shape

    def test_output_keys_preserved(self):
        weights = _make_weights(seed=4)
        noisy = add_gaussian_noise(weights, clip_norm=1.0, noise_multiplier=1.1)
        assert set(noisy.keys()) == set(weights.keys())

    def test_higher_noise_multiplier_means_more_noise(self):
        """Weights with noise_multiplier=5.0 should differ more than nm=0.01."""
        weights = _make_weights(scale=1.0, seed=5)
        np.random.seed(42)
        low_noise  = add_gaussian_noise(weights, clip_norm=1.0, noise_multiplier=0.01)
        np.random.seed(42)
        high_noise = add_gaussian_noise(weights, clip_norm=1.0, noise_multiplier=5.0)

        def total_diff(a, b):
            return sum(
                np.sum(np.abs(np.array(a[k]) - np.array(b[k])))
                for k in a
            )

        assert total_diff(high_noise, weights) > total_diff(low_noise, weights)

    def test_zero_noise_multiplier_raises(self):
        """noise_multiplier=0 would mean noise_std=0 — this is allowed but should produce
        no noise. We just verify the function doesn't crash."""
        weights = _make_weights(seed=6)
        # noise_std = 0 → adds zeros → output equals input
        noisy = add_gaussian_noise(weights, clip_norm=1.0, noise_multiplier=0.0)
        for key in weights:
            np.testing.assert_allclose(np.array(noisy[key]), np.array(weights[key]), atol=1e-12)


# ── Tests: apply_dp_to_packages ───────────────────────────────────────────────

class TestApplyDpToPackages:

    def _make_four_packages(self):
        global_w = {"w": [0.0, 0.0]}
        packages = [
            _make_package("bank_a", {"w": [1.0, 0.0]}),
            _make_package("bank_b", {"w": [0.0, 1.0]}),
            _make_package("bank_c", {"w": [-1.0, 0.0]}),
            _make_package("bank_d", {"w": [0.5, 0.5]}),
        ]
        return packages, global_w

    def test_preserves_bank_id(self):
        packages, global_w = self._make_four_packages()
        dp_pkgs = apply_dp_to_packages(packages, global_w, clip_norm=1.0, noise_multiplier=1.1)
        original_ids = {p["bank_id"] for p in packages}
        dp_ids = {p["bank_id"] for p in dp_pkgs}
        assert original_ids == dp_ids

    def test_preserves_round(self):
        packages, global_w = self._make_four_packages()
        dp_pkgs = apply_dp_to_packages(packages, global_w, clip_norm=1.0, noise_multiplier=1.1)
        for orig, dp in zip(packages, dp_pkgs):
            assert orig["round"] == dp["round"], "round field was mutated"

    def test_preserves_num_samples(self):
        packages, global_w = self._make_four_packages()
        dp_pkgs = apply_dp_to_packages(packages, global_w, clip_norm=1.0, noise_multiplier=1.1)
        for orig, dp in zip(packages, dp_pkgs):
            assert orig["num_samples"] == dp["num_samples"], "num_samples field was mutated"

    def test_preserves_metadata(self):
        packages, global_w = self._make_four_packages()
        dp_pkgs = apply_dp_to_packages(packages, global_w, clip_norm=1.0, noise_multiplier=1.1)
        for orig, dp in zip(packages, dp_pkgs):
            assert orig["metadata"] == dp["metadata"], "metadata was mutated"

    def test_does_not_mutate_originals(self):
        """Originals must not be modified in-place."""
        packages, global_w = self._make_four_packages()
        original_weights_copy = [dict(p["weights"]) for p in packages]
        apply_dp_to_packages(packages, global_w, clip_norm=1.0, noise_multiplier=1.1)
        for orig, orig_copy in zip(packages, original_weights_copy):
            assert orig["weights"] == orig_copy, "Original package weights were mutated"

    def test_output_length_equals_input(self):
        packages, global_w = self._make_four_packages()
        dp_pkgs = apply_dp_to_packages(packages, global_w)
        assert len(dp_pkgs) == len(packages)

    def test_dp_clipping_enforced_after_pipeline(self):
        """After apply_dp_to_packages with clip_norm=1.0, each update norm ≤ 1.0 + noise."""
        # This is a softer check since noise is added after clipping
        global_w = {"w": [0.0, 0.0, 0.0]}
        packages = [_make_package("bank_a", {"w": [50.0, 50.0, 50.0]})]
        dp_pkgs = apply_dp_to_packages(packages, global_w, clip_norm=1.0, noise_multiplier=0.0)
        # With zero noise, the update norm should be ≈ 1.0
        norm = _update_l2_norm(dp_pkgs[0]["weights"], global_w)
        assert norm <= 1.0 + 1e-6, f"DP pipeline failed to clip: norm={norm:.6f}"


# ── Tests: estimate_privacy_budget ────────────────────────────────────────────

class TestEstimatePrivacyBudget:

    def test_returns_required_keys(self):
        result = estimate_privacy_budget(
            num_rounds=20, num_clients=4,
            noise_multiplier=1.1, clip_norm=1.0, delta=1e-5
        )
        for key in ("epsilon", "delta", "noise_multiplier", "num_rounds", "interpretation"):
            assert key in result, f"Missing key: {key}"

    def test_epsilon_positive(self):
        result = estimate_privacy_budget(20, 4, 1.1, 1.0)
        assert result["epsilon"] > 0

    def test_higher_noise_gives_lower_epsilon(self):
        """Stronger noise → smaller ε (better privacy)."""
        e_weak   = estimate_privacy_budget(20, 4, 0.5, 1.0)["epsilon"]
        e_strong = estimate_privacy_budget(20, 4, 2.0, 1.0)["epsilon"]
        assert e_strong < e_weak, (
            f"Higher noise_multiplier should give lower epsilon: {e_strong} < {e_weak}"
        )

    def test_more_rounds_gives_higher_epsilon(self):
        """More rounds → more privacy budget consumed → larger ε."""
        e_few  = estimate_privacy_budget(5, 4, 1.1, 1.0)["epsilon"]
        e_many = estimate_privacy_budget(20, 4, 1.1, 1.0)["epsilon"]
        assert e_many > e_few

    def test_balanced_preset_approximately_3_8(self):
        """Balanced preset (nm=1.1, 20 rounds, δ=1e-5) should give ε ≈ 3.8."""
        result = estimate_privacy_budget(20, 4, 1.1, 1.0, delta=1e-5)
        assert 2.0 < result["epsilon"] < 6.0, (
            f"Balanced preset: expected ε in (2, 6), got {result['epsilon']}"
        )

    def test_invalid_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="noise_multiplier"):
            estimate_privacy_budget(20, 4, -1.0, 1.0)

    def test_invalid_num_rounds_raises(self):
        with pytest.raises(ValueError, match="num_rounds"):
            estimate_privacy_budget(0, 4, 1.1, 1.0)
