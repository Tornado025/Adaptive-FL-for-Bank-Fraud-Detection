"""
Tests for Phase 2 — Reliability Scorer
========================================
Covers:
  - val_auc = 0.85 → score within 0.01 of 0.5
  - Higher val_auc → strictly higher score (monotonicity)
  - All scores in [0.0, 1.0]
  - Missing val_auc in metadata → raises KeyError (documented design choice)
  - score_all_clients returns correct bank_id mapping
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reliability_scorer import (
    score_client_reliability,
    score_all_clients,
    DEFAULT_THRESHOLD,
    DEFAULT_SENSITIVITY,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_package(bank_id: str, val_auc: float) -> dict:
    return {
        "bank_id": bank_id,
        "round": 1,
        "num_samples": 50000,
        "weights": {},
        "metadata": {
            "val_auc": val_auc,
            "val_loss": 0.3,
            "local_epochs_trained": 3,
            "mu": 0.01,
            "proximal_term": 0.001,
        },
    }


def _make_package_no_auc(bank_id: str) -> dict:
    """Package intentionally missing val_auc to test KeyError handling."""
    return {
        "bank_id": bank_id,
        "round": 1,
        "num_samples": 50000,
        "weights": {},
        "metadata": {
            "val_loss": 0.3,
            "local_epochs_trained": 3,
            "mu": 0.01,
            "proximal_term": 0.001,
            # val_auc intentionally missing
        },
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestScoreClientReliability:
    """Tests for the sigmoid-shaped score_client_reliability function."""

    def test_threshold_auc_returns_neutral_score(self):
        """val_auc = 0.85 (threshold) must return score within 0.01 of 0.5."""
        pkg = _make_package("bank_a", 0.85)
        score = score_client_reliability(pkg)
        assert abs(score - 0.5) <= 0.01, f"Expected ≈0.5, got {score:.4f}"

    def test_high_auc_returns_high_score(self):
        """val_auc = 0.93 should return a high reliability score (≈0.88)."""
        pkg = _make_package("bank_a", 0.93)
        score = score_client_reliability(pkg)
        # sigmoid((0.93 - 0.85) * 10) = sigmoid(0.8) ≈ 0.6900
        assert score > 0.6, f"Expected > 0.6 for val_auc=0.93, got {score:.4f}"

    def test_low_auc_returns_low_score(self):
        """val_auc = 0.70 should return a low reliability score (≈0.18)."""
        pkg = _make_package("bank_a", 0.70)
        score = score_client_reliability(pkg)
        # sigmoid((0.70 - 0.85) * 10) = sigmoid(-1.5) ≈ 0.182
        assert score < 0.3, f"Expected < 0.3 for val_auc=0.70, got {score:.4f}"

    def test_monotonicity(self):
        """Higher val_auc must always yield a strictly higher score."""
        aucs = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
        scores = [score_client_reliability(_make_package("bank_a", a)) for a in aucs]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Non-monotonic: score({aucs[i]})={scores[i]:.4f} >= "
                f"score({aucs[i+1]})={scores[i+1]:.4f}"
            )

    def test_scores_bounded_in_unit_interval(self):
        """All scores must lie in [0.0, 1.0]."""
        test_aucs = [0.0, 0.01, 0.5, 0.85, 0.95, 0.99, 1.0]
        for auc in test_aucs:
            score = score_client_reliability(_make_package("bank_a", auc))
            assert 0.0 <= score <= 1.0, f"Score {score:.4f} out of [0,1] for auc={auc}"

    def test_custom_threshold_and_sensitivity(self):
        """Custom threshold=0.90, sensitivity=20 shifts the neutral point."""
        pkg = _make_package("bank_a", 0.90)
        score = score_client_reliability(pkg, threshold=0.90, sensitivity_factor=20)
        assert abs(score - 0.5) <= 0.01, f"Expected ≈0.5 for custom threshold=0.90, got {score:.4f}"

    def test_missing_val_auc_raises_key_error(self):
        """
        A package missing val_auc raises KeyError.
        Design choice: no silent fallback — malformed packages from Track 2
        should be caught at ingestion rather than silently trusted.
        """
        pkg = _make_package_no_auc("bank_a")
        with pytest.raises(KeyError):
            score_client_reliability(pkg)

    def test_missing_metadata_raises_key_error(self):
        """A package entirely missing the metadata key raises KeyError."""
        pkg = {
            "bank_id": "bank_a",
            "round": 1,
            "num_samples": 50000,
            "weights": {},
            # metadata key missing entirely
        }
        with pytest.raises(KeyError):
            score_client_reliability(pkg)


class TestScoreAllClients:
    """Tests for the batch scoring function."""

    def test_returns_correct_bank_ids(self):
        packages = [
            _make_package("bank_a", 0.92),
            _make_package("bank_b", 0.80),
            _make_package("bank_c", 0.87),
            _make_package("bank_d", 0.95),
        ]
        scores = score_all_clients(packages)
        assert set(scores.keys()) == {"bank_a", "bank_b", "bank_c", "bank_d"}

    def test_ordering_matches_val_auc_ordering(self):
        """bank with highest val_auc must have highest score."""
        packages = [
            _make_package("bank_a", 0.80),
            _make_package("bank_b", 0.92),
            _make_package("bank_c", 0.75),
            _make_package("bank_d", 0.88),
        ]
        scores = score_all_clients(packages)
        assert scores["bank_b"] > scores["bank_d"] > scores["bank_a"] > scores["bank_c"]

    def test_all_scores_in_unit_interval(self):
        packages = [
            _make_package("bank_a", 0.60),
            _make_package("bank_b", 0.85),
            _make_package("bank_c", 0.99),
        ]
        scores = score_all_clients(packages)
        for bid, s in scores.items():
            assert 0.0 <= s <= 1.0, f"{bid} score {s:.4f} out of [0,1]"

    def test_single_client(self):
        packages = [_make_package("bank_a", 0.90)]
        scores = score_all_clients(packages)
        assert "bank_a" in scores
        assert 0.0 <= scores["bank_a"] <= 1.0

    def test_example_values_from_spec(self):
        """
        Validate the specific example values from the specification:
          val_auc = 0.85 → score ≈ 0.5
          val_auc = 0.93 → score > 0.5
          val_auc = 0.70 → score < 0.5
        """
        pkgs = [
            _make_package("bank_a", 0.85),
            _make_package("bank_b", 0.93),
            _make_package("bank_c", 0.70),
        ]
        scores = score_all_clients(pkgs)
        assert abs(scores["bank_a"] - 0.5) <= 0.01
        assert scores["bank_b"] > 0.5
        assert scores["bank_c"] < 0.5
