"""
FL Server — Central Aggregation Node
======================================
Holds global model state across FL rounds and delegates each aggregation step
to Track 3's master aggregator (aggregator.py).

Track 4 will eventually wrap this in a FastAPI endpoint. For now it is used
directly by fl_runner.py via Python import (same-process, zero HTTP overhead).

Track 4 import contract (unchanged from spec):
    from server.fl_server import FLServer
    server = FLServer(initial_weights=..., method="custom", ...)
    result = server.aggregate_round(weight_packages, round_num=n)
    new_global = server.get_global_weights()
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_SERVER_DIR = os.path.dirname(__file__)                        # server/
_REPO_ROOT  = os.path.abspath(os.path.join(_SERVER_DIR, ".."))  # repo root
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server.src.aggregator import (
    aggregate,
    AggregationResult,
)


class FLServer:
    """
    Stateful FL server. One instance per experiment run.

    Attributes:
        current_global_weights — The current global base-layer weight dict.
                                  Updated in-place after every aggregate_round().
        method                 — Aggregation method ("fedavg", "fedprox",
                                  "dp_fedavg", or "custom").
        dp_config              — DP settings dict or None.
                                  Required for "dp_fedavg" and "custom".
        fedprox_mu             — Server-side FedProx coefficient (used to
                                  interpret reported proximal_term values).
        round_history          — List of round_diagnostics dicts, one per round.
        current_round          — Most recently completed round number.
    """

    def __init__(
        self,
        initial_weights: dict[str, list],
        method: str = "custom",
        dp_config: Optional[dict] = None,
        fedprox_mu: float = 0.01,
    ):
        """
        Args:
            initial_weights: Starting global base-layer weights (nested lists).
                             Typically extracted from a fresh FraudDetectionMLP.
            method:          Aggregation strategy. See aggregate() for options.
            dp_config:       {"clip_norm": float, "noise_multiplier": float}
                             or None (no DP). Automatically set for "custom"
                             and "dp_fedavg" if dp_config is not supplied.
            fedprox_mu:      Fallback mu used when a client's package lacks it.
        """
        self.current_global_weights: dict[str, list] = initial_weights
        self.method = method
        self.fedprox_mu = fedprox_mu
        self.round_history: list[dict] = []
        self.current_round: int = 0

        # Auto-supply a sensible default DP config for methods that need it
        if dp_config is None and method in ("dp_fedavg", "custom"):
            dp_config = {"clip_norm": 1.0, "noise_multiplier": 1.1}
        self.dp_config = dp_config

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def aggregate_round(
        self,
        weight_packages: list[dict],
        round_num: Optional[int] = None,
    ) -> AggregationResult:
        """
        Runs one FL aggregation round.

        Args:
            weight_packages: List of Track 3 weight_package dicts — one per
                             participating bank, as returned by FLClient.
            round_num:       Current round number. Defaults to current_round + 1.

        Returns:
            AggregationResult with:
              .new_global_weights  — updated global weight dict
              .round_diagnostics   — per-client scores, effective weights,
                                     privacy budget, etc. (for dashboard)
        """
        if round_num is None:
            round_num = self.current_round + 1

        result: AggregationResult = aggregate(
            weight_packages=weight_packages,
            current_global_weights=self.current_global_weights,
            method=self.method,
            dp_config=self.dp_config,
            fedprox_mu=self.fedprox_mu,
            num_rounds=round_num,
        )

        # Commit new weights and advance round counter
        self.current_global_weights = result.new_global_weights
        self.current_round = round_num
        self.round_history.append({
            "round": round_num,
            "diagnostics": result.round_diagnostics,
        })

        return result

    def get_global_weights(self) -> dict[str, list]:
        """Returns the current global base-layer weights (broadcast to clients)."""
        return self.current_global_weights

    def get_round_summary(self) -> list[dict]:
        """Returns a condensed summary of every completed round's diagnostics."""
        summary = []
        for entry in self.round_history:
            rnd = entry["round"]
            diag = entry["diagnostics"]
            per_client_summary = {
                bid: {
                    "val_auc":          d.get("val_auc"),
                    "effective_weight": d.get("effective_weight"),
                    "reliability_score": d.get("reliability_score"),
                    "conflict_penalty": d.get("conflict_penalty"),
                }
                for bid, d in diag.get("per_client", {}).items()
            }
            budget = diag.get("privacy_budget_used")
            summary.append({
                "round":              rnd,
                "method":             diag.get("method"),
                "per_client":         per_client_summary,
                "privacy_budget":     budget,
            })
        return summary
