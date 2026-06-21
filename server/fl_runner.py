"""
FL Runner — End-to-End Federated Learning Orchestrator
========================================================
Single entry point for running N complete FL rounds across all four banks.
Coordinates FLClient (Track 2 / client side) and FLServer (Track 3 / server side)
in a controlled, reproducible experiment loop.

Usage:
    # From the repo root (Adaptive-FL-for-Bank-Fraud-Detection/)
    python server/fl_runner.py                                   # defaults: 10 rounds, custom
    python server/fl_runner.py --rounds 5 --method fedavg       # FedAvg baseline
    python server/fl_runner.py --rounds 10 --method fedprox     # FedProx only
    python server/fl_runner.py --rounds 20 --method custom      # Full pipeline (default)
    python server/fl_runner.py --rounds 20 --method custom --noise_multiplier 2.0  # strong DP

Outputs:
    fl_results.json                  ← Round-by-round metrics + final aggregate
    evaluation/per_bank_results.json ← Per-bank AUC, F1, FNR, FPR on val split
    evaluation/global_summary.json   ← Aggregate across all four banks

Architecture (per round):
    Server broadcasts global_weights
        ↓
    [bank_a, bank_b, bank_c, bank_d] run local_train_and_package()
        ↓  (FedProx loss, 3 epochs each)
    Server aggregates weight_packages → new_global_weights
        ↓
    Print per-client diagnostics (val_auc, effective_weight, conflict_penalty)
    After final round:
        → evaluate_global_model() on all four bank val splits
        → save fl_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Generator

# ── Repo root on sys.path — enables all cross-module imports ──────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.src.dataloaders import get_dataloader
from client.src.model import FraudDetectionMLP
from client.src.fl_client import FLClient
from server.fl_server import FLServer

BANK_IDS   = ["bank_a", "bank_b", "bank_c", "bank_d"]
DB_ROOT    = REPO_ROOT / "data" / "databases"
EVAL_DIR   = Path(__file__).parent / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initialise_global_weights(feature_dim: int) -> dict[str, list]:
    """
    Creates a fresh FraudDetectionMLP and extracts its base-layer weights as
    the starting global weights for round 1.

    Using random initialisation (not a saved checkpoint) ensures no bank
    has an unfair advantage from round-0 knowledge.
    """
    import torch
    model = FraudDetectionMLP(input_dim=feature_dim)
    weights: dict[str, list] = {}
    for name, param in model.state_dict().items():
        if name.startswith("base_layers"):
            if "running" in name or "tracked" in name:
                continue
            weights[name] = param.cpu().numpy().tolist()
    return weights


def _get_feature_dim() -> int:
    """Initialises FEATURE_DIM via a minimal dataloader call on bank_a."""
    get_dataloader(str(DB_ROOT / "bank_a.db"), batch_size=1, split="val")
    from data.src.dataloaders import FEATURE_DIM
    return FEATURE_DIM


def _print_round_header(round_num: int, total: int, method: str) -> None:
    print(f"\n{'='*64}")
    print(f"  FL Round {round_num}/{total}  |  method={method}")
    print(f"{'='*64}")


def _print_round_diagnostics(round_num: int, diag: dict) -> None:
    print(f"\n  [Server] Round {round_num} aggregation complete:")
    print(f"  {'Bank':<10} {'val_auc':>8} {'eff_w':>7} {'reliability':>12} {'conflict':>10} {'dp':>5}")
    print(f"  {'-'*60}")
    for bid, d in diag["per_client"].items():
        print(
            f"  {bid:<10} {d['val_auc']:>8.4f} {d['effective_weight']:>7.4f} "
            f"{d['reliability_score']:>12.4f} {d['conflict_penalty']:>10.4f} "
            f"{'yes' if d['dp_applied'] else 'no':>5}"
        )
    if diag.get("privacy_budget_used"):
        b = diag["privacy_budget_used"]
        interp = b.get("interpretation", "")
        print(f"\n  Privacy budget consumed: eps={b['epsilon']:.4f} {interp}")


# ---------------------------------------------------------------------------
# Core FL generator — yields one dict per round + final complete event
# ---------------------------------------------------------------------------


def run_fl_gen(args: argparse.Namespace) -> Generator[dict, None, None]:
    """
    Generator that runs the full FL loop and yields a structured dict after
    each round. A final dict with event='complete' is yielded after all rounds.

    Both the CLI (run_fl) and the FastAPI layer (api.py) consume this generator.
    No FL logic is duplicated between them.

    Yields per round:
        {
          "event": "round",
          "round": int,
          "total_rounds": int,
          "method": str,
          "per_client": { bank_id: { val_auc, val_loss, effective_weight,
                                     reliability_score, conflict_penalty,
                                     proximal_term, dp_applied } },
          "privacy_budget": dict | None,
          "test_evaluation": { "aggregate": dict, "per_bank": dict }
        }

    Yields on completion:
        {
          "event": "complete",
          "output": { ...full fl_results.json equivalent... }
        }
    """
    t_start = time.time()

    # ── Initialise ──────────────────────────────────────────────────────
    feature_dim = _get_feature_dim()
    global_weights = _initialise_global_weights(feature_dim)

    dp_config = None
    if args.method in ("dp_fedavg", "custom"):
        dp_config = {"clip_norm": args.clip_norm, "noise_multiplier": args.noise_multiplier}

    server = FLServer(
        initial_weights=global_weights,
        method=args.method,
        dp_config=dp_config,
        fedprox_mu=args.mu,
    )

    clients = {
        bid: FLClient(
            bank_id=bid,
            db_path=str(DB_ROOT / f"{bid}.db"),
            mu=args.mu,
            local_epochs=args.local_epochs,
            device=args.device,
        )
        for bid in BANK_IDS
    }

    # ── FL Round Loop ────────────────────────────────────────────────────
    round_log: list[dict] = []
    round_report = None

    for round_num in range(1, args.rounds + 1):
        _print_round_header(round_num, args.rounds, args.method)

        # Broadcast global weights
        current_global = server.get_global_weights()

        # Local training at each bank
        weight_packages = []
        for bid in BANK_IDS:
            print(f"\n  [{bid}] Local training (round {round_num})...")
            t_client = time.time()
            pkg = clients[bid].local_train_and_package(
                global_weights=current_global,
                round_num=round_num,
            )
            elapsed = time.time() - t_client
            weight_packages.append(pkg)
            meta = pkg["metadata"]
            print(
                f"  [{bid}] Done in {elapsed:.1f}s | "
                f"val_auc={meta['val_auc']:.4f} | "
                f"proximal_term={meta['proximal_term']:.6f}"
            )

        # Server aggregation
        print(f"\n  [Server] Aggregating {len(weight_packages)} client updates...")
        result = server.aggregate_round(weight_packages, round_num=round_num)
        _print_round_diagnostics(round_num, result.round_diagnostics)

        # Evaluate Global Model
        print(f"\n  [Server] Evaluating global model (round {round_num})...")
        from server.evaluation.global_evaluator import evaluate_global_model

        client_models = {bid: clients[bid].model for bid in BANK_IDS}
        bank_db_paths = {bid: str(DB_ROOT / f"{bid}.db") for bid in BANK_IDS}

        round_report = evaluate_global_model(
            global_weights=server.get_global_weights(),
            model_template=client_models,
            bank_db_paths=bank_db_paths,
            method=args.method,
            num_rounds=round_num,
        )

        agg = round_report.aggregate
        print(
            f"  --> Global Test Metrics: AUC={agg['weighted_auc']:.4f} | "
            f"F1={agg['mean_f1']:.4f} | FNR={agg['mean_fnr']:.4f} | FPR={agg['mean_fpr']:.4f}"
        )

        # Build this round's structured event dict
        from dataclasses import asdict
        per_client_data = {
            bid: {
                "val_auc":           result.round_diagnostics["per_client"][bid]["val_auc"],
                "val_loss":          next(
                    p["metadata"]["val_loss"]
                    for p in weight_packages if p["bank_id"] == bid
                ),
                "effective_weight":  result.round_diagnostics["per_client"][bid]["effective_weight"],
                "reliability_score": result.round_diagnostics["per_client"][bid]["reliability_score"],
                "conflict_penalty":  result.round_diagnostics["per_client"][bid]["conflict_penalty"],
                "proximal_term":     next(
                    p["metadata"]["proximal_term"]
                    for p in weight_packages if p["bank_id"] == bid
                ),
                "dp_applied":        result.round_diagnostics["per_client"][bid]["dp_applied"],
            }
            for bid in BANK_IDS
        }

        round_event = {
            "event":          "round",
            "round":          round_num,
            "total_rounds":   args.rounds,
            "method":         args.method,
            "per_client":     per_client_data,
            "privacy_budget": result.round_diagnostics.get("privacy_budget_used"),
            "test_evaluation": {
                "aggregate": agg,
                "per_bank":  {bid: asdict(round_report.per_bank[bid]) for bid in BANK_IDS},
            },
        }

        round_log.append({
            "round":           round_num,
            "per_client":      per_client_data,
            "privacy_budget":  result.round_diagnostics.get("privacy_budget_used"),
            "test_evaluation": round_event["test_evaluation"],
        })

        yield round_event

    # ── Final save ───────────────────────────────────────────────────────
    from dataclasses import asdict

    output = {
        "method":          args.method,
        "rounds":          args.rounds,
        "local_epochs":    args.local_epochs,
        "mu":              args.mu,
        "dp_config":       dp_config,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "round_history":   round_log,
        "final_aggregate": round_report.aggregate if round_report else {},
        "final_per_bank":  (
            {bid: asdict(round_report.per_bank[bid]) for bid in BANK_IDS}
            if round_report else {}
        ),
    }

    out_path = REPO_ROOT / "fl_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to {out_path}")
    print(f"  Total time: {output['elapsed_seconds']:.1f}s")
    print("\n  Done.\n")

    yield {"event": "complete", "output": output}


# ---------------------------------------------------------------------------
# CLI wrapper — consumes the generator, prints as before
# ---------------------------------------------------------------------------


def run_fl(args: argparse.Namespace) -> None:
    """CLI entry point. Consumes run_fl_gen() and prints progress."""
    print("\n" + "+" + "="*62 + "+")
    print("|" + "  Adaptive FL for Bank Fraud Detection".center(62) + "|")
    print("+" + "="*62 + "+")
    print(f"  Method:         {args.method}")
    print(f"  Rounds:         {args.rounds}")
    print(f"  Local epochs:   {args.local_epochs}")
    print(f"  FedProx mu:     {args.mu}")
    if args.method in ("dp_fedavg", "custom"):
        print(f"  DP clip_norm:   {args.clip_norm}")
        print(f"  DP noise_mult:  {args.noise_multiplier}")
    print("\n  Initialising dataloaders and model architecture...")

    for event in run_fl_gen(args):
        if event["event"] == "complete":
            output = event["output"]
            # ── Final summary table ──────────────────────────────────────
            print(f"\n{'+'*64}")
            print(f"  Final Global Model Evaluation (after {args.rounds} rounds)")
            print(f"{'+'*64}")
            print(f"\n  {'Metric':<25} {'Value':>10}")
            print(f"  {'-'*37}")
            for k, v in output["final_aggregate"].items():
                print(f"  {k:<25} {str(v):>10}")
            print(
                f"\n  {'Bank':<10} {'AUC':>7} {'F1':>7} {'FNR':>7} "
                f"{'FPR':>7} {'Acc':>7} {'Samples':>9}"
            )
            print(f"  {'-'*57}")
            for bid in BANK_IDS:
                r = output["final_per_bank"][bid]
                print(
                    f"  {bid:<10} {r['auc_roc']:>7.4f} {r['f1_score']:>7.4f} "
                    f"{r['false_negative_rate']:>7.4f} {r['false_positive_rate']:>7.4f} "
                    f"{r['accuracy']:>7.4f} {r['num_test_samples']:>9,}"
                )
        # round events: already printed inside run_fl_gen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end federated learning across all four banks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Number of FL rounds to run.",
    )
    parser.add_argument(
        "--method", default="custom",
        choices=["fedavg", "fedprox", "dp_fedavg", "custom"],
        help="Aggregation method.",
    )
    parser.add_argument(
        "--local_epochs", type=int, default=3,
        help="Local training epochs per client per round.",
    )
    parser.add_argument(
        "--mu", type=float, default=0.01,
        help="FedProx proximal coefficient.",
    )
    parser.add_argument(
        "--clip_norm", type=float, default=1.0,
        help="DP L2 clipping norm (only used for dp_fedavg / custom).",
    )
    parser.add_argument(
        "--noise_multiplier", type=float, default=1.1,
        help="DP Gaussian noise multiplier (only used for dp_fedavg / custom).",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="PyTorch device for local training.",
    )
    args = parser.parse_args()
    run_fl(args)


if __name__ == "__main__":
    main()
