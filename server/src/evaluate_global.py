"""
Global model evaluation.
- Loads the GLOBAL model checkpoint (not per-bank client models).
- Evaluates on each bank's val set PLUS all banks combined.
- Computes: AUC-ROC, Precision, Recall, F1, and Confusion Matrix.
- Saves results to server/results/global_eval_results.json.
- Saves ROC + confusion-matrix plots to server/results/global_eval_r<N>.png.
"""

import os
import re
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

from data.src.dataloaders import get_dataloader, FEATURE_DIM
from client.src.model import FraudDetectionMLP
from client.src.weight_extractor import load_base_weights
from server.src.model_manager import ModelManager


def _find_best_threshold(labels, probs):
    """Find threshold that maximises F1 on the val set."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (np.array(probs) > t).astype(int)
        try:
            score = f1_score(labels, preds, zero_division=0)
        except ValueError:
            continue
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1


def _eval_one_bank(global_model, val_loader, device):
    """Run inference and return (labels, probs)."""
    global_model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            logits = global_model(features)
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def run_evaluation(repo_root: str, global_weight_dict: dict, round_num: int) -> dict:
    banks = ["bank_a", "bank_b", "bank_c", "bank_d"]
    device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # 1.  Build the global model using ONLY the global weights            #
    # ------------------------------------------------------------------ #
    # Ensure FEATURE_DIM is set
    global FEATURE_DIM
    if FEATURE_DIM is None:
        db_path = os.path.join(repo_root, "data", "databases", "bank_a.db")
        get_dataloader(db_path, batch_size=256, split="val")
        from data.src.dataloaders import FEATURE_DIM as FD
        FEATURE_DIM = FD

    from data.src.dataloaders import FEATURE_DIM as FD
    global_model = FraudDetectionMLP(input_dim=FD)
    load_base_weights(global_model, global_weight_dict)
    global_model = global_model.to(device)

    # ------------------------------------------------------------------ #
    # 2.  Collect per-bank + combined labels/probs                        #
    # ------------------------------------------------------------------ #
    bank_results = {}
    combined_labels, combined_probs = [], []

    for bank in banks:
        db_path = os.path.join(repo_root, "data", "databases", f"{bank}.db")
        val_loader = get_dataloader(db_path, batch_size=256, split="val")
        labels, probs = _eval_one_bank(global_model, val_loader, device)

        bank_results[bank] = {"labels": labels, "probs": probs}
        combined_labels.append(labels)
        combined_probs.append(probs)

    combined_labels = np.concatenate(combined_labels)
    combined_probs  = np.concatenate(combined_probs)

    # ------------------------------------------------------------------ #
    # 3.  Compute metrics for each bank + combined                        #
    # ------------------------------------------------------------------ #
    def compute_metrics(labels, probs, name):
        labels = np.array(labels, dtype=int)
        probs  = np.array(probs,  dtype=float)

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc     = auc(fpr, tpr)

        thresh, best_f1 = _find_best_threshold(labels, probs)
        preds = (probs > thresh).astype(int)

        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1   = f1_score(labels, preds, zero_division=0)
        cm   = confusion_matrix(labels, preds).tolist()
        ap   = average_precision_score(labels, probs)

        # PR curve
        pr_prec, pr_rec, _ = precision_recall_curve(labels, probs)

        n_pos = int(labels.sum())
        n_neg = int((1 - labels).sum())

        print(f"  [{name}] AUC={roc_auc:.4f}  P={prec:.4f}  R={rec:.4f}"
              f"  F1={f1:.4f}  AP={ap:.4f}  thresh={thresh:.2f}"
              f"  n_pos={n_pos}  n_neg={n_neg}")

        return {
            "name": name,
            "roc_auc": round(float(roc_auc), 6),
            "avg_precision": round(float(ap), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
            "best_threshold": round(float(thresh), 4),
            "confusion_matrix": cm,
            "n_positive": n_pos,
            "n_negative": n_neg,
            # Store curve arrays as lists for JSON serialisation
            "fpr": [round(v, 6) for v in fpr.tolist()],
            "tpr": [round(v, 6) for v in tpr.tolist()],
            "pr_precision": [round(v, 6) for v in pr_prec.tolist()],
            "pr_recall":    [round(v, 6) for v in pr_rec.tolist()],
        }

    metrics = {}
    print(f"\n[EVAL] Global model evaluation — Round {round_num:03d}")
    print("-" * 60)
    for bank in banks:
        d = bank_results[bank]
        metrics[bank] = compute_metrics(d["labels"], d["probs"], bank)

    metrics["combined"] = compute_metrics(combined_labels, combined_probs, "combined")

    # ------------------------------------------------------------------ #
    # 4.  Save metrics JSON                                               #
    # ------------------------------------------------------------------ #
    results_dir = os.path.join(repo_root, "server", "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, f"global_eval_results.json")
    payload = {"round": round_num, "banks": metrics}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[EVAL] Saved metrics JSON to {json_path}")

    # ------------------------------------------------------------------ #
    # 5.  Plot: ROC + PR + Confusion Matrix (per bank + combined)         #
    # ------------------------------------------------------------------ #
    all_entities = banks + ["combined"]
    all_labels_map = {b: bank_results[b]["labels"] for b in banks}
    all_labels_map["combined"] = combined_labels
    all_probs_map  = {b: bank_results[b]["probs"]  for b in banks}
    all_probs_map["combined"]  = combined_probs

    COLORS = {
        "bank_a": "#6c63ff", "bank_b": "#00d4aa",
        "bank_c": "#ffd166", "bank_d": "#ff6b6b", "combined": "#ffffff"
    }

    # --- Plot 1: All ROC curves on one axes + per-entity confusion matrices ---
    n_entities = len(all_entities)  # 5: 4 banks + combined
    fig = plt.figure(figsize=(20, 22), facecolor="#1b1e32")

    # ROC overlay (top row)
    ax_roc = fig.add_subplot(3, 3, (1, 2))
    ax_roc.set_facecolor("#141626")
    ax_roc.plot([0, 1], [0, 1], "w--", alpha=0.3, lw=1)
    for ent in all_entities:
        m = metrics[ent]
        ax_roc.plot(m["fpr"], m["tpr"],
                    color=COLORS[ent], lw=2.5 if ent == "combined" else 1.8,
                    label=f"{ent}  AUC={m['roc_auc']:.3f}")
    ax_roc.set_xlabel("False Positive Rate", color="white")
    ax_roc.set_ylabel("True Positive Rate", color="white")
    ax_roc.set_title("ROC Curves — All Banks + Combined (Global Model)", color="white", fontsize=13)
    ax_roc.tick_params(colors="white")
    ax_roc.legend(loc="lower right", fontsize=9, framealpha=0.3)
    for sp in ax_roc.spines.values():
        sp.set_edgecolor("#252840")

    # PR overlay (top row, right)
    ax_pr = fig.add_subplot(3, 3, 3)
    ax_pr.set_facecolor("#141626")
    for ent in all_entities:
        m = metrics[ent]
        ax_pr.plot(m["pr_recall"], m["pr_precision"],
                   color=COLORS[ent], lw=2.5 if ent == "combined" else 1.8,
                   label=f"{ent}  AP={m['avg_precision']:.3f}")
    ax_pr.set_xlabel("Recall", color="white")
    ax_pr.set_ylabel("Precision", color="white")
    ax_pr.set_title("Precision-Recall Curves (Global Model)", color="white", fontsize=13)
    ax_pr.tick_params(colors="white")
    ax_pr.legend(loc="upper right", fontsize=9, framealpha=0.3)
    for sp in ax_pr.spines.values():
        sp.set_edgecolor("#252840")

    # Confusion matrices (one per entity, 5 total = rows 2-3)
    cm_positions = [4, 5, 6, 7, 8]  # subplot indices for 3x3 grid
    for i, ent in enumerate(all_entities):
        ax = fig.add_subplot(3, 3, cm_positions[i])
        ax.set_facecolor("#141626")
        cm_data = np.array(metrics[ent]["confusion_matrix"])
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, annot_kws={"color": "white", "size": 11})
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        label_str = ent.upper() if ent != "combined" else "COMBINED"
        m = metrics[ent]
        ax.set_title(
            f"{label_str}\nAUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}"
            f"\nP={m['precision']:.3f}  R={m['recall']:.3f}",
            color="white", fontsize=10
        )
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#252840")

    # 9th slot — metric bar chart summary
    ax_bar = fig.add_subplot(3, 3, 9)
    ax_bar.set_facecolor("#141626")
    bar_entities = banks + ["combined"]
    bar_aucs = [metrics[e]["roc_auc"] for e in bar_entities]
    bar_f1s  = [metrics[e]["f1"]      for e in bar_entities]
    bar_prec = [metrics[e]["precision"] for e in bar_entities]
    bar_rec  = [metrics[e]["recall"]    for e in bar_entities]
    x = np.arange(len(bar_entities))
    w = 0.2
    ax_bar.bar(x - 1.5*w, bar_aucs, w, label="AUC-ROC", color="#6c63ff", alpha=0.85)
    ax_bar.bar(x - 0.5*w, bar_prec, w, label="Precision", color="#00d4aa", alpha=0.85)
    ax_bar.bar(x + 0.5*w, bar_rec,  w, label="Recall",    color="#ffd166", alpha=0.85)
    ax_bar.bar(x + 1.5*w, bar_f1s,  w, label="F1",        color="#ff6b6b", alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([e.replace("_", "\n") for e in bar_entities], color="white", fontsize=9)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Metric Summary (Global Model)", color="white", fontsize=11)
    ax_bar.tick_params(colors="white")
    ax_bar.legend(fontsize=8, framealpha=0.3, loc="lower right")
    ax_bar.set_facecolor("#141626")
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#252840")

    fig.suptitle(
        f"Global Model Evaluation — Round {round_num:03d}",
        fontsize=16, color="white", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = os.path.join(results_dir, f"global_eval_r{round_num:03d}.png")
    plt.savefig(plot_path, dpi=150, facecolor="#1b1e32")
    plt.close()
    print(f"[EVAL] Saved evaluation plot to {plot_path}")

    return metrics


def main() -> None:
    models_dir = os.path.join(REPO_ROOT, "server", "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Server models directory not found: {models_dir}")

    pattern = re.compile(r"global_model_r(\d{3})\.pt$")
    latest_round = None
    for name in os.listdir(models_dir):
        match = pattern.match(name)
        if match:
            round_num = int(match.group(1))
            if latest_round is None or round_num > latest_round:
                latest_round = round_num

    if latest_round is None:
        raise FileNotFoundError(f"No global model checkpoints found in {models_dir}")

    # Ensure FEATURE_DIM is initialised
    db_path = os.path.join(REPO_ROOT, "data", "databases", "bank_a.db")
    get_dataloader(db_path, batch_size=256, split="val")
    from data.src.dataloaders import FEATURE_DIM as FD

    manager = ModelManager(REPO_ROOT, feature_dim=FD, device="cpu")
    manager.load_checkpoint(latest_round)
    global_weight_dict = manager.get_global_weight_dict(latest_round)

    run_evaluation(REPO_ROOT, global_weight_dict, latest_round)


if __name__ == "__main__":
    main()
