import json
import os
import time
import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self, repo_root: str, banks: list[str]):
        self.results_dir = os.path.join(repo_root, "server", "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.history = {
            "round": [],
            "per_bank_auc": {b: [] for b in banks},
            "per_bank_val_loss": {b: [] for b in banks},
            "per_bank_f1": {b: [] for b in banks},
            "per_bank_precision": {b: [] for b in banks},
            "per_bank_recall": {b: [] for b in banks},
            "avg_auc": [],
            "round_duration_s": [],
        }

    def record_round(
        self, round_num: int, histories: dict[str, object], duration_s: float
    ) -> None:
        per_bank_auc = []

        for bank_id, history in histories.items():
            # Use BEST epoch AUC across local epochs (not just the last)
            bank_auc  = max(history.val_auc)  if history.val_auc  else 0.0
            bank_loss = history.val_loss[-1]   if history.val_loss else 0.0
            bank_f1   = max(history.val_f1)        if history.val_f1        else 0.0
            bank_prec = history.val_precision[-1]  if history.val_precision else 0.0
            bank_rec  = history.val_recall[-1]     if history.val_recall    else 0.0

            self.history["per_bank_auc"][bank_id].append(bank_auc)
            self.history["per_bank_val_loss"][bank_id].append(bank_loss)
            self.history["per_bank_f1"][bank_id].append(bank_f1)
            self.history["per_bank_precision"][bank_id].append(bank_prec)
            self.history["per_bank_recall"][bank_id].append(bank_rec)
            per_bank_auc.append(bank_auc)

        avg_auc = sum(per_bank_auc) / max(1, len(per_bank_auc))
        self.history["round"].append(round_num)
        self.history["avg_auc"].append(avg_auc)
        self.history["round_duration_s"].append(duration_s)

        header = f"[ROUND {round_num:03d}] AUC Summary"
        print(header)
        print("-" * len(header))
        for bank_id, auc_val in self.history["per_bank_auc"].items():
            print(f"  {bank_id}: {auc_val[-1]:.4f}")
        print(f"  avg_auc: {avg_auc:.4f}")

    def save_log(self) -> str:
        path = os.path.join(self.results_dir, "round_log.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def plot_convergence(self) -> str:
        rounds = self.history["round"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax_auc = axes[0]
        for bank_id, auc_values in self.history["per_bank_auc"].items():
            ax_auc.plot(rounds, auc_values, label=bank_id)
        ax_auc.plot(
            rounds,
            self.history["avg_auc"],
            label="avg_auc",
            linewidth=2.5,
            color="black",
        )
        ax_auc.set_ylabel("Validation AUC")
        ax_auc.set_title("AUC vs Round")
        ax_auc.grid(True, alpha=0.3)
        ax_auc.legend()

        ax_loss = axes[1]
        for bank_id, loss_values in self.history["per_bank_val_loss"].items():
            ax_loss.plot(rounds, loss_values, label=bank_id)
        ax_loss.set_xlabel("Round")
        ax_loss.set_ylabel("Validation Loss")
        ax_loss.set_title("Validation Loss vs Round")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()

        plt.tight_layout()
        path = os.path.join(self.results_dir, "convergence.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[METRICS] Saved convergence plot to {path}")
        return path

    def print_final_summary(self) -> None:
        if not self.history["round"]:
            print("[METRICS] No rounds recorded.")
            return

        best_idx = max(
            range(len(self.history["avg_auc"])),
            key=lambda i: self.history["avg_auc"][i],
        )
        best_round = self.history["round"][best_idx]
        best_avg_auc = self.history["avg_auc"][best_idx]
        total_time = sum(self.history["round_duration_s"])

        print("\n=== FINAL SUMMARY ===")
        print(f"Best round: {best_round:03d} (avg_auc={best_avg_auc:.4f})")
        for bank_id, auc_values in self.history["per_bank_auc"].items():
            best_bank_auc = max(auc_values) if auc_values else 0.0
            print(f"  {bank_id}: best_auc={best_bank_auc:.4f}")
        print(f"Total training time: {total_time:.1f}s")
