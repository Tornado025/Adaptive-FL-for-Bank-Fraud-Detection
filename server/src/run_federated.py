import os
import sys
import time
from dataclasses import dataclass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

from data.src.dataloaders import get_dataloader, FEATURE_DIM
from server.src.aggregator import aggregate
from server.src.evaluate_global import run_evaluation
from server.src.metrics_tracker import MetricsTracker
from server.src.model_manager import ModelManager
from server.src.round_coordinator import RoundCoordinator


@dataclass
class FLConfig:
    num_rounds: int = 10
    local_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    strategy: str = "fedavg"
    fedprox_mu: float = 0.01
    dp_noise_std: float = 0.01
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    banks: list = None

    def __post_init__(self):
        if self.banks is None:
            self.banks = ["bank_a", "bank_b", "bank_c", "bank_d"]


def main() -> None:
    config = FLConfig()
    banner = (
        "=" * 70
        + "\n"
        + "  ADAPTIVE FEDERATED LEARNING - BANK FRAUD DETECTION\n"
        + "=" * 70
    )
    print(banner)
    print(config)

    db_path = os.path.join(REPO_ROOT, "data", "databases", "bank_a.db")
    _ = get_dataloader(db_path, batch_size=config.batch_size, split="train")
    from data.src.dataloaders import FEATURE_DIM

    model_manager = ModelManager(REPO_ROOT, feature_dim=FEATURE_DIM, device=config.device)
    model_manager.initialize_global_model()

    coordinator = RoundCoordinator(REPO_ROOT, config.banks, FEATURE_DIM, config)
    metrics_tracker = MetricsTracker(REPO_ROOT, config.banks)

    for round_num in range(1, config.num_rounds + 1):
        start = time.time()
        global_weight_dict = model_manager.get_global_weight_dict(round_num)
        weight_dicts, histories = coordinator.run_round(round_num, global_weight_dict)
        aggregated = aggregate(
            weight_dicts,
            strategy=config.strategy,
            global_weight_dict=global_weight_dict,
            fedprox_mu=config.fedprox_mu,
        )
        model_manager.apply_aggregated_weights(aggregated)
        model_manager.save_checkpoint(round_num)
        metrics_tracker.record_round(
            round_num, histories, duration_s=time.time() - start
        )

    metrics_tracker.save_log()
    metrics_tracker.plot_convergence()
    metrics_tracker.print_final_summary()

    run_evaluation(
        REPO_ROOT,
        model_manager.get_global_weight_dict(config.num_rounds),
        config.num_rounds,
    )

    print("[DONE] Federated training complete.")


if __name__ == "__main__":
    main()
