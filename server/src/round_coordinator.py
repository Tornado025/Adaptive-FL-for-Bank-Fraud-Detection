import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

from client.src.model import FraudDetectionMLP
from client.src.train import TrainingConfig, TrainingHistory, train_local
from client.src.weight_extractor import extract_base_weights, load_base_weights
from data.src.dataloaders import get_dataloader, FEATURE_DIM


class RoundCoordinator:
    def __init__(self, repo_root: str, banks: list[str], feature_dim: int, fl_config):
        self.repo_root = repo_root
        self.banks = banks
        self.feature_dim = feature_dim if feature_dim is not None else FEATURE_DIM
        self.fl_config = fl_config

    def _train_one_bank(
        self, bank_id: str, global_weight_dict: dict, round_num: int
    ) -> tuple[dict, TrainingHistory]:
        db_path = os.path.join(self.repo_root, "data", "databases", f"{bank_id}.db")
        train_loader = get_dataloader(
            db_path, batch_size=self.fl_config.batch_size, split="train"
        )
        val_loader = get_dataloader(
            db_path, batch_size=self.fl_config.batch_size, split="val"
        )

        model = FraudDetectionMLP(input_dim=self.feature_dim)
        load_base_weights(model, global_weight_dict)

        config = TrainingConfig(
            bank_id=bank_id,
            local_epochs=self.fl_config.local_epochs,
            learning_rate=self.fl_config.learning_rate,
            batch_size=self.fl_config.batch_size,
            weight_decay=self.fl_config.weight_decay,
            device=self.fl_config.device,
            grad_clip_norm=self.fl_config.grad_clip_norm,
            dp_noise_std=self.fl_config.dp_noise_std,
        )

        global_params = None
        if self.fl_config.strategy == "fedprox":
            config.fedprox_mu = self.fl_config.fedprox_mu
            global_params = [p.clone().detach() for p in model.parameters()]

        history = train_local(
            model, train_loader, val_loader, config, global_params=global_params
        )

        num_samples = len(train_loader.dataset)
        weight_dict = extract_base_weights(
            model,
            bank_id=bank_id,
            round_num=round_num,
            num_samples=num_samples,
            dp_noise_std=self.fl_config.dp_noise_std,
        )

        return weight_dict, history

    def run_round(
        self, round_num: int, global_weight_dict: dict
    ) -> tuple[list[dict], dict[str, TrainingHistory]]:
        print("\n" + "=" * 70)
        print(f"[ROUND {round_num:03d}] Starting federated round")
        print("=" * 70)

        weight_dicts = []
        histories = {}

        for bank_id in self.banks:
            weight_dict, history = self._train_one_bank(
                bank_id, global_weight_dict, round_num
            )
            weight_dicts.append(weight_dict)
            histories[bank_id] = history
            print(
                f"[ROUND {round_num:03d}] Completed {bank_id} | "
                f"samples={weight_dict.get('num_samples', 0)}"
            )

        return weight_dicts, histories
