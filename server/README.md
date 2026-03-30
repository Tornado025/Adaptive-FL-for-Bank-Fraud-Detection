# Track 3 — Federated Algorithms (Server-Side)

**Role: The Federated Research Scientist**

You live on the Central Server. You never see raw transaction data. You never touch the database. What you receive are **weight dictionaries** from Track 2, and your job is to figure out the smartest possible way to merge them into a single improved global model.

Your work is a progression: start with the dumbest possible algorithm (FedAvg), prove it works, then build progressively smarter versions on top of it. By Phase 4, you should have a custom aggregation algorithm that measurably outperforms the baseline. The benchmark results you produce are the project's core scientific contribution.

---

## File Structure

```
track3_federated_algorithms/
├── src/
│   ├── fed_avg.py             # Phase 1: Standard FedAvg baseline
│   ├── reliability_scorer.py  # Phase 2: Proxy dataset testing + reliability multipliers
│   ├── conflict_resolver.py   # Phase 3: Cosine similarity + conflict penalty math
│   ├── aggregator.py          # Phase 4: Master aggregation function combining all phases
│   └── benchmark.py           # Phase 4: Compares FedAvg vs Custom across simulated rounds
├── proxy_data/                # Small labeled dataset for server-side reliability testing
├── tests/
│   ├── test_fed_avg.py
│   ├── test_reliability.py
│   └── test_aggregator.py
└── requirements.txt
```

---

## The Federated Learning Round (Your Context)

Each "FL round" follows this sequence. You own steps 3 and 4:

```
Round N:
  1. [Track 4]  Server broadcasts current global base weights to all bank nodes
  2. [Track 2]  Each bank trains locally for 3 epochs, extracts base weights
  3. [Track 4]  Banks POST their weight packages to the server API
  4. [YOU]      Server receives N weight packages, aggregates them → new global weights
  5. [Track 4]  Server stores new global weights, increments round counter
  6. [Track 4]  WebSocket broadcasts updated metrics to the dashboard
```

Your aggregation function (step 4) is the heart of the system. It takes a list of weight packages and returns one merged weight dictionary.

---

## The Weight Package Format

This is what Track 2 sends you. Understand it deeply.

```python
weight_package = {
    "bank_id": "bank_a",
    "round": 3,
    "num_samples": 82341,
    "weights": {
        "base_layers.0.weight": [[...], ...],   # 2D list, shape [256, FEATURE_DIM]
        "base_layers.0.bias": [...],             # 1D list, shape [256]
        "base_layers.2.weight": [[...], ...],   # shape [128, 256]
        "base_layers.2.bias": [...],             # shape [128]
        # ... all base layer params
    },
    "metadata": {
        "val_loss": 0.312,
        "val_auc": 0.934,
        "local_epochs_trained": 3
    }
}
```

Your aggregator receives a list of these packages (one per participating bank) and returns a single `weights` dictionary with the same keys.

---

## Phase 1 — Standard FedAvg Baseline

**File: `src/fed_avg.py`**

Implement the canonical Federated Averaging algorithm (McMahan et al., 2017). This is your control group. Every subsequent improvement must beat this.

**Algorithm:**
For each weight tensor, compute a weighted average across all clients, where the weight for each client is proportional to its number of training samples.

```
global_weight[key] = Σ (n_k / N_total) * client_weight_k[key]
```
where `n_k` is `num_samples` for client k and `N_total = Σ n_k`.

```python
def federated_average(weight_packages: list[dict]) -> dict[str, list]:
    """
    Standard FedAvg. Sample-weighted mean of all base layer tensors.

    Args:
        weight_packages: List of weight package dicts from Track 2

    Returns:
        A single weights dict with the same keys, containing the averaged tensors.
        (JSON-serializable: values are Python lists, not tensors)
    """
```

**Validation test:** With identical weights from all clients, `federated_average()` must return weights numerically identical to the input (within floating point tolerance `1e-6`). Write this as a unit test in `tests/test_fed_avg.py`.

**Baseline benchmark:** Run 10 simulated FL rounds using FedAvg. Record the global model's AUC-ROC on the proxy dataset after each round. This is your baseline curve. Save to `proxy_data/fedavg_baseline_results.json`.

---

## Phase 2 — Client Reliability Scoring

**File: `src/reliability_scorer.py`**

Not all bank updates are equally trustworthy. A bank with noisy data, a poor local model, or adversarial behavior should have less influence on the global model. You will assess reliability using a **proxy dataset** stored on the server.

**The Proxy Dataset:**

Create a small, clean, balanced dataset in `proxy_data/`. This should be approximately 2,000–5,000 rows sampled from the IEEE-CIS dataset with a controlled 50/50 fraud/non-fraud split. This dataset represents "ground truth" — the server uses it to test whether a bank's update actually improves fraud detection.

```python
# proxy_data/proxy_dataset.csv  (2000 rows, balanced 50/50)
# Generated once from the IEEE-CIS data. NOT from any single bank's distribution.
```

**Reliability Scoring Algorithm:**

```python
def score_client_reliability(
    weight_package: dict,
    current_global_weights: dict,
    proxy_loader: DataLoader,
    model_template: nn.Module,
) -> float:
    """
    Returns a reliability score in [0.0, 1.0].

    Process:
    1. Load the CURRENT global weights into a model instance
    2. Evaluate this baseline model on the proxy dataset → baseline_auc
    3. Load the CLIENT'S proposed weights into the same model
    4. Evaluate this candidate model on the proxy dataset → candidate_auc
    5. Compute delta = candidate_auc - baseline_auc
    6. Convert delta to a [0, 1] reliability score using a sigmoid transform:
       score = sigmoid(delta * sensitivity_factor)
       where sensitivity_factor = 10 (tunable hyperparameter)

    A score of 0.5 means the update is neutral (no change from baseline).
    A score > 0.5 means the update improves performance on the proxy.
    A score < 0.5 means the update hurts performance (potentially unreliable or adversarial).
    """
```

```python
def score_all_clients(
    weight_packages: list[dict],
    current_global_weights: dict,
    proxy_loader: DataLoader,
    model_template: nn.Module,
) -> dict[str, float]:
    """Returns {'bank_a': 0.73, 'bank_b': 0.61, 'bank_c': 0.88, 'bank_d': 0.45}"""
```

**Important design note:** The proxy dataset should be representative of the general fraud detection task, not skewed toward any single bank's profile. It is the server's "ground truth compass."

---

## Phase 3 — Conflict-Aware Math

**File: `src/conflict_resolver.py`**

Banks with Non-IID data will sometimes produce updates that conflict with each other — their weight tensors push the global model in opposing directions. Naive averaging silently cancels these out. You will detect and penalize them explicitly.

**Cosine Similarity Between Weight Updates:**

First, convert weight updates into flat vectors for comparison:

```python
def weights_to_vector(weights: dict[str, list]) -> np.ndarray:
    """Flattens all weight tensors in a dict into a single 1D numpy array."""
    return np.concatenate([np.array(v).flatten() for v in weights.values()])
```

Then compute pairwise cosine similarity across all clients:

```python
def compute_pairwise_conflict(weight_packages: list[dict]) -> np.ndarray:
    """
    Returns an NxN matrix of cosine similarities between all client weight vectors.
    Values close to 1.0: high agreement (pointing in the same direction)
    Values close to 0.0: orthogonal (no relationship)
    Values close to -1.0: direct conflict (pointing in opposite directions)
    """
```

**Conflict Penalty:**

```python
def compute_conflict_penalty(weight_packages: list[dict]) -> dict[str, float]:
    """
    For each client, computes how much it conflicts with the majority direction.

    Algorithm:
    1. Compute the mean weight vector across all clients (the "consensus direction")
    2. For each client, compute cosine_similarity(client_vector, consensus_vector)
    3. penalty_k = max(0, -cosine_similarity_k)
       (only negative cosine sim — direct opposition — incurs a penalty)
    4. normalized_penalty_k = penalty_k / max(all penalties + ε)

    Returns: {'bank_a': 0.0, 'bank_b': 0.12, 'bank_c': 0.0, 'bank_d': 0.73}
    (bank_d is strongly conflicting with the majority)
    """
```

**Key insight:** A bank like Bank C (high-fraud region) may legitimately produce weights that conflict with the other banks. The penalty should not completely silence a bank — it should modulate its influence. The final weight for a client is:

```
effective_weight_k = reliability_score_k * (1 - conflict_penalty_k) * (n_k / N_total)
```

The `effective_weight_k` values are then re-normalized to sum to 1 before aggregation.

---

## Phase 4 — Merging & Benchmarking

**File: `src/aggregator.py`**

Combine Phase 1, 2, and 3 into one master aggregation function. This is the function that Track 4's API will call.

```python
def aggregate(
    weight_packages: list[dict],
    current_global_weights: dict,
    proxy_loader: DataLoader,
    model_template: nn.Module,
    method: str = "custom",  # "fedavg" or "custom"
) -> AggregationResult:
    """
    The master aggregation function. Called once per FL round.

    If method == "fedavg": runs Phase 1 only.
    If method == "custom": runs Phases 1+2+3 combined.

    Returns AggregationResult with:
    - new_global_weights: dict
    - round_diagnostics: dict containing per-client reliability scores,
      conflict penalties, and effective weights used
    """

@dataclass
class AggregationResult:
    new_global_weights: dict[str, list]
    round_diagnostics: dict  # Sent to Track 4 for dashboard display
```

**File: `src/benchmark.py`**

Run a controlled experiment to prove your custom algorithm beats FedAvg. Simulate 20 FL rounds using the 4 bank DataLoaders and compare both methods.

```python
def run_benchmark(
    bank_db_paths: dict[str, str],
    num_rounds: int = 20,
    methods: list[str] = ["fedavg", "custom"],
) -> BenchmarkReport:
    """
    For each method:
    - Initialize a fresh global model
    - Run `num_rounds` of simulated FL (Track 2's train_local + your aggregation)
    - After each round, evaluate the global model on the proxy dataset
    - Record AUC-ROC, F1, loss per round per method

    Save results to proxy_data/benchmark_results.json and plots to proxy_data/benchmark_plots/
    """
```

**Target result:** Your custom algorithm must achieve a higher final AUC-ROC than FedAvg on the proxy dataset, and must converge faster (reach 0.90 AUC in fewer rounds). Document the result in `proxy_data/BENCHMARK_SUMMARY.md`.

---

## Mathematical Summary

The final per-client aggregation weight is:

```
raw_weight_k  =  (n_k / N) × reliability_k × (1 - conflict_penalty_k)

effective_weight_k  =  raw_weight_k / Σ raw_weight_j    (normalize to sum = 1)

global_weights[key]  =  Σ_k ( effective_weight_k × client_k_weights[key] )
```

Where:
- `n_k / N` — sample-size fairness (FedAvg component)
- `reliability_k ∈ [0,1]` — proxy dataset performance delta (Phase 2)
- `1 - conflict_penalty_k ∈ [0,1]` — majority-direction agreement (Phase 3)

---

## Integration Contract with Track 4

Track 4 calls `aggregate()` via an internal Python import (not HTTP). The aggregator runs in the same server container process.

```python
# Track 4 calls this after receiving all client weight packages
from track3_federated_algorithms.src.aggregator import aggregate, AggregationResult

result: AggregationResult = aggregate(
    weight_packages=received_packages,
    current_global_weights=server_state.global_weights,
    proxy_loader=server_state.proxy_loader,
    model_template=server_state.model_template,
    method="custom"
)

server_state.global_weights = result.new_global_weights
# result.round_diagnostics is broadcast to the dashboard via WebSocket
```

---

## Dependencies

```
torch>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10   # for cosine similarity
pandas>=2.0
matplotlib>=3.7
```

Install: `pip install -r requirements.txt`
