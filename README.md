# Adaptive Federated Learning for Bank Fraud Detection

This repository contains a Federated Learning (FL) framework designed for fraud detection across multiple bank databases. The system enables collaborative model training while maintaining data isolation and transaction privacy.

## System Overview

The project implements an asynchronous Federated Learning pipeline utilizing a Split-Neural Network architecture. This ensures that only general feature representation layers are shared, while classification layers remain local and personalized to each financial institution.

### Core Components

*   **Split-NN Architecture:** The global model consists only of base feature extraction layers. Personalized top layers are maintained locally by each bank to handle institution-specific fraud patterns.
*   **FedProx Optimization:** Implements a proximal term in the local objective function to stabilize training and mitigate client drift in non-IID data distributions.
*   **Differential Privacy (DP):** Incorporates a Gaussian mechanism to add calibrated noise to weight updates, ensuring formal (epsilon, delta)-differential privacy.
*   **Adaptive Aggregation:** The server utilizes reliability scoring based on validation performance and conflict resolution via weight similarity to optimize the global model convergence.
*   **Local Normalization (FedBN):** Batch normalization statistics are kept local to each bank to prevent instability during weight aggregation and improve local inference quality.

## Project Structure

```text
├── client/                 # Client-side infrastructure
│   ├── src/
│   │   ├── model.py        # FraudDetectionMLP architecture
│   │   ├── train.py        # FedProx training implementation
│   │   ├── fl_client.py    # Client orchestration logic
│   │   └── weight_extractor.py
│   └── models/             # Local model checkpoints
├── server/                 # Server-side coordination
│   ├── src/                # Core aggregation algorithms
│   │   ├── aggregator.py   # Master aggregation pipeline
│   │   ├── dp_mechanism.py # Differential Privacy implementation
│   │   └── fedprox.py      # Server-side optimization handling
│   ├── evaluation/         # Automated performance reports
│   ├── fl_server.py        # Global state management
│   └── fl_runner.py        # Experiment orchestration script
├── data/                   # Data management layer
│   ├── databases/          # SQLite bank databases
│   └── src/                # Dataloading and preprocessing
├── evaluation/             # Root-level aggregate results
└── fl_results.json         # Detailed experiment logs
```

## Operational Workflow

1.  **Broadcast:** The central server distributes the current global base layer weights to all participating clients.
2.  **Local Optimization:** Each client attaches its private top layers and executes local training using the FedProx optimizer on its private data split.
3.  **Weight Extraction:** Clients extract the updated base layer weights and package them with local performance metadata (Validation AUC, loss).
4.  **Server Aggregation:** The server receives the weight packages and executes the aggregation pipeline:
    *   Applies L2 clipping and Gaussian noise for Differential Privacy.
    *   Calculates reliability coefficients and conflict penalties.
    *   Computes the new global model via weighted averaging.
5.  **Global Evaluation:** The resulting model is evaluated against the held-out validation sets of all clients to track convergence.

## Execution Instructions

### Environment Requirements
*   Python 3.10 or higher
*   PyTorch
*   NumPy, Pandas, Scikit-learn

### Running Experiments
Experiments are managed via the `fl_runner.py` script located in the `server/` directory.

```powershell
cd server
python fl_runner.py --rounds 20 --method custom --local_epochs 3 --noise_multiplier 0.5
```

### Supported Aggregation Methods
*   `fedavg`: Standard Federated Averaging baseline.
*   `fedprox`: Federated Proximal optimization.
*   `dp_fedavg`: Federated Averaging with Differential Privacy.
*   `custom`: Complete pipeline including DP, FedProx, reliability scoring, and conflict resolution.

## Output and Diagnostics
The system generates a comprehensive `fl_results.json` file at the repository root containing round-by-round metrics, effective weights, and privacy budget consumption logs. Detailed per-bank performance metrics are stored in the `server/evaluation/` directory.
