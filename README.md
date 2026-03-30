# Federated Fraud Detection System

A privacy-preserving, federated learning system for bank fraud detection across isolated institutions. Banks collaboratively train a shared fraud detection model **without ever sharing raw transaction data**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CENTRAL SERVER                           │
│   Track 3: Federated Algorithms  ◄──►  Track 4: Infrastructure  │
│   (FedAvg / Custom Aggregation)         (FastAPI + WebSockets)  │
└──────────────────┬──────────────────────────────┬──────────────┘
                   │ weights up/down              │ weights up/down
        ┌──────────▼──────────┐        ┌──────────▼──────────┐
        │      BANK NODE A    │        │      BANK NODE B     │
        │  Track 1: Data Eng  │        │  Track 1: Data Eng   │
        │  Track 2: Local ML  │        │  Track 2: Local ML   │
        │  (Domestic Retail)  │        │  (Intl. Corporate)   │
        └─────────────────────┘        └──────────────────────┘
```

## Track Overview

| Track | Role | Key Responsibility |
|-------|------|--------------------|
| **Track 1** | Data Architect | Source, clean, and Non-IID skew the IEEE-CIS dataset into per-bank SQLite databases |
| **Track 2** | Local ML Engineer | Build the MLP, local training loop, and weight extraction for shared base layers |
| **Track 3** | Federated Research Scientist | Server-side aggregation: FedAvg baseline → reliability scoring → conflict-aware merging |
| **Track 4** | Platform Engineer | Docker orchestration, FastAPI transport layer, Next.js dashboard with live telemetry |

## Repository Structure

```
federated-fraud-detection/
├── track1_data_engineering/     ← Data Architect
├── track2_deep_learning/        ← Local ML Engineer
├── track3_federated_algorithms/ ← Federated Research Scientist
├── track4_infrastructure/       ← Platform Engineer
├── docs/                        ← Shared architecture diagrams & API contracts
├── scripts/                     ← Shared utility scripts
├── docker-compose.yml           ← Root-level orchestration (delegates to Track 4)
└── requirements.txt             ← Shared top-level deps
```

## Integration Points

- **Track 1 → Track 2**: DataLoaders feed directly into the PyTorch training loop. Coordinate on batch shape and label format.
- **Track 2 → Track 3**: Extracted weight dictionaries (JSON/pickle) are the payload. Agree on key naming conventions.
- **Track 3 → Track 4**: Aggregated global weights are returned via the FastAPI response. Agree on response schema.
- **Track 4 → All**: Docker networking and environment variables connect everything. All services register on the same Docker network.

## Getting Started

```bash
# Spin up the full system
docker-compose up --build

# Or run each track independently during development
cd track1_data_engineering && python src/preprocess.py
cd track2_deep_learning && python src/train.py
cd track3_federated_algorithms && python src/aggregator.py
cd track4_infrastructure && uvicorn api.main:app --reload
```

## Key Design Decisions

- **Non-IID data**: Each bank's data is deliberately skewed to simulate real-world distribution mismatch — the core challenge of federated learning.
- **Split model architecture**: Only the "shared base layers" travel over the network. "Personalized top layers" stay local forever.
- **Conflict-aware aggregation**: The server penalizes updates whose weights conflict (low cosine similarity) with the global consensus, going beyond naive averaging.
